import unittest
import tempfile
import math
from pathlib import Path

from mlb_decision_model.score_model import (
    TeamScoreModel, PoissonRegression, _nb_pmf, _poisson_pmf,
    estimate_dispersion, handicap_probability, moneyline_probability,
    sum_probability, three_way_probability, total_probability,
    win_one_loss_probability,
)
from mlb_decision_model.train_score_model import read_score_training_csv
from mlb_decision_model.predict_today import build_predictions


class PoissonMathTests(unittest.TestCase):
    def test_pmf_matches_manual_formula(self):
        lam = 3.0
        manual = math.exp(-lam) * lam**2 / math.factorial(2)
        self.assertAlmostEqual(_poisson_pmf(2, lam), manual, places=9)

    def test_pmf_sums_to_one(self):
        total = sum(_poisson_pmf(k, 3.0) for k in range(50))
        self.assertAlmostEqual(total, 1.0, places=6)

    def test_zero_lambda_puts_all_mass_at_zero(self):
        self.assertEqual(_poisson_pmf(0, 0.0), 1.0)
        self.assertEqual(_poisson_pmf(1, 0.0), 0.0)


class NegativeBinomialMathTests(unittest.TestCase):
    def test_pmf_sums_to_one_and_converges_to_poisson(self):
        self.assertAlmostEqual(sum(_nb_pmf(k, 4.5, 3.0) for k in range(80)), 1.0, places=7)
        difference = sum(
            abs(_nb_pmf(k, 4.0, 1e6) - _poisson_pmf(k, 4.0))
            for k in range(40)
        )
        self.assertLess(difference, 0.001)

    def test_dispersion_estimation_detects_overdispersion(self):
        actual = [0, 1, 2, 8, 9, 10, 3, 4, 5, 7] * 20
        self.assertLess(estimate_dispersion(actual, [4.0] * len(actual)), 1e6)


class MarketProbabilityTests(unittest.TestCase):
    def test_integer_lines_and_invalid_selections_are_rejected(self):
        grid = [[.25,.25],[.25,.25]]
        for function, selection in ((total_probability,"언더"),(handicap_probability,"승")):
            with self.assertRaises(ValueError):
                function(grid, selection, 2)
        with self.assertRaises(ValueError):
            moneyline_probability(grid,"무")

    def test_high_scoring_grid_retains_probability_mass(self):
        model = TeamScoreModel()
        model.home.intercept = math.log(22)
        model.away.intercept = math.log(18)
        grid = model.score_grid({})
        self.assertGreater(len(grid), 16)
        self.assertAlmostEqual(sum(map(sum,grid)),1,places=9)
        self.assertAlmostEqual(total_probability(grid,"언더",40.5)+total_probability(grid,"오버",40.5),1,places=9)
    def test_symmetric_grid_gives_fifty_fifty_moneyline(self):
        grid = [[1 / 9] * 3 for _ in range(3)]
        self.assertAlmostEqual(moneyline_probability(grid, "승"), 0.5, places=9)

    def test_certain_home_win_grid(self):
        grid = [[0.0] * 10 for _ in range(10)]
        grid[0][5] = 1.0  # away=0, home=5
        self.assertEqual(moneyline_probability(grid, "승"), 1.0)
        self.assertEqual(total_probability(grid, "언더", 7.5), 1.0)
        self.assertEqual(total_probability(grid, "오버", 7.5), 0.0)
        self.assertEqual(handicap_probability(grid, "승", -3.5), 1.0)
        self.assertEqual(handicap_probability(grid, "승", -6.5), 0.0)

    def test_each_market_has_its_own_settlement_formula(self):
        grid = [[0.0] * 5 for _ in range(5)]
        grid[1][3] = 0.4  # 홈 2점 차 승, 합계 짝
        grid[2][3] = 0.3  # 홈 1점 차 승, 합계 홀
        grid[2][2] = 0.2  # 동점, 합계 짝
        grid[4][1] = 0.1  # 홈 3점 차 패, 합계 홀
        self.assertAlmostEqual(win_one_loss_probability(grid, "승"), 0.4)
        self.assertAlmostEqual(win_one_loss_probability(grid, "1"), 0.5)
        self.assertAlmostEqual(win_one_loss_probability(grid, "패"), 0.1)
        self.assertAlmostEqual(three_way_probability(grid, "승"), 0.7)
        self.assertAlmostEqual(three_way_probability(grid, "무"), 0.2)
        self.assertAlmostEqual(three_way_probability(grid, "패"), 0.1)
        self.assertAlmostEqual(sum_probability(grid, "홀"), 0.4)
        self.assertAlmostEqual(sum_probability(grid, "짝"), 0.6)


class PoissonRegressionTests(unittest.TestCase):
    def test_learns_correct_sign_of_relationship(self):
        # starter_edge가 클수록 득점이 느는 관계를 심어둔 합성 데이터
        rows, runs = [], []
        for i in range(60):
            edge = (i % 21 - 10) / 5.0
            rows.append({"starter_edge": edge, "bullpen_edge": 0.0, "lineup_edge": 0.0,
                         "team_matchup_edge": 0.0, "bvp_edge": 0.0, "availability_edge": 0.0,
                         "defense_edge": 0.0, "weather_edge": 0.0, "rest_edge": 0.0})
            runs.append(max(0, round(4.0 + edge)))
        model = PoissonRegression().fit(rows, runs, epochs=1000)
        self.assertGreater(model.weights[0], 0)  # starter_edge -> 양의 가중치

    def test_at_least_20_games_required(self):
        with self.assertRaises(ValueError):
            PoissonRegression().fit([{}] * 5, [1] * 5)


class TeamScoreModelSaveLoadTests(unittest.TestCase):
    def test_round_trip_preserves_predictions(self):
        rows = [{"starter_edge": (i % 5) - 2, "bullpen_edge": 0.0, "lineup_edge": 0.0,
                 "team_matchup_edge": 0.0, "bvp_edge": 0.0, "availability_edge": 0.0,
                 "defense_edge": 0.0, "weather_edge": 0.0, "rest_edge": 0.0} for i in range(25)]
        home_runs = [max(0, 4 + (i % 5) - 2) for i in range(25)]
        away_runs = [max(0, 4 - (i % 5) + 2) for i in range(25)]
        model = TeamScoreModel().fit(rows, home_runs, away_runs)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "score.json"
            model.save(path)
            loaded = TeamScoreModel.load(path)
            grid_before = model.score_grid(rows[0])
            grid_after = loaded.score_grid(rows[0])
            self.assertAlmostEqual(grid_before[0][0], grid_after[0][0], places=9)

    def test_rejects_wrong_feature_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text('{"feature_version": "v1.0", "home": {}, "away": {}}', encoding="utf-8")
            with self.assertRaises(ValueError):
                TeamScoreModel.load(path)


class ReadScoreTrainingCsvTests(unittest.TestCase):
    def test_skips_rows_without_f5_labels_when_segment_is_f5(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.csv"
            header = ("feature_version,starter_edge,bullpen_edge,lineup_edge,team_matchup_edge,"
                       "bvp_edge,availability_edge,defense_edge,weather_edge,rest_edge,"
                       "home_runs,away_runs,home_f5_runs,away_f5_runs\n")
            rows = header
            for i in range(25):
                f5 = "2,1" if i % 2 == 0 else ","  # 절반은 F5 라벨 없음
                rows += f"v2.0,0,0,0,0,0,0,0,0,0,4,3,{f5}\n"
            path.write_text(rows, encoding="utf-8")
            full_rows, _, _ = read_score_training_csv(path, segment="full")
            f5_rows, _, _ = read_score_training_csv(path, segment="f5")
            self.assertEqual(len(full_rows), 25)
            self.assertEqual(len(f5_rows), 13)  # i=0,2,4,...,24 -> 13개


class BuildPredictionsTests(unittest.TestCase):
    def test_missing_features_are_rejected(self):
        with self.assertRaises(ValueError):
            build_predictions([{"event_id":"A","features":{}}], TeamScoreModel())
    def test_produces_expected_market_rows(self):
        rows = [{"starter_edge": (i % 5) - 2, "bullpen_edge": 0.0, "lineup_edge": 0.0,
                 "team_matchup_edge": 0.0, "bvp_edge": 0.0, "availability_edge": 0.0,
                 "defense_edge": 0.0, "weather_edge": 0.0, "rest_edge": 0.0} for i in range(25)]
        home_runs = [max(0, 4 + (i % 5) - 2) for i in range(25)]
        away_runs = [max(0, 4 - (i % 5) + 2) for i in range(25)]
        model = TeamScoreModel().fit(rows, home_runs, away_runs)
        games = [{"event_id": "A vs B", "features": rows[0], "total_lines": [7.5], "handicap_lines": [-1.5]}]
        predictions = build_predictions(games, model)
        markets = {(p["market"], p["selection"], p["line"]) for p in predictions}
        self.assertIn(("moneyline", "승", None), markets)
        self.assertIn(("moneyline", "패", None), markets)
        self.assertIn(("total", "언더", 7.5), markets)
        self.assertIn(("total", "오버", 7.5), markets)
        self.assertIn(("handicap", "승", -1.5), markets)
        self.assertIn(("handicap", "패", -1.5), markets)
        self.assertIn(("three_way", "1", None), markets)
        self.assertIn(("sum", "홀", None), markets)

    def test_f5_model_produces_three_way_market(self):
        model = TeamScoreModel(segment="f5")
        game = {"event_id": "A vs B", "features": {name: 0.0 for name in model.feature_names}}
        predictions = build_predictions([game], model)
        self.assertEqual({row["period"] for row in predictions}, {"first_five"})
        self.assertEqual(
            {row["selection"] for row in predictions if row["market"] == "three_way"},
            {"승", "무", "패"},
        )


if __name__ == "__main__":
    unittest.main()
