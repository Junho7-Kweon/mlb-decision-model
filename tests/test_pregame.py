import copy
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from mlb_decision_model.pregame import build_features, read_games, attach_pregame_probabilities
from mlb_decision_model.model_probability import ModelProbabilityUnavailable
from mlb_decision_model.score_model import TeamScoreModel
from mlb_decision_model.daily_data import Client, DailyDataUnavailable, observations

NOW = datetime(2026, 9, 9, 8, tzinfo=timezone.utc)


def game():
    team = dict(starter_id=1, starter_era=2, bullpen_era=3, team_slg=.5,
                errors_per_game=.4, days_since_last_game=1, wins=70, losses=60)
    return dict(event_id="bdl:1", event_aliases=["홈팀 vs 원정팀"], source_id="user_input", source_url="local:verified",
                source_version="test", starts_at="2026-09-09T12:00:00+00:00", as_of=NOW.isoformat(), retrieved_at=NOW.isoformat(),
                status="scheduled", feature_contract="retrosheet_decay_v2", home=team,
                away={**team, "starter_id": 2, "starter_era": 6, "bullpen_era": 5, "team_slg": .3, "errors_per_game": .9},
                h2h_home_wins=3, h2h_away_wins=2)


class PregameTests(unittest.TestCase):
    def test_training_feature_scale(self):
        f = build_features(game())
        self.assertEqual(f["starter_edge"], 1)
        self.assertEqual(f["bullpen_edge"], .5)
        self.assertEqual(f["lineup_edge"], 2)
        self.assertEqual(f["defense_edge"], 1)
        self.assertEqual(f["weather_edge"], 0)

    def test_missing_does_not_become_zero(self):
        g = game()
        del g["home"]["starter_era"]
        with self.assertRaises(ValueError):
            build_features(g)

    def test_stale_future_and_started_snapshots(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "daily.json"
            for fields, invalid in (({"retrieved_at": "2026-09-09T09:00:00+00:00"}, True),
                                    ({"starts_at": "2026-09-09T08:00:00+00:00", "as_of": "2026-09-09T07:00:00+00:00"}, False),
                                    ({"retrieved_at": "2026-09-08T22:00:00+00:00", "as_of": "2026-09-08T22:00:00+00:00"}, False)):
                path.write_text(json.dumps({"games": [{**game(), **fields}]}), encoding="utf-8")
                if invalid:
                    with self.assertRaises(ModelProbabilityUnavailable):
                        read_games(path, NOW)
                else:
                    self.assertEqual(read_games(path, NOW), [])

    def test_recent_mlb_warmup_snapshot_survives_nominal_start_only_briefly(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "daily.json"
            warmup = {**game(), "starts_at": "2026-09-09T07:55:00+00:00",
                      "as_of": "2026-09-09T08:00:00+00:00",
                      "retrieved_at": "2026-09-09T08:00:00+00:00",
                      "source_abstract_game_state": "Preview",
                      "source_detailed_state": "Warmup"}
            path.write_text(json.dumps({"games": [warmup]}), encoding="utf-8")
            self.assertEqual(len(read_games(path, NOW)), 1)
            self.assertEqual(read_games(path, datetime(2026, 9, 9, 8, 11, tzinfo=timezone.utc)), [])

            warmup["source_abstract_game_state"] = "Live"
            path.write_text(json.dumps({"games": [warmup]}), encoding="utf-8")
            self.assertEqual(len(read_games(path, NOW)), 1)

    def test_team_features_change_probabilities_not_odds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = TeamScoreModel()
            model.home.weights[0] = .3
            model.away.weights[0] = -.3
            model.save(root / "model.json")
            first = game()
            second = copy.deepcopy(first)
            second.update(event_id="bdl:2", event_aliases=["다른팀 vs 상대팀"])
            second["home"]["starter_era"], second["away"]["starter_era"] = 6, 2
            path = root / "daily.json"
            path.write_text(json.dumps({"games": [first, second]}), encoding="utf-8")
            picks = [dict(event_id=g["event_aliases"][0], period="full", market="moneyline", selection="승", odds=1.5, name=g["event_id"]) for g in (first, second)]
            rows, _ = attach_pregame_probabilities(picks, path, root / "model.json", None, NOW)
            self.assertGreater(rows[0]["probability"], rows[1]["probability"])
            picks[0]["odds"] = 20
            changed, _ = attach_pregame_probabilities(picks, path, root / "model.json", None, NOW)
            self.assertEqual(rows[0]["probability"], changed[0]["probability"])
            self.assertEqual(rows[0]["event_id"], "bdl:1")
            self.assertIn("features", rows[0]["feature_snapshot"])
            second["event_aliases"] = first["event_aliases"]
            path.write_text(json.dumps({"games": [first, second]}), encoding="utf-8")
            with self.assertRaises(ModelProbabilityUnavailable):
                attach_pregame_probabilities(picks, path, root / "model.json", None, NOW)

    def test_no_key_means_no_network(self):
        with patch.dict("os.environ", {}, clear=True), patch("mlb_decision_model.daily_data.urlopen") as network:
            with self.assertRaises(DailyDataUnavailable):
                Client()
            network.assert_not_called()

    def test_dashboard_never_uses_neutral_fallback(self):
        from mlb_decision_model import dashboard
        with tempfile.TemporaryDirectory() as directory, patch.object(dashboard, "PREGAME_PATH", Path(directory)/"none"), patch.object(dashboard, "MODEL_PROBABILITIES_PATH", Path(directory)/"none2"):
            with self.assertRaises(ModelProbabilityUnavailable):
                dashboard.resolve_model_picks([{"event_id": "A", "market": "moneyline", "selection": "승", "odds": 1.8}])

    def test_box_score_replay_and_missing_detail(self):
        h = dict(id=1, abbreviation="LAD", display_name="Los Angeles Dodgers")
        a = dict(id=2, abbreviation="CIN", display_name="Cincinnati Reds")
        target = dict(id=10, home_team=h, away_team=a, date="2026-09-09T12:00:00Z", status_state="scheduled")
        old = dict(id=9, home_team=h, away_team=a, date="2026-09-08T00:00:00Z", status_state="final",
                   home_team_data=dict(runs=4, errors=0), away_team_data=dict(runs=2, errors=1))
        stats, lineup = [], []
        for team in (h, a):
            player = team["id"] * 10
            lineup.append(dict(game_id=10, team=team, player=dict(id=player), is_probable_pitcher=True))
            for offset in (0, 1):
                stats.append(dict(game_id=9, team=team, player=dict(id=player+offset), pitching_outs=18 if offset==0 else 9, er=1,
                                  games_started=1-offset, at_bats=10, hits=3, doubles=1, triples=0, hr=0))
        result = observations(target, [old], stats, lineup, NOW)
        self.assertEqual(result["home"]["starter_era"], 1.5)
        self.assertEqual(result["home"]["bullpen_era"], 3)
        self.assertEqual(result["home"]["team_slg"], .4)
        self.assertEqual(result["h2h_home_wins"], 1)
        stats[0]["pitching_outs"] = None
        stats[0]["ip"] = "6.0"
        with self.assertRaises(DailyDataUnavailable):
            observations(target, [old], stats, lineup, NOW)


if __name__ == "__main__":
    unittest.main()
