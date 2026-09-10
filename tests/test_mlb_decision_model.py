import unittest
import tempfile
import zipfile
import json
from unittest.mock import patch
from pathlib import Path

from mlb_decision_model.decision import Pick, evaluate_market, rank_combinations
from mlb_decision_model.dashboard import analyze_payload, ocr_extract_payload
from mlb_decision_model.features import (
    DataQualityError,
    FEATURE_NAMES,
    FEATURE_VERSION,
    build_game_features,
    shrunk_bvp_average,
)
from mlb_decision_model.model import CalibratedLogisticModel
from mlb_decision_model.sources import SourcePolicyError, require_approved_source
from mlb_decision_model.retrosheet import EXPECTED_CSVS, inspect_bundle
from mlb_decision_model.kspo import parse_results
from mlb_decision_model.snapshot import validate_picks
from mlb_decision_model.ocr_extract import parse_draft_odds


class FeatureTests(unittest.TestCase):
    def test_small_bvp_is_shrunk(self):
        batter = {"woba_vs_r": 0.330, "bvp": {"ab": 3, "h": 2}}
        avg, reliability = shrunk_bvp_average(batter, "R")
        self.assertLess(avg, 0.40)
        self.assertLess(reliability, 0.10)

    def test_feature_direction(self):
        strong = {
            "starter": {"era": 2.5, "fip": 2.8, "k_rate": 0.30, "bb_rate": 0.05},
            "bullpen": [{"era": 2.8, "fip": 3.0, "availability": 1.0}],
            "closer": {"era": 2.2, "fip": 2.5, "availability": 1.0},
            "lineup": [{"woba_vs_r": 0.370}],
            "bench": [{"woba_vs_r": 0.340}],
        }
        weak = {
            "starter": {"era": 5.0, "fip": 4.8, "k_rate": 0.17, "bb_rate": 0.12},
            "bullpen": [{"era": 5.2, "fip": 5.0, "availability": 0.5}],
            "closer": {"era": 4.9, "fip": 4.8, "availability": 0.4},
            "lineup": [{"woba_vs_r": 0.285}],
            "bench": [{"woba_vs_r": 0.280}],
        }
        features = build_game_features({"home": strong, "away": weak})
        self.assertEqual(set(features), set(FEATURE_NAMES))
        self.assertGreater(features["starter_edge"], 0)
        self.assertGreater(features["bullpen_edge"], 0)
        self.assertGreater(features["lineup_edge"], 0)

    def test_missing_bullpen_is_not_scored_as_bad_performance(self):
        team = {
            "starter": {"era": 3.5},
            "bullpen": [],
            "lineup": [{"woba_vs_r": 0.320}],
        }
        with self.assertRaises(DataQualityError):
            build_game_features({"home": team, "away": team})


class ModelSchemaTests(unittest.TestCase):
    def test_saved_model_is_bound_to_v2_feature_contract(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "model.json"
            CalibratedLogisticModel().save(path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["feature_version"], FEATURE_VERSION)
            self.assertEqual(tuple(payload["feature_names"]), FEATURE_NAMES)
            loaded = CalibratedLogisticModel.load(path)
            self.assertEqual(loaded.feature_names, FEATURE_NAMES)

    def test_legacy_model_without_feature_version_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "legacy.json"
            path.write_text(
                json.dumps(
                    {
                        "feature_names": ["starter_edge", "closer_edge"],
                        "means": [0.0, 0.0],
                        "scales": [1.0, 1.0],
                        "weights": [0.0, 0.0],
                        "intercept": 0.0,
                        "calibration_a": 1.0,
                        "calibration_b": 0.0,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "incompatible feature schema"):
                CalibratedLogisticModel.load(path)


class SourcePolicyTests(unittest.TestCase):
    def test_open_bulk_source_is_allowed(self):
        policy = require_approved_source("retrosheet", automated=True, bulk=True)
        self.assertEqual(policy.status, "allowed")

    def test_mlb_personal_connection_is_allowed_but_bulk_is_blocked(self):
        self.assertTrue(require_approved_source("mlb_statsapi", automated=True).automated_allowed)
        with self.assertRaises(SourcePolicyError):
            require_approved_source("mlb_statsapi", bulk=True)

    def test_unknown_source_is_blocked_by_default(self):
        with self.assertRaises(SourcePolicyError):
            require_approved_source("random_github_csv", bulk=True)

    def test_kspo_delayed_results_source_is_allowed(self):
        policy = require_approved_source("data_go_kr_kspo_results", automated=True)
        self.assertFalse(policy.bulk_allowed)

    def test_retrosheet_bundle_requires_official_host_and_expected_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            archive = Path(temp_dir) / "retrosheet.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                for name in EXPECTED_CSVS:
                    bundle.writestr(name, "gameid,value\nTEST,1\n")
            manifest = inspect_bundle(
                archive, "https://www.retrosheet.org/downloads/example.zip"
            )
            self.assertEqual(manifest["source_id"], "retrosheet")
            with self.assertRaises(SourcePolicyError):
                inspect_bundle(archive, "https://example.com/copied-data.zip")


class DecisionTests(unittest.TestCase):
    def setUp(self):
        # Ranking unit tests must not depend on the user's live API cache.
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        for name in ("PREGAME_PATH", "MODEL_PROBABILITIES_PATH"):
            replacement = patch("mlb_decision_model.dashboard." + name, Path(directory.name) / name)
            replacement.start()
            self.addCleanup(replacement.stop)

    def test_opposite_side_is_evaluated(self):
        home, away = evaluate_market(0.44, 1.72, 2.15)
        self.assertEqual(home.action, "FADE")
        self.assertEqual(away.action, "BET")

    def test_combinations_rank_by_ev(self):
        picks = [
            Pick("A", 0.60, 1.90),
            Pick("B", 0.58, 1.95),
            Pick("C", 0.45, 2.00),
        ]
        ranked = rank_combinations(picks, legs=2, simulations=2000)
        self.assertEqual(set(ranked[0].picks), {"A", "B"})

    def test_combinations_can_rank_by_survival(self):
        picks = [
            Pick("safe-1", 0.72, 1.30),
            Pick("safe-2", 0.68, 1.35),
            Pick("value", 0.45, 2.60),
        ]
        ranked = rank_combinations(
            picks, legs=2, simulations=2000, sort_by="survival"
        )
        self.assertEqual(set(ranked[0].picks), {"safe-1", "safe-2"})

    def test_same_game_markets_are_not_combined(self):
        picks = [
            Pick("SEA 승", 0.60, 1.80, "SEA-PHI"),
            Pick("SEA-PHI 언더", 0.58, 1.85, "SEA-PHI"),
            Pick("SD 승", 0.57, 1.90, "SD-PIT"),
        ]
        ranked = rank_combinations(
            picks, legs=2, simulations=1000, sort_by="survival"
        )
        self.assertEqual(len(ranked), 2)
        self.assertTrue(all(set(row.picks) != {"SEA 승", "SEA-PHI 언더"} for row in ranked))

    def test_dashboard_returns_survival_and_ev_rankings(self):
        payload = {
            "legs": 2,
            "simulations": 1000,
            "top_n": None,
            "picks": [
                {"name": "A", "probability": 0.65, "odds": 1.7},
                {"name": "B", "probability": 0.60, "odds": 1.8},
                {"name": "C", "probability": 0.45, "odds": 2.5},
            ],
        }
        result = analyze_payload(payload)
        self.assertEqual(set(result["survival_rank"][0]["picks"]), {"A", "B"})
        self.assertEqual(result["assumption"], "independent-picks")

    def test_ocr_extract_payload_requires_image(self):
        with self.assertRaises(ValueError):
            ocr_extract_payload({})

    def test_ocr_extract_payload_returns_drafts_for_confirmation(self):
        import base64
        from unittest.mock import patch
        from mlb_decision_model.ocr_extract import DraftOdds

        tiny_png = base64.b64encode(
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00"
            b"\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc"
            b"\xf8\xcf\xc0\x00\x00\x03\x01\x01\x00\x18\xdd\x8d\xb0\x00\x00"
            b"\x00\x00IEND\xaeB`\x82"
        ).decode()
        fake_drafts = [DraftOdds("moneyline", "승", 1.90, "패", 1.64, "ok")]
        with patch("mlb_decision_model.dashboard.extract_odds", return_value=fake_drafts):
            result = ocr_extract_payload({"image_data_url": f"data:image/png;base64,{tiny_png}"})
        self.assertEqual(result["drafts"][0]["odds_a"], 1.90)
        self.assertIn("초안", result["note"])

    def test_dashboard_returns_all_combinations_when_top_n_not_set(self):
        picks = [
            {"event_id": str(i), "name": f"pick{i}", "probability": 0.5, "odds": 1.9}
            for i in range(5)
        ]
        result = analyze_payload({"legs": 2, "simulations": 500, "picks": picks})
        self.assertEqual(len(result["survival_rank"]), 10)
        self.assertEqual(len(result["ev_rank"]), 10)

    def test_top_five_preserves_global_worst_and_full_candidate_count(self):
        picks = [{"event_id":event,"name":f"{event}-{i}","probability":.8-i*.1,"odds":1.8+i*.2}
                 for event in ("A","B") for i in range(3)]
        result = analyze_payload({"legs":2,"top_n":5,"picks":picks})
        self.assertEqual(result["combination_count"],9)
        self.assertEqual(len(result["survival_rank"]),5)
        self.assertAlmostEqual(result["worst_survival"]["hit_probability"],.36)
        self.assertLess(result["worst_survival"]["hit_probability"],result["survival_rank"][-1]["hit_probability"])


class InputTests(unittest.TestCase):
    def test_snapshot_pick_validation(self):
        picks = validate_picks(
            [
                {"name": "SEA 승", "probability": 0.61, "odds": 1.72},
                {"name": "SD 승", "probability": 0.58, "odds": 1.80},
            ]
        )
        self.assertEqual(len(picks), 2)

    def test_kspo_response_parser(self):
        payload = {
            "header": {"resultCode": "0", "resultMsg": "NORMAL SERVICE"},
            "body": {
                "items": {
                    "item": {
                        "match_ymd": "20260801",
                        "match_tm": "1830",
                        "hteam_han_nm": "홈",
                        "ateam_han_nm": "원정",
                        "leag_han_nm": "MLB",
                        "stdm_han_nm": "구장",
                        "match_sport_han_nm": "야구",
                        "match_end_val": "홈승",
                        "obj_prod_nm": "프로토",
                    }
                }
            },
        }
        results = parse_results(payload)
        self.assertEqual(results[0].home_team, "홈")
        self.assertEqual(results[0].result, "홈승")


class OcrExtractTests(unittest.TestCase):
    def test_sum_never_replaces_moneyline_after_lost_decimals(self):
        text = "필라델피아 필리스 vs 애틀랜타 브레이브스\n야구 승패\n승 패\n2374 141\n야구 SUM\n홀 짝\n1.58 2.09"
        self.assertNotIn("moneyline", [d.market for d in parse_draft_odds(text)])

    def test_mixed_decimal_loss_never_promotes_later_market(self):
        text = "야구 승패\n승 패\n2374 1.41\n야구 SUM\n홀 짝\n1.58 2.09"
        self.assertNotIn("moneyline", [d.market for d in parse_draft_odds(text)])

    def test_total_only_capture_has_no_moneyline(self):
        text = "야구 언더오버 U/O 7.5\n언더 오버\n1.78 1.74"
        self.assertNotIn("moneyline", [d.market for d in parse_draft_odds(text)])
    REAL_OCR_SEA_PHI = (
        "08.27(목)\n05:10\n\n08.26 수\n23:00 마감\n\n® BE ue\n\n6731   ae\n"
        "야구 승패\n조합\n\n022 아구승1패\n조합\n\n6733\n\n6734\n\n6735\n\n"
        "야구 핸디캡 1+2.5\n\n조합\n야구 언더오버 4/0 7.5\n\n조합\n야구 SUM\n\n"
        "시애들 매리너스 필라델피아 필리스\n\n"
        "층          패\n1.90         1.64\n"
        "é             패\n2.85     3.      2.08\n"
        "é          패\n1.29         277\n"
        "언더         오버\n1.61         1.94\n"
        "=          a\n1.58         2.09\n"
    )

    REAL_OCR_SD_PIT = (
        "08.27(목)\n05:10\n\n6736 조합\n야구 승패\n\n야구승1패\n조합\n\n"
        "6738\n\n6739\n\n6740\n\n야구 핸디캡 H-2.5\n\n조합\n"
        "야구 언더오버 U/O 8.5\n\n조합\n야구 SUM\n\n"
        "샌디에이고 파드리스 피츠버그 파이어리츠\n\n"
        "Fy          패\n1.65           1.89\n"
        "충      1      패\n2.33         3.30         2.40\n"
        "층          패\n3.23               1.21\n"
        "언더         오버\n1.72           1.80\n"
        "=          짝\n1.59           2.07\n"
    )

    def test_extracts_correct_moneyline_and_total_despite_garbled_labels(self):
        results = {d.market: d for d in parse_draft_odds(self.REAL_OCR_SEA_PHI)}
        self.assertEqual((results["moneyline"].odds_a, results["moneyline"].odds_b), (1.90, 1.64))
        self.assertEqual((results["total"].odds_a, results["total"].odds_b), (1.61, 1.94))
        self.assertEqual(results["moneyline"].confidence, "ok")
        self.assertEqual(results["total"].confidence, "ok")

    def test_does_not_confuse_total_header_line_with_total_label_line(self):
        results = {d.market: d for d in parse_draft_odds(self.REAL_OCR_SEA_PHI)}
        self.assertIsNotNone(results["total"].odds_a)
        self.assertIsNotNone(results["total"].odds_b)

    def test_second_real_capture_also_parses_correctly(self):
        results = {d.market: d for d in parse_draft_odds(self.REAL_OCR_SD_PIT)}
        self.assertEqual((results["moneyline"].odds_a, results["moneyline"].odds_b), (1.65, 1.89))
        self.assertEqual((results["total"].odds_a, results["total"].odds_b), (1.72, 1.80))

    def test_missing_total_header_yields_no_total_entry(self):
        text_without_total = "야구 승패\n조합\n\n팀A 팀B\n\n승   패\n1.50   2.50\n"
        results = parse_draft_odds(text_without_total)
        markets = {d.market for d in results}
        self.assertIn("moneyline", markets)
        self.assertNotIn("total", markets)


if __name__ == "__main__":
    unittest.main()
