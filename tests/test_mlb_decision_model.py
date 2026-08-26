import unittest
import tempfile
import zipfile
from pathlib import Path

from mlb_decision_model.decision import Pick, evaluate_market, rank_combinations
from mlb_decision_model.dashboard import analyze_payload, ocr_extract_payload
from mlb_decision_model.features import (
    DataQualityError,
    build_game_features,
    shrunk_bvp_average,
)
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


class SourcePolicyTests(unittest.TestCase):
    def test_open_bulk_source_is_allowed(self):
        policy = require_approved_source("retrosheet", automated=True, bulk=True)
        self.assertEqual(policy.status, "allowed")

    def test_mlb_automated_collection_is_blocked(self):
        with self.assertRaises(SourcePolicyError):
            require_approved_source("mlb_statsapi", automated=True)

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

        # 1x1 PNG (실제 내용은 안 중요함 - extract_odds 자체를 모킹해서
        # tesseract 바이너리 없이도 이 경로를 검증한다)
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

    def test_dashboard_defaults_to_top_5_combinations(self):
        picks = [
            {"event_id": str(i), "name": f"pick{i}", "probability": 0.5, "odds": 1.9}
            for i in range(5)
        ]
        result = analyze_payload({"legs": 2, "simulations": 500, "picks": picks})
        # C(5,2) = 10개 조합이 가능하지만 기본 top_n=5로 잘려야 한다
        self.assertEqual(len(result["survival_rank"]), 5)
        self.assertEqual(len(result["ev_rank"]), 5)


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
    """실제 프로토 캡처 2장을 Tesseract(kor+eng)로 돌려서 나온 원본 텍스트를
    그대로 고정한 테스트다. 이미지 파일이나 tesseract 바이너리 없이도 파서
    로직만 재현 가능하게 검증한다. 라벨(승/홀 등)이 OCR로 깨지는 실제 사례가
    포함되어 있다 - 그래서 라벨이 아니라 줄 순서로 파싱한다."""

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
        # "야구 언더오버 4/0 7.5" 헤더 줄도 "언더"를 포함하므로, 진짜 라벨 줄
        # ("언더  오버")과 헷갈리면 헤더 다음의 빈 줄에서 숫자를 못 찾는다.
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
