import json
import tempfile
import unittest
from pathlib import Path

from mlb_decision_model.model_probability import (
    ModelProbabilityUnavailable,
    attach_model_probabilities,
    attach_score_model_probabilities,
    model_probability_status,
)
from mlb_decision_model.features import FEATURE_NAMES
from mlb_decision_model.score_model import TeamScoreModel


class ModelProbabilityJoinTests(unittest.TestCase):
    def test_expired_projection_is_not_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"predictions.json"
            path.write_text(json.dumps({"expires_at":"2000-01-01T00:00:00+00:00","predictions":[]}),encoding="utf-8")
            self.assertEqual(model_probability_status(path).status,"not_ready")
    def test_missing_projection_never_falls_back_to_bookmaker_probability(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "model_probabilities.json"
            picks = [
                {
                    "event_id": "SEA vs PHI",
                    "period": "full",
                    "market": "total",
                    "selection": "언더",
                    "line": 7.5,
                    "name": "SEA vs PHI · 언더 7.5",
                    "odds": 1.78,
                }
            ]
            with self.assertRaises(ModelProbabilityUnavailable):
                attach_model_probabilities(picks, missing)
            self.assertEqual(model_probability_status(missing).status, "not_ready")

    def test_projection_is_joined_by_event_period_market_selection_and_line(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model_probabilities.json"
            path.write_text(
                json.dumps(
                    {
                        "model_version": "score-v1",
                        "generated_at": "2026-09-04T12:00:00+09:00",
                        "predictions": [
                            {
                                "event_id": "SEA vs PHI",
                                "period": "full",
                                "market": "total",
                                "selection": "언더",
                                "line": 7.5,
                                "probability": 0.584,
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            rows, status = attach_model_probabilities(
                [
                    {
                        "event_id": " sea  VS  phi ",
                        "period": "full",
                        "market": "total",
                        "selection": "언더",
                        "line": "7.5",
                        "name": "SEA vs PHI · 언더 7.5",
                        "odds": 1.78,
                    }
                ],
                path,
            )
            self.assertEqual(rows[0]["probability"], 0.584)
            self.assertEqual(rows[0]["probability_source"], "model")
            self.assertEqual(status.model_version, "score-v1")

    def test_score_model_baseline_supports_all_full_markets_without_odds_input(self):
        rows = [
            {name: (i % 5 - 2 if name == "starter_edge" else 0.0) for name in FEATURE_NAMES}
            for i in range(25)
        ]
        model = TeamScoreModel().fit(rows, [3 + i % 4 for i in range(25)], [2 + i % 5 for i in range(25)])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "score.json"
            model.save(path)
            picks = [
                {"event_id":"A vs B","period":"full","market":"moneyline","selection":"승","line":None,"name":"승","odds":1.8},
                {"event_id":"A vs B","period":"full","market":"three_way","selection":"1","line":None,"name":"1","odds":3.2},
                {"event_id":"A vs B","period":"full","market":"handicap","selection":"패","line":-1.5,"name":"핸디 패","odds":1.9},
                {"event_id":"A vs B","period":"full","market":"total","selection":"오버","line":7.5,"name":"오버","odds":1.9},
                {"event_id":"A vs B","period":"full","market":"sum","selection":"짝","line":None,"name":"짝","odds":2.0},
            ]
            resolved, status = attach_score_model_probabilities(picks, path)
            self.assertEqual(status.status, "ready_baseline")
            self.assertTrue(all(0 < row["probability"] < 1 for row in resolved))
            self.assertTrue(all(row["probability_source"] == "score_model_neutral_baseline" for row in resolved))


if __name__ == "__main__":
    unittest.main()
