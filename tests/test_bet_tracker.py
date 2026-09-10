import unittest
import tempfile
from pathlib import Path

from mlb_decision_model.bet_tracker import BetLeg, append_legs, load_legs, record_outcome, summarize


class BetLegTests(unittest.TestCase):
    def test_derived_fields_computed_correctly(self):
        leg = BetLeg("2026-09-02", "T1", 1, "NYY@LAA", "total", "언더7.5", 1.79, 0.55, "선발폼")
        self.assertAlmostEqual(leg.break_even, 1 / 1.79, places=4)
        self.assertAlmostEqual(leg.edge, 0.55 - 1 / 1.79, places=4)
        self.assertAlmostEqual(leg.ev, 0.55 * 1.79 - 1, places=4)

    def test_missing_odds_does_not_crash_and_returns_none(self):
        # 배당을 아직 모를 때(예: 확률 분석만 먼저 기록) odds=0으로 두면
        # 0으로 나누기 에러가 아니라 None을 반환해야 한다.
        leg = BetLeg("2026-09-03", "PENDING", 1, "NYY@LAA", "total", "언더7.5", 0.0, 0.57, "선발폼")
        self.assertIsNone(leg.break_even)
        self.assertIsNone(leg.edge)
        self.assertIsNone(leg.ev)
        row = leg.to_row()  # append_legs()가 쓰는 경로도 안 깨지는지 확인
        self.assertIsNone(row["break_even"])


class TrackerPersistenceTests(unittest.TestCase):
    def test_append_and_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bets.csv"
            leg = BetLeg("2026-09-02", "T1", 1, "NYY@LAA", "total", "언더7.5", 1.79, 0.55, "선발폼")
            append_legs([leg], path)
            rows = load_legs(path)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["pick"], "언더7.5")
            self.assertEqual(rows[0]["actual_outcome"], "")

    def test_record_outcome_updates_matching_leg_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bets.csv"
            append_legs([
                BetLeg("2026-09-02", "T1", 1, "NYY@LAA", "total", "언더", 1.79, 0.55, "선발폼"),
                BetLeg("2026-09-02", "T1", 2, "ARI@PHI", "total", "언더", 1.79, 0.52, "팀흐름"),
            ], path)
            found = record_outcome("T1", 1, "LOSS", "9-5", path)
            self.assertTrue(found)
            rows = load_legs(path)
            self.assertEqual(rows[0]["actual_outcome"], "LOSS")
            self.assertEqual(rows[1]["actual_outcome"], "")  # 다른 다리는 안 건드림

    def test_record_outcome_returns_false_when_not_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bets.csv"
            append_legs([BetLeg("2026-09-02", "T1", 1, "G", "total", "픽", 1.5, 0.5, "")], path)
            self.assertFalse(record_outcome("T99", 1, "WIN", path=path))


class SummaryTests(unittest.TestCase):
    def test_empty_log_returns_zero_legs(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary = summarize(Path(tmp) / "empty.csv")
            self.assertEqual(summary["total_legs"], 0)

    def test_summary_breaks_down_by_factor(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bets.csv"
            append_legs([
                BetLeg("2026-09-02", "T1", 1, "G1", "total", "픽1", 1.79, 0.55, "불펜,선발폼"),
                BetLeg("2026-09-02", "T2", 1, "G2", "total", "픽2", 1.79, 0.52, "선발폼"),
            ], path)
            record_outcome("T1", 1, "WIN", path=path)
            record_outcome("T2", 1, "LOSS", path=path)
            summary = summarize(path)
            self.assertEqual(summary["total_legs"], 2)
            self.assertEqual(summary["win_rate"], 0.5)
            # "선발폼"은 두 다리 다 포함(1승1패=50%), "불펜"은 1다리만(1승=100%)
            self.assertEqual(summary["by_factor"]["선발폼"]["win_rate"], 0.5)
            self.assertEqual(summary["by_factor"]["불펜"]["win_rate"], 1.0)
            self.assertEqual(summary["by_factor"]["불펜"]["sample"], 1)

    def test_pending_outcomes_are_excluded_from_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bets.csv"
            append_legs([BetLeg("2026-09-02", "T1", 1, "G1", "total", "픽1", 1.79, 0.55, "")], path)
            # 결과 아직 안 채움
            summary = summarize(path)
            self.assertEqual(summary["total_legs"], 0)


if __name__ == "__main__":
    unittest.main()
