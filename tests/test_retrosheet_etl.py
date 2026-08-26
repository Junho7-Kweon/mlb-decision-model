import unittest

from mlb_decision_model.retrosheet_etl import (
    LEAGUE_AVERAGE_ERA,
    PitcherForm,
    build_training_rows,
)


def _game(gid, date, home, away, wteam, lteam):
    return {"gid": gid, "date": date, "hometeam": home, "visteam": away,
            "wteam": wteam, "lteam": lteam}


def _pitch(gid, pid, team, p_seq, ipouts, er, bb=0, k=0, bfp=None):
    return {"gid": gid, "id": pid, "team": team, "p_seq": str(p_seq),
            "p_ipouts": str(ipouts), "p_er": str(er), "p_w": str(bb),
            "p_k": str(k), "p_bfp": str(bfp if bfp is not None else ipouts + bb + k)}


class PitcherFormTests(unittest.TestCase):
    def test_no_sample_returns_league_average(self):
        self.assertEqual(PitcherForm().era(), LEAGUE_AVERAGE_ERA)

    def test_era_computed_correctly(self):
        form = PitcherForm()
        form.add(ipouts=18, er=1, bb=1, k=6, bfp=24)  # 6 IP, 1자책
        self.assertAlmostEqual(form.era(), 1.5)  # 1*27/18


class BuildTrainingRowsTests(unittest.TestCase):
    """실제 Retrosheet 컬럼 스키마(csvcontents.html 확인됨)와 동일한 형태의
    합성 데이터로 시점 안전성(point-in-time safety)을 검증한다."""

    def test_first_start_has_no_leakage_uses_league_average(self):
        games = [_game("G1", "20240401", "TMA", "TMB", "TMA", "TMB")]
        pitching = [
            _pitch("G1", "PIT001", "TMA", 1, ipouts=18, er=1),
            _pitch("G1", "PITB01", "TMB", 1, ipouts=15, er=4),
        ]
        rows = build_training_rows(games, pitching, "20240101", "20241231")
        self.assertEqual(len(rows), 1)
        # 첫 경기이므로 두 선발 다 아직 표본이 없어 starter_edge가 0이어야 한다
        self.assertEqual(rows[0]["starter_edge"], 0.0)
        self.assertEqual(rows[0]["home_win"], 1)

    def test_second_start_reflects_only_prior_game_not_current_or_future(self):
        games = [
            _game("G1", "20240401", "TMA", "TMB", "TMA", "TMB"),
            _game("G2", "20240406", "TMB", "TMA", "TMB", "TMA"),  # TMA는 원정
        ]
        pitching = [
            _pitch("G1", "PIT001", "TMA", 1, ipouts=18, er=1),   # G1: ERA 1.50
            _pitch("G1", "PITB01", "TMB", 1, ipouts=15, er=4),
            _pitch("G2", "PIT001", "TMA", 1, ipouts=15, er=6),   # G2에서는 부진(누수 검증용) - 이 값이 G2 피처에 영향 주면 안 됨
            _pitch("G2", "PITB02", "TMB", 1, ipouts=18, er=2),
        ]
        rows = build_training_rows(games, pitching, "20240101", "20241231")
        self.assertEqual(len(rows), 2)
        g2 = rows[1]
        self.assertEqual(g2["_gid"], "G2")
        # G2 시점에 PIT001(원정, away)의 사전 ERA는 G1 성적(1.50)만 반영돼야 하고,
        # G2 자신의 6자책 실점(방금 던진 그 경기)은 절대 섞이면 안 된다.
        # home(TMB)은 첫 등판(PITB02)이라 아직 표본 없음 = 리그평균(4.30).
        # starter_edge = (away_era - home_era)/4 = (1.50 - 4.30)/4 = -0.70
        self.assertAlmostEqual(g2["starter_edge"], -0.70, places=2)

    def test_games_outside_date_range_are_excluded(self):
        games = [_game("G1", "20220101", "TMA", "TMB", "TMA", "TMB")]
        pitching = [_pitch("G1", "PIT001", "TMA", 1, ipouts=18, er=1)]
        rows = build_training_rows(games, pitching, "20240101", "20241231")
        self.assertEqual(rows, [])

    def test_game_missing_starter_is_skipped_not_guessed(self):
        games = [_game("G1", "20240401", "TMA", "TMB", "TMA", "TMB")]
        pitching = [_pitch("G1", "PITB01", "TMB", 1, ipouts=15, er=4)]  # 홈 선발 정보 없음
        rows = build_training_rows(games, pitching, "20240101", "20241231")
        self.assertEqual(rows, [])

    def test_rest_edge_reflects_days_since_last_game(self):
        games = [
            _game("G1", "20240401", "TMA", "TMB", "TMA", "TMB"),
            _game("G2", "20240402", "TMA", "TMC", "TMA", "TMC"),  # TMA 하루 휴식(원정팀 TMC는 첫 경기=기본 4일 취급)
        ]
        pitching = [
            _pitch("G1", "PIT001", "TMA", 1, ipouts=18, er=1),
            _pitch("G1", "PITB01", "TMB", 1, ipouts=15, er=4),
            _pitch("G2", "PIT002", "TMA", 1, ipouts=15, er=3),
            _pitch("G2", "PITC01", "TMC", 1, ipouts=15, er=3),
        ]
        rows = build_training_rows(games, pitching, "20240101", "20241231")
        g2 = rows[1]
        # TMA: 4/1 경기 후 4/2 등판 -> 1일 휴식 (기본 4일 대비 -3, clamp[-2,6] -> -2)
        # TMC: 이번 윈도우 첫 경기 -> 기본값 4일 취급 -> 0
        # rest_edge = (home_rest - away_rest)/4 = (-2 - 0)/4 = -0.5
        self.assertAlmostEqual(g2["rest_edge"], -0.5, places=2)


if __name__ == "__main__":
    unittest.main()
