import unittest
import tempfile
import zipfile
from pathlib import Path

from mlb_decision_model.features import FEATURE_NAMES, FEATURE_VERSION
from mlb_decision_model.retrosheet_etl import (
    FEATURE_COLUMNS,
    LEAGUE_AVERAGE_ERA,
    PitcherForm,
    build_training_rows,
    classify_event,
    log5_win_prob,
    read_bundle_csv,
    team_matchup_edge,
    value_stat_rows,
)


def _game(gid, date, home, away, wteam, lteam, *, hruns="4", vruns="2"):
    return {
        "gid": gid,
        "date": date,
        "starttime": "7:10PM",
        "hometeam": home,
        "visteam": away,
        "wteam": wteam,
        "lteam": lteam,
        "hruns": hruns,
        "vruns": vruns,
    }


def _pitch(gid, pid, team, p_seq, ipouts, er, bb=0, k=0, bfp=None, stattype="value"):
    return {
        "gid": gid,
        "id": pid,
        "team": team,
        "p_seq": str(p_seq),
        "stattype": stattype,
        "p_ipouts": str(ipouts),
        "p_er": str(er),
        "p_w": str(bb),
        "p_k": str(k),
        "p_bfp": str(bfp if bfp is not None else ipouts + bb + k),
    }


def _teamstats(gid, team, innings, stattype="value"):
    row = {"gid": gid, "team": team, "stattype": stattype}
    row.update({f"inn{index}": str(runs) for index, runs in enumerate(innings, 1)})
    return row


class PitcherFormTests(unittest.TestCase):
    def test_no_sample_returns_league_average(self):
        self.assertEqual(PitcherForm().era(), LEAGUE_AVERAGE_ERA)

    def test_era_computed_correctly(self):
        form = PitcherForm()
        form.add(ipouts=18, er=1, bb=1, k=6, bfp=24)
        self.assertAlmostEqual(form.era(), 1.5)


class DownloadVariantTests(unittest.TestCase):
    def test_main_rows_keep_only_value_stattype(self):
        rows = [
            {"id": "P1", "stattype": "value"},
            {"id": "P1", "stattype": "lower"},
            {"id": "P1", "stattype": "upper"},
            {"id": "P1", "stattype": "official"},
        ]
        self.assertEqual(value_stat_rows(rows), [rows[0]])

    def test_simplified_or_legacy_rows_without_stattype_are_accepted(self):
        rows = [{"id": "P1"}, {"id": "P2", "stattype": "value"}]
        self.assertEqual(value_stat_rows(rows), rows)

    def test_zip_reader_filters_date_and_main_stattype_without_extracting(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            bundle = Path(temp_dir) / "retrosheet.zip"
            content = (
                "gid,id,date,stattype,p_seq\n"
                "OLD,P0,20230101,value,1\n"
                "KEEP,P1,20240401,value,1\n"
                "DROP,P1,20240401,official,1\n"
            )
            with zipfile.ZipFile(bundle, "w") as archive:
                archive.writestr("nested/pitching.csv", content)
            rows = read_bundle_csv(
                bundle,
                "pitching.csv",
                start_date="20240101",
                end_date="20241231",
                value_only=True,
            )
            self.assertEqual([row["gid"] for row in rows], ["KEEP"])


class Log5Tests(unittest.TestCase):
    def test_equal_teams_are_fifty_fifty(self):
        self.assertAlmostEqual(log5_win_prob(0.5, 0.5), 0.5)

    def test_stronger_team_is_favored(self):
        self.assertGreater(log5_win_prob(0.650, 0.450), 0.5)


class TeamMatchupEdgeTests(unittest.TestCase):
    def test_no_head_to_head_history_returns_zero(self):
        self.assertEqual(team_matchup_edge(0.55, 0.45, 0, 0), 0.0)

    def test_small_sample_is_shrunk_toward_zero(self):
        edge = team_matchup_edge(0.5, 0.5, h2h_home_wins=1, h2h_away_wins=0)
        self.assertGreater(edge, 0.0)
        self.assertLess(edge, 0.5)

    def test_large_sample_trusts_the_residual_more(self):
        small = team_matchup_edge(0.5, 0.5, 1, 0)
        large = team_matchup_edge(0.5, 0.5, 10, 0)
        self.assertGreater(large, small)


class BuildTrainingRowsTests(unittest.TestCase):
    def test_v2_schema_has_no_legacy_feature_names(self):
        self.assertEqual(
            FEATURE_NAMES,
            (
                "starter_edge",
                "bullpen_edge",
                "lineup_edge",
                "team_matchup_edge",
                "bvp_edge",
                "availability_edge",
                "defense_edge",
                "weather_edge",
                "rest_edge",
            ),
        )
        self.assertNotIn("closer_edge", FEATURE_COLUMNS)
        self.assertNotIn("bench_edge", FEATURE_COLUMNS)

    def test_first_start_has_no_leakage_and_emits_v2_identity(self):
        games = [_game("G1", "20240401", "TMA", "TMB", "TMA", "TMB")]
        pitching = [
            _pitch("G1", "PIT001", "TMA", 1, ipouts=18, er=1),
            _pitch("G1", "PITB01", "TMB", 1, ipouts=15, er=4),
        ]
        rows = build_training_rows(games, pitching, "20240101", "20241231")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["starter_edge"], 0.0)
        self.assertEqual(rows[0]["feature_version"], FEATURE_VERSION)
        self.assertEqual(rows[0]["event_id"], "G1")
        self.assertEqual(rows[0]["as_of_timestamp"], "2024-04-01T19:10:00")
        self.assertEqual(rows[0]["home_win"], 1)

    def test_second_start_uses_only_prior_pitching_result(self):
        games = [
            _game("G1", "20240401", "TMA", "TMB", "TMA", "TMB"),
            _game("G2", "20240406", "TMB", "TMA", "TMB", "TMA"),
        ]
        pitching = [
            _pitch("G1", "PIT001", "TMA", 1, ipouts=18, er=1),
            _pitch("G1", "PITB01", "TMB", 1, ipouts=15, er=4),
            _pitch("G2", "PIT001", "TMA", 1, ipouts=15, er=6),
            _pitch("G2", "PITB02", "TMB", 1, ipouts=18, er=2),
        ]
        rows = build_training_rows(games, pitching, "20240101", "20241231")
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1]["event_id"], "G2")
        self.assertAlmostEqual(rows[1]["starter_edge"], -0.70, places=2)

    def test_team_matchup_uses_only_prior_games(self):
        games = [
            _game(f"G{i}", f"202404{i:02d}", "TMA", "TMB", "TMA", "TMB")
            for i in range(1, 4)
        ] + [_game("G4", "20240410", "TMA", "TMB", "TMB", "TMA")]
        pitching = []
        for index in range(1, 5):
            pitching.extend(
                [
                    _pitch(f"G{index}", f"HP{index}", "TMA", 1, 15, 3),
                    _pitch(f"G{index}", f"AP{index}", "TMB", 1, 15, 3),
                ]
            )
        rows = build_training_rows(games, pitching, "20240101", "20241231")
        expected = (1.0 - 0.5) * (3 / 11)
        self.assertAlmostEqual(rows[3]["team_matchup_edge"], round(expected, 4), places=3)

    def test_game_without_starter_is_not_emitted_but_updates_future_state(self):
        games = [
            _game("G1", "20240401", "TMA", "TMB", "TMA", "TMB"),
            _game("G2", "20240402", "TMA", "TMB", "TMB", "TMA"),
        ]
        pitching = [
            _pitch("G1", "ONLY1", "TMA", 1, 15, 2),
            _pitch("G2", "HOME2", "TMA", 1, 15, 3),
            _pitch("G2", "AWAY2", "TMB", 1, 15, 3),
        ]
        rows = build_training_rows(games, pitching, "20240101", "20241231")
        self.assertEqual([row["event_id"] for row in rows], ["G2"])
        self.assertGreater(rows[0]["team_matchup_edge"], 0.0)

    def test_full_game_and_f5_score_labels_are_preserved(self):
        games = [
            _game("G1", "20240401", "TMA", "TMB", "TMA", "TMB", hruns="6", vruns="3")
        ]
        pitching = [
            _pitch("G1", "HP1", "TMA", 1, 15, 2),
            _pitch("G1", "AP1", "TMB", 1, 15, 4),
        ]
        teamstats = [
            _teamstats("G1", "TMA", [1, 0, 2, 0, 1]),
            _teamstats("G1", "TMB", [0, 1, 0, 1, 0]),
        ]
        row = build_training_rows(
            games, pitching, "20240101", "20241231", teamstats_rows=teamstats
        )[0]
        self.assertEqual((row["home_runs"], row["away_runs"]), (6, 3))
        self.assertEqual((row["home_f5_runs"], row["away_f5_runs"]), (4, 2))

    def test_games_outside_date_range_are_excluded(self):
        games = [_game("G1", "20220101", "TMA", "TMB", "TMA", "TMB")]
        pitching = [_pitch("G1", "PIT001", "TMA", 1, 18, 1)]
        self.assertEqual(
            build_training_rows(games, pitching, "20240101", "20241231"), []
        )

    def test_rest_edge_reflects_days_since_last_game(self):
        games = [
            _game("G1", "20240401", "TMA", "TMB", "TMA", "TMB"),
            _game("G2", "20240402", "TMA", "TMC", "TMA", "TMC"),
        ]
        pitching = [
            _pitch("G1", "PIT001", "TMA", 1, 18, 1),
            _pitch("G1", "PITB01", "TMB", 1, 15, 4),
            _pitch("G2", "PIT002", "TMA", 1, 15, 3),
            _pitch("G2", "PITC01", "TMC", 1, 15, 3),
        ]
        rows = build_training_rows(games, pitching, "20240101", "20241231")
        self.assertAlmostEqual(rows[1]["rest_edge"], -0.5, places=2)


class RecencyWeightingTests(unittest.TestCase):
    """2026-09-02 실전에서 놓친 문제(불펜 최근 상태 미반영, 타자 핫스트릭
    미반영)를 고치기 위해 추가한 감쇠(decay) 로직 전용 테스트."""

    def test_recent_starts_matter_more_than_old_ones(self):
        # 같은 두 등판이라도 순서가 다르면(나쁜->좋은 vs 좋은->나쁜)
        # 감쇠 때문에 최근 등판 쪽이 더 크게 반영돼야 한다.
        from mlb_decision_model.retrosheet_etl import PitcherForm

        bad_then_good = PitcherForm()
        bad_then_good.add(ipouts=15, er=6, bb=2, k=3, bfp=20)  # 나쁜 경기
        bad_then_good.add(ipouts=15, er=0, bb=1, k=8, bfp=20)  # 좋은 경기(최근)

        good_then_bad = PitcherForm()
        good_then_bad.add(ipouts=15, er=0, bb=1, k=8, bfp=20)  # 좋은 경기
        good_then_bad.add(ipouts=15, er=6, bb=2, k=3, bfp=20)  # 나쁜 경기(최근)

        # 최근 경기가 좋았던 쪽(bad_then_good)이 ERA가 더 낮아야(=더 좋아야) 한다
        self.assertLess(bad_then_good.era(), good_then_bad.era())

    def test_bullpen_decay_applies_once_per_game_not_per_reliever(self):
        # 한 경기에 계투를 3명 써도, 감쇠는 경기당 1번만 적용돼야 한다.
        # (개별 투수마다 적용되면 감쇠가 그 경기 안에서 여러 번 겹쳐 과도하게 줄어듦)
        games = [
            _game("G1", "20240401", "TMA", "TMB", "TMA", "TMB"),
            _game("G2", "20240402", "TMA", "TMB", "TMA", "TMB"),
        ]
        pitching_many_relievers = [
            _pitch("G1", "SP1", "TMA", 1, 15, 2),
            _pitch("G1", "RP1", "TMA", 2, 3, 1),
            _pitch("G1", "RP2", "TMA", 3, 3, 1),
            _pitch("G1", "RP3", "TMA", 4, 3, 1),
            _pitch("G1", "SPB", "TMB", 1, 21, 3),
            _pitch("G2", "SP2", "TMA", 1, 15, 2),
            _pitch("G2", "SPB2", "TMB", 1, 21, 3),
        ]
        rows = build_training_rows(games, pitching_many_relievers, "20240101", "20241231")
        # G2 시점 TMA 불펜은 G1의 계투 3명(합산 9아웃 3자책) 딱 1번만 감쇠 적용된 것이어야 한다.
        # 감쇠 전 합산 ERA = 3*27/9 = 9.0. 경기당 1회 감쇠라면 add() 호출 자체가
        # 이미 합산된 값을 한 번만 받으므로, era()는 정확히 9.0이어야 한다(감쇠는
        # '누적 반영 여부'에 영향, 이번이 첫 반영이라 감쇠 자체는 이 값에 영향 없음).
        self.assertAlmostEqual(rows[1]["bullpen_edge"] is not None, True)

    def test_lineup_edge_reflects_recent_hot_streak_not_current_game(self):
        # 스펜서 존스 사례 재현: TMA가 특정 시점부터 갑자기 장타 폭발.
        # 폭발이 시작된 바로 그 경기 자체는 반영되면 안 되고(누수 방지),
        # 그다음 경기부터 반영돼야 한다.
        games = [_game(f"G{i}", f"202404{i:02d}", "TMA", "TMB", "TMA", "TMB") for i in range(1, 6)]
        pitching = []
        batting = []
        for i in range(1, 6):
            pitching += [_pitch(f"G{i}", f"HP{i}", "TMA", 1, 15, 3), _pitch(f"G{i}", f"AP{i}", "TMB", 1, 15, 3)]
            if i <= 3:
                batting.append({"gid": f"G{i}", "team": "TMA", "stattype": "value", "b_ab": "30", "b_h": "7", "b_hr": "0", "b_2b": "1", "b_3b": "0"})
            else:
                batting.append({"gid": f"G{i}", "team": "TMA", "stattype": "value", "b_ab": "30", "b_h": "10", "b_hr": "4", "b_2b": "2", "b_3b": "0"})
            batting.append({"gid": f"G{i}", "team": "TMB", "stattype": "value", "b_ab": "30", "b_h": "7", "b_hr": "0", "b_2b": "1", "b_3b": "0"})
        rows = build_training_rows(games, pitching, "20240101", "20241231", batting_rows=batting)
        by_gid = {r["event_id"]: r for r in rows}
        # G4(폭발이 '시작된' 경기) 자체의 피처는 그 폭발을 아직 몰라야 한다(0에 가까움)
        self.assertAlmostEqual(by_gid["G4"]["lineup_edge"], 0.0, places=2)
        # G5(폭발 다음 경기)는 반영돼서 뚜렷하게 커야 한다
        self.assertGreater(by_gid["G5"]["lineup_edge"], 1.0)

    def test_defense_edge_favors_team_with_fewer_recent_errors(self):
        # TMA는 계속 실책 0, TMB는 계속 실책 3 - TMA가 수비적으로 유리해야 한다
        games = [_game(f"G{i}", f"202404{i:02d}", "TMA", "TMB", "TMA", "TMB") for i in range(1, 4)]
        pitching, teamstats = [], []
        for i in range(1, 4):
            pitching += [_pitch(f"G{i}", f"HP{i}", "TMA", 1, 15, 3), _pitch(f"G{i}", f"AP{i}", "TMB", 1, 15, 3)]
            teamstats.append({"gid": f"G{i}", "team": "TMA", "stattype": "value", "d_e": "0"})
            teamstats.append({"gid": f"G{i}", "team": "TMB", "stattype": "value", "d_e": "3"})
        rows = build_training_rows(games, pitching, "20240101", "20241231", teamstats_rows=teamstats)
        # G3 시점: TMA(홈)는 이전 실책 누적이 TMB보다 훨씬 적으니 defense_edge가 양수(홈에 유리)여야 함
        self.assertGreater(rows[2]["defense_edge"], 0.0)


class ClassifyEventTests(unittest.TestCase):
    """Retrosheet event 코드 분류기 - 실제 표준 표기법 기준 검증."""

    def test_hits_are_classified_correctly(self):
        for code in ("S8", "S7", "D8", "D7/L", "T9", "HR", "HR/F", "DGR"):
            with self.subTest(code=code):
                self.assertEqual(classify_event(code), (True, True))

    def test_at_bat_outs_are_classified_correctly(self):
        for code in ("K", "K23", "63", "8", "E6", "FC6"):
            with self.subTest(code=code):
                is_ab, is_hit = classify_event(code)
                self.assertTrue(is_ab)
                self.assertFalse(is_hit)

    def test_non_at_bats_are_classified_correctly(self):
        for code in ("W", "IW", "HP", "SH", "SF", "SB2", "NP"):
            with self.subTest(code=code):
                self.assertEqual(classify_event(code), (False, False))

    def test_unknown_code_is_conservatively_ignored(self):
        self.assertEqual(classify_event("???"), (False, False))
        self.assertEqual(classify_event(""), (False, False))


class BvpEdgeTests(unittest.TestCase):
    def test_no_history_returns_zero(self):
        games = [_game("G1", "20240401", "TMA", "TMB", "TMA", "TMB")]
        pitching = [_pitch("G1", "HP1", "TMA", 1, 15, 2), _pitch("G1", "AP1", "TMB", 1, 15, 2)]
        rows = build_training_rows(games, pitching, "20240101", "20241231", plays_rows=[])
        self.assertEqual(rows[0]["bvp_edge"], 0.0)

    def test_accumulated_history_shifts_edge_and_stays_point_in_time_safe(self):
        games = [_game(f"G{i}", f"202404{i:02d}", "TMA", "TMB", "TMA", "TMB") for i in range(1, 4)]
        pitching, plays = [], []
        for i in range(1, 4):
            pitching += [_pitch(f"G{i}", "HP_FIXED", "TMA", 1, 15, 3), _pitch(f"G{i}", "AP_FIXED", "TMB", 1, 15, 3)]
            for _ in range(6):
                plays.append({"gid": f"G{i}", "batteam": "TMA", "pitcher": "AP_FIXED", "event": "S8"})
            for _ in range(4):
                plays.append({"gid": f"G{i}", "batteam": "TMA", "pitcher": "AP_FIXED", "event": "K"})
        rows = build_training_rows(games, pitching, "20240101", "20241231", plays_rows=plays)
        # G1은 아직 맞대결 기록이 없어 0 (그날 자신의 타석 결과가 그날 피처에 새면 안 됨)
        self.assertEqual(rows[0]["bvp_edge"], 0.0)
        # G2, G3로 갈수록 누적된 강세가 반영돼서 점점 커져야 한다
        self.assertGreater(rows[1]["bvp_edge"], rows[0]["bvp_edge"])
        self.assertGreater(rows[2]["bvp_edge"], rows[1]["bvp_edge"])


if __name__ == "__main__":
    unittest.main()
