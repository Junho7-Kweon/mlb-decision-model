import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mlb_decision_model.mlb_daily_data import Client, splits, team_form, pitcher_form, refresh, completed_game, pregame_game, team_aliases
from datetime import datetime, timezone
from mlb_decision_model.daily_data import DailyDataUnavailable
from mlb_decision_model.postgame_review import analyze_game, collect_date


def fixtures():
    game = {1: {"gameDate": "2026-09-08T18:00:00Z", "teams": {"home": {"team": {"id": 10}}, "away": {"team": {"id": 20}}}}}
    def row(stat):
        return {"game": {"gamePk": 1}, "team": {"id": 10}, "opponent": {"id": 20}, "isWin": True, "stat": stat}
    def block(group, rows):
        return {"group": {"displayName": group}, "splits": rows}
    logs = [{"stats": [block("hitting", [row(dict(atBats=30, hits=10, doubles=2, triples=0, homeRuns=1))]),
                        block("pitching", [row(dict(outs=27, earnedRuns=3, battersFaced=33))]),
                        block("fielding", [row(dict(errors=1))])]}]
    arms = [{"id": p, "stats": [block("pitching", [row(s)])]} for p, s in [(100, dict(outs=18, earnedRuns=1, gamesStarted=1, battersFaced=21)), (101, dict(outs=9, earnedRuns=2, gamesStarted=0, battersFaced=12))]]
    return game, logs, arms


class MLBDataTests(unittest.TestCase):
    def test_mlb_preview_states_are_accepted_until_live(self):
        for detailed in ("Scheduled", "Pre-Game", "Warmup"):
            self.assertTrue(pregame_game({"status": {"abstractGameState": "Preview", "detailedState": detailed}}))
        self.assertTrue(pregame_game({"status": {"abstractGameState": "Live", "detailedState": "Warmup"}}))
        self.assertFalse(pregame_game({"status": {"abstractGameState": "Live", "detailedState": "In Progress"}}))
        self.assertFalse(pregame_game({"status": {"abstractGameState": "Final", "detailedState": "Postponed"}}))

    def test_reviewed_korean_ocr_team_aliases(self):
        from mlb_decision_model.daily_data import KOREAN
        self.assertEqual(len(KOREAN), 30)
        angels = team_aliases({"name": "Los Angeles Angels", "abbreviation": "LAA"})
        royals = team_aliases({"name": "Kansas City Royals", "abbreviation": "KC"})
        self.assertIn("에인절스", angels)
        self.assertIn("LA 에인절스", angels)
        self.assertIn("캔자스시티 로얄스", royals)

    def test_postponed_is_not_completed_despite_final_abstract_state(self):
        g = {"status": {"codedGameState": "D", "abstractGameState": "Final"}, "gameDate": "2026-05-23T17:35:00Z"}
        self.assertFalse(completed_game(g, datetime(2026, 9, 10, tzinfo=timezone.utc)))

    def test_team_er_need_not_equal_individual_er(self):
        g, logs, arms = fixtures()
        logs[0]["stats"][1]["splits"][0]["stat"]["earnedRuns"] = 1
        self.assertEqual(team_form(10, logs, arms, g)[0]["bullpen_era"], 6.)

    def test_bullpen_not_whole_team_era(self):
        g, logs, arms = fixtures()
        form, h2h, last = team_form(10, logs, arms, g)
        self.assertEqual(form["bullpen_era"], 6.)
        self.assertEqual(form["team_slg"], .5)
        self.assertEqual(h2h[20], 1)
        self.assertEqual(str(last), "2026-09-08")

    def test_missing_relief_blocks_prediction(self):
        g, logs, arms = fixtures()
        with self.assertRaises(DailyDataUnavailable):
            team_form(10, logs, arms[:1], g)

    def test_missing_relief_can_be_repaired_from_boxscore(self):
        g, logs, arms = fixtures()
        side = {"team": {"id": 10}, "pitchers": [100, 101], "players": {
            f"ID{p['id']}": {"stats": {"pitching": p["stats"][0]["splits"][0]["stat"]}} for p in arms}}
        form = team_form(10, logs, arms[:1], g, repair=lambda gid: {"teams": {"home": side}})[0]
        self.assertEqual(form["bullpen_era"], 6.)

    def test_missing_fielding_blocks_prediction(self):
        g, logs, arms = fixtures()
        logs[0]["stats"].pop()
        with self.assertRaises(DailyDataUnavailable):
            team_form(10, logs, arms, g)

    def test_pitcher_only_starts(self):
        g, _, arms = fixtures()
        self.assertEqual(pitcher_form(100, [{"people": arms}], g), 1.5)
        with self.assertRaises(DailyDataUnavailable):
            pitcher_form(101, [{"people": arms}], g)

    def test_truncated_response_rejected(self):
        with self.assertRaises(DailyDataUnavailable):
            splits({"stats": [{"group": {"displayName": "pitching"}, "totalSplits": 2, "splits": [{}]}]}, "pitching")

    def test_odds_endpoint_rejected_without_network(self):
        with tempfile.TemporaryDirectory() as d, patch("mlb_decision_model.mlb_daily_data.urlopen") as network:
            with self.assertRaises(ValueError):
                Client(Path(d)).get("odds")
            network.assert_not_called()

    def test_past_date_not_backfilled_with_current_data(self):
        with tempfile.TemporaryDirectory() as d, patch("mlb_decision_model.mlb_daily_data.urlopen") as network:
            with self.assertRaises(DailyDataUnavailable):
                refresh(Path(d) / "out.json", "2020-01-01")
            network.assert_not_called()

    def test_postgame_review_derives_turning_points(self):
        game = {
            "gamePk": 123,
            "gameType": "R",
            "gameDate": "2026-09-17T01:00:00Z",
            "status": {"abstractGameState": "Final", "codedGameState": "F"},
            "teams": {
                "away": {"team": {"id": 10, "name": "Away"}, "score": 5},
                "home": {"team": {"id": 20, "name": "Home"}, "score": 3},
            },
        }
        plays = {"allPlays": [
            self._play(1, "top", 1, 0, 1, "Solo home run"),
            self._play(1, "bottom", 1, 2, 2, "Two-run double"),
            self._play(7, "top", 4, 2, 3, "Three-run home run"),
            self._play(8, "bottom", 4, 3, 1, "RBI single"),
            self._play(9, "top", 5, 3, 1, "RBI double"),
        ]}

        review = analyze_game(game, plays)

        self.assertEqual(review["winner"], "away")
        self.assertEqual(review["derived"]["total_runs"], 8)
        self.assertEqual(review["derived"]["run_margin"], 2)
        self.assertEqual(review["derived"]["first_five_runs"], 3)
        self.assertEqual(review["derived"]["late_runs_innings_7_plus"], 5)
        self.assertEqual(review["derived"]["lead_changes"], 2)
        self.assertEqual(review["derived"]["winner_max_deficit"], 1)
        self.assertTrue(review["derived"]["comeback_win"])
        self.assertEqual(review["derived"]["decisive_inning"], 7)
        self.assertIn("PERMANENT_LEAD", review["key_turning_points"][0]["tags"])

    def test_postgame_review_rejects_score_mismatch(self):
        game = {
            "gamePk": 123,
            "gameType": "R",
            "gameDate": "2026-09-17T01:00:00Z",
            "status": {"abstractGameState": "Final", "codedGameState": "F"},
            "teams": {
                "away": {"team": {"id": 10, "name": "Away"}, "score": 2},
                "home": {"team": {"id": 20, "name": "Home"}, "score": 1},
            },
        }
        with self.assertRaises(ValueError):
            analyze_game(game, {"allPlays": [self._play(9, "top", 1, 1, 1, "Tie game")]})

    def test_collect_date_writes_all_korean_date_finals(self):
        game = {
            "gamePk": 123,
            "gameType": "R",
            "gameDate": "2026-09-16T23:30:00Z",
            "status": {"abstractGameState": "Final", "codedGameState": "F"},
            "teams": {
                "away": {"team": {"id": 10, "name": "Away"}, "score": 1},
                "home": {"team": {"id": 20, "name": "Home"}, "score": 0},
            },
        }

        class FakeClient:
            def get(self, endpoint, params=None, ttl=0):
                if endpoint == "schedule":
                    return {"dates": [{"games": [game]}]}
                self.assert_endpoint = endpoint
                return {"allPlays": [MLBDataTests._play(4, "top", 1, 0, 1, "Solo home run")]}

        with tempfile.TemporaryDirectory() as d:
            result = collect_date(
                Path(d),
                "2026-09-17",
                client=FakeClient(),
                now=datetime(2026, 9, 17, 12, tzinfo=timezone.utc),
            )
            self.assertEqual(result["game_count"], 1)
            self.assertTrue(Path(result["path"]).is_file())
            self.assertEqual(result["payload"]["games"][0]["event_id"], "mlb:123")

    @staticmethod
    def _play(inning, half, away_score, home_score, runs, description):
        return {
            "about": {"inning": inning, "halfInning": half},
            "result": {
                "awayScore": away_score,
                "homeScore": home_score,
                "rbi": runs,
                "event": "Run",
                "description": description,
            },
        }
