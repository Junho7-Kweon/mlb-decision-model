import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mlb_decision_model.mlb_daily_data import Client, splits, team_form, pitcher_form, refresh, completed_game
from datetime import datetime, timezone
from mlb_decision_model.daily_data import DailyDataUnavailable


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
