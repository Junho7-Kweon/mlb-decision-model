"""Collect every finalized MLB game and derive factual turning-point labels.

Raw MLB facts remain source data.  The derived labels in this module are the
project-specific layer used for error analysis and future feature research.
They are not injected into the deployed model until a historical backfill,
chronological validation and a new training manifest exist.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import date as Date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .mlb_daily_data import BASE, KST, Client, save
from .pregame import timestamp


SCHEMA_VERSION = "mlb-postgame-review-v1"


class PostgameReviewUnavailable(RuntimeError):
    """Raised when a requested date has no fully finalized regular games."""


def _sign(value: int) -> int:
    return (value > 0) - (value < 0)


def _is_final_regular_game(game: dict[str, Any]) -> bool:
    status = game.get("status", {})
    return (
        game.get("gameType") == "R"
        and status.get("abstractGameState") == "Final"
        and status.get("codedGameState") in {"F", "O"}
        and all("score" in game.get("teams", {}).get(side, {}) for side in ("away", "home"))
    )


def analyze_game(game: dict[str, Any], play_by_play: dict[str, Any]) -> dict[str, Any]:
    """Turn one official final and its plays into deterministic review labels."""
    away = game["teams"]["away"]
    home = game["teams"]["home"]
    final_away, final_home = int(away["score"]), int(home["score"])
    if final_away == final_home:
        raise ValueError("final MLB game cannot be tied")
    winner = "home" if final_home > final_away else "away"

    previous_away = previous_home = 0
    scoring: list[dict[str, Any]] = []
    max_inning = 0
    half_inning_runs: defaultdict[tuple[int, str], int] = defaultdict(int)
    winner_max_deficit = 0
    lead_changes = ties_after_start = 0

    for play_index, play in enumerate(play_by_play.get("allPlays", [])):
        about, result = play.get("about", {}), play.get("result", {})
        inning = int(about.get("inning") or 0)
        half = str(about.get("halfInning") or "").casefold()
        max_inning = max(max_inning, inning)
        after_away = int(result.get("awayScore", previous_away))
        after_home = int(result.get("homeScore", previous_home))
        before_diff = previous_home - previous_away
        after_diff = after_home - after_away
        winner_deficit = (
            previous_away - previous_home
            if winner == "home"
            else previous_home - previous_away
        )
        winner_max_deficit = max(winner_max_deficit, winner_deficit)
        if _sign(before_diff) * _sign(after_diff) < 0:
            lead_changes += 1
        if after_diff == 0 and after_home + after_away > 0 and before_diff != 0:
            ties_after_start += 1
        runs_scored = max(0, after_away + after_home - previous_away - previous_home)
        if runs_scored:
            half_inning_runs[(inning, half)] += runs_scored
            tags: list[str] = []
            if runs_scored >= 2:
                tags.append("MULTI_RUN_PLAY")
            if inning >= 7:
                tags.append("LATE_GAME")
            if abs(before_diff) <= 1:
                tags.append("CLOSE_SCORE_BEFORE")
            if _sign(before_diff) * _sign(after_diff) < 0:
                tags.append("LEAD_CHANGE")
            elif before_diff != 0 and after_diff == 0:
                tags.append("GAME_TIED")
            scoring.append(
                {
                    "play_index": play_index,
                    "inning": inning,
                    "half": half,
                    "event": result.get("event"),
                    "description": result.get("description"),
                    "runs_scored": runs_scored,
                    "score_before": {"away": previous_away, "home": previous_home},
                    "score_after": {"away": after_away, "home": after_home},
                    "tags": tags,
                }
            )
        previous_away, previous_home = after_away, after_home

    if (previous_away, previous_home) != (final_away, final_home):
        raise ValueError("play-by-play score does not match official final")

    def winner_leads(point: dict[str, Any]) -> bool:
        score = point["score_after"]
        return score[winner] > score["away" if winner == "home" else "home"]

    decisive = None
    for index, point in enumerate(scoring):
        if winner_leads(point) and all(winner_leads(later) for later in scoring[index:]):
            decisive = point
            break
    if decisive is not None:
        decisive["tags"] = [*decisive["tags"], "PERMANENT_LEAD"]

    selected: list[dict[str, Any]] = []
    for point in [
        decisive,
        *sorted(scoring, key=lambda row: (row["runs_scored"], row["inning"]), reverse=True),
    ]:
        if point is None or any(row["play_index"] == point["play_index"] for row in selected):
            continue
        selected.append(point)
        if len(selected) == 3:
            break

    first_five_runs = sum(row["runs_scored"] for row in scoring if row["inning"] <= 5)
    late_runs = sum(row["runs_scored"] for row in scoring if row["inning"] >= 7)
    max_half = max(half_inning_runs.values(), default=0)
    return {
        "event_id": f"mlb:{game['gamePk']}",
        "game_pk": game["gamePk"],
        "starts_at": game["gameDate"],
        "away": {"team_id": away["team"]["id"], "name": away["team"]["name"], "runs": final_away},
        "home": {"team_id": home["team"]["id"], "name": home["team"]["name"], "runs": final_home},
        "winner": winner,
        "derived": {
            "total_runs": final_away + final_home,
            "run_margin": abs(final_home - final_away),
            "first_five_runs": first_five_runs,
            "late_runs_innings_7_plus": late_runs,
            "lead_changes": lead_changes,
            "ties_after_start": ties_after_start,
            "winner_max_deficit": winner_max_deficit,
            "largest_scoring_play_runs": max((row["runs_scored"] for row in scoring), default=0),
            "largest_half_inning_runs": max_half,
            "decisive_inning": decisive["inning"] if decisive else None,
            "decisive_half": decisive["half"] if decisive else None,
            "extra_innings": max_inning > 9,
            "one_run_game": abs(final_home - final_away) == 1,
            "blowout_margin_5_plus": abs(final_home - final_away) >= 5,
            "comeback_win": winner_max_deficit > 0,
        },
        "key_turning_points": selected,
    }


def collect_date(
    output_root: Path,
    game_date_kst: str | Date,
    *,
    client: Client | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    day = Date.fromisoformat(game_date_kst) if isinstance(game_date_kst, str) else game_date_kst
    now = now or datetime.now(timezone.utc)
    if day > now.astimezone(KST).date():
        raise PostgameReviewUnavailable("미래 경기 결과는 수집할 수 없습니다")
    output_root = Path(output_root)
    client = client or Client(output_root.parent / "mlb_cache")
    schedule = client.get(
        "schedule",
        {
            "sportId": 1,
            "startDate": (day - timedelta(days=1)).isoformat(),
            "endDate": day.isoformat(),
        },
        ttl=0,
    )
    finals = [
        game
        for date_block in schedule.get("dates", [])
        for game in date_block.get("games", [])
        if _is_final_regular_game(game)
        and timestamp(game["gameDate"]).astimezone(KST).date() == day
    ]
    if not finals:
        raise PostgameReviewUnavailable("선택한 한국 날짜에 확정된 정규시즌 경기가 없습니다")
    reviews = [
        analyze_game(game, client.get(f"game/{game['gamePk']}/playByPlay", ttl=30 * 86400))
        for game in sorted(finals, key=lambda row: row["gameDate"])
    ]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "game_date_kst": day.isoformat(),
        "collected_at": now.isoformat(),
        "source": {"source_id": "mlb_statsapi", "source_url": BASE, "scope": "final_regular_games_and_play_by_play"},
        "game_count": len(reviews),
        "games": reviews,
        "training_policy": {
            "current_model_use": "official finals feed next pregame team-form aggregation",
            "turning_point_use": "candidate feature and error-attribution layer",
            "direct_immediate_weight_update": False,
            "requirements_before_training": ["historical backfill", "deduplication", "chronological holdout", "training manifest"],
        },
    }
    destination = output_root / f"{day.isoformat()}.json"
    save(destination, payload)
    return {"path": str(destination), "game_count": len(reviews), "payload": payload}


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect MLB final games and turning-point labels")
    parser.add_argument("--date", required=True, help="Korean calendar date (YYYY-MM-DD)")
    parser.add_argument("--out-dir", type=Path, default=Path("data/private/game_reviews"))
    args = parser.parse_args()
    result = collect_date(args.out_dir, args.date)
    print(json.dumps({"path": result["path"], "game_count": result["game_count"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
