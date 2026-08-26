from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


FEATURE_NAMES = (
    "starter_edge",
    "bullpen_edge",
    "closer_edge",
    "lineup_edge",
    "bvp_edge",
    "bench_edge",
    "availability_edge",
    "defense_edge",
    "rest_edge",
)


class DataQualityError(ValueError):
    """Raised when a prediction would confuse missing data with team strength."""


def _num(data: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = data.get(key, default)
    return default if value is None else float(value)


def _handed_woba(batter: dict[str, Any], pitcher_hand: str) -> float:
    key = "woba_vs_l" if pitcher_hand.upper() == "L" else "woba_vs_r"
    return _num(batter, key, _num(batter, "woba", 0.310))


def shrunk_bvp_average(
    batter: dict[str, Any], pitcher_hand: str, prior_pa: float = 40.0
) -> tuple[float, float]:
    """Return empirical-Bayes BvP average and its 0..1 reliability.

    A small batter-vs-pitcher sample is pulled toward the batter's handedness
    split. This prevents 2-for-3 histories from dominating the model.
    """
    bvp = batter.get("bvp", {}) or {}
    at_bats = _num(bvp, "ab")
    hits = _num(bvp, "h")
    walks = _num(bvp, "bb")
    pa = at_bats + walks
    prior_avg = max(0.120, min(0.420, _handed_woba(batter, pitcher_hand) - 0.070))
    posterior = (hits + prior_pa * prior_avg) / max(1.0, at_bats + prior_pa)
    reliability = pa / (pa + prior_pa)
    return posterior, reliability


def _pitcher_score(pitcher: dict[str, Any]) -> float:
    era = _num(pitcher, "era", 4.30)
    fip = _num(pitcher, "fip", era)
    xera = _num(pitcher, "xera", fip)
    k_rate = _num(pitcher, "k_rate", 0.220)
    bb_rate = _num(pitcher, "bb_rate", 0.085)
    gb_rate = _num(pitcher, "gb_rate", 0.420)
    rest_days = _num(pitcher, "rest_days", 4.0)
    velocity_delta = _num(pitcher, "velocity_delta", 0.0)
    quality = -0.40 * era - 0.35 * fip - 0.25 * xera
    skills = 8.0 * (k_rate - bb_rate) + 0.8 * (gb_rate - 0.42)
    readiness = 0.06 * min(2.0, max(-2.0, rest_days - 4.0)) + 0.12 * velocity_delta
    return quality + skills + readiness


def _availability(pitcher: dict[str, Any]) -> float:
    explicit = pitcher.get("availability")
    if explicit is not None:
        return max(0.0, min(1.0, float(explicit)))
    pitches_1d = _num(pitcher, "pitches_1d")
    pitches_3d = _num(pitcher, "pitches_3d")
    back_to_back = bool(pitcher.get("pitched_back_to_back", False))
    penalty = 0.018 * pitches_1d + 0.006 * max(0.0, pitches_3d - pitches_1d)
    if back_to_back:
        penalty += 0.20
    return max(0.0, min(1.0, 1.0 - penalty))


def _bullpen_score(relievers: Iterable[dict[str, Any]]) -> tuple[float, float]:
    arms = list(relievers)
    if not arms:
        raise DataQualityError(
            "bullpen data is required; missing data must not be scored as a weak bullpen"
        )
    weighted_quality = 0.0
    usable_weight = 0.0
    total_availability = 0.0
    for arm in arms:
        avail = _availability(arm)
        leverage = max(0.25, _num(arm, "leverage_weight", 1.0))
        weight = avail * leverage
        weighted_quality += _pitcher_score(arm) * weight
        usable_weight += weight
        total_availability += avail
    score = weighted_quality / max(0.25, usable_weight)
    availability = total_availability / len(arms)
    return score, availability


def _lineup_scores(
    batters: Iterable[dict[str, Any]], pitcher_hand: str
) -> tuple[float, float]:
    lineup = list(batters)
    if not lineup:
        return 0.310, 0.240
    split_total = 0.0
    bvp_total = 0.0
    weight_total = 0.0
    for index, batter in enumerate(lineup):
        order_weight = max(0.65, 1.08 - 0.045 * index)
        split = _handed_woba(batter, pitcher_hand)
        bvp_avg, reliability = shrunk_bvp_average(batter, pitcher_hand)
        bvp_equivalent_woba = bvp_avg + 0.070
        bvp_blend = split * (1.0 - reliability) + bvp_equivalent_woba * reliability
        split_total += split * order_weight
        bvp_total += bvp_blend * order_weight
        weight_total += order_weight
    return split_total / weight_total, bvp_total / weight_total


def _bench_score(bench: Iterable[dict[str, Any]], opposing_hand: str) -> float:
    candidates = sorted(
        (_handed_woba(player, opposing_hand) for player in bench), reverse=True
    )
    if not candidates:
        return 0.300
    return sum(candidates[:3]) / min(3, len(candidates))


@dataclass(frozen=True)
class TeamFeatures:
    starter: float
    bullpen: float
    closer: float
    lineup: float
    bvp: float
    bench: float
    availability: float
    defense: float
    rest: float


def _team_features(team: dict[str, Any], opponent: dict[str, Any]) -> TeamFeatures:
    starter = team.get("starter", {}) or {}
    opposing_starter = opponent.get("starter", {}) or {}
    opposing_hand = str(opposing_starter.get("hand", "R"))
    bullpen, availability = _bullpen_score(team.get("bullpen", []) or [])
    closer = team.get("closer", {}) or {}
    closer_score = _pitcher_score(closer) * _availability(closer) if closer else bullpen
    lineup, bvp = _lineup_scores(team.get("lineup", []) or [], opposing_hand)
    bench = _bench_score(team.get("bench", []) or [], opposing_hand)
    return TeamFeatures(
        starter=_pitcher_score(starter),
        bullpen=bullpen,
        closer=closer_score,
        lineup=lineup,
        bvp=bvp,
        bench=bench,
        availability=availability,
        defense=_num(team, "defense_runs_per_game"),
        rest=_num(team, "rest_days", 1.0),
    )


def build_game_features(game: dict[str, Any]) -> dict[str, float]:
    """Build signed home-minus-away model features from a nested game record."""
    home = game["home"]
    away = game["away"]
    h = _team_features(home, away)
    a = _team_features(away, home)
    return {
        "starter_edge": h.starter - a.starter,
        "bullpen_edge": h.bullpen - a.bullpen,
        "closer_edge": h.closer - a.closer,
        "lineup_edge": h.lineup - a.lineup,
        "bvp_edge": h.bvp - a.bvp,
        "bench_edge": h.bench - a.bench,
        "availability_edge": h.availability - a.availability,
        "defense_edge": h.defense - a.defense,
        "rest_edge": h.rest - a.rest,
    }
