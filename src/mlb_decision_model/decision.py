from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class MarketDecision:
    side: str
    probability: float
    odds: float
    break_even: float
    expected_return: float
    edge: float
    action: str


def evaluate_market(
    home_probability: float,
    home_odds: float,
    away_odds: float,
    minimum_edge: float = 0.025,
) -> tuple[MarketDecision, MarketDecision]:
    """Evaluate both sides; a weak user pick automatically exposes its opposite."""
    home_probability = max(0.001, min(0.999, home_probability))
    values = []
    for side, probability, odds in (
        ("home", home_probability, home_odds),
        ("away", 1.0 - home_probability, away_odds),
    ):
        break_even = 1.0 / odds
        edge = probability - break_even
        ev = probability * odds - 1.0
        if edge >= minimum_edge:
            action = "BET"
        elif edge <= -minimum_edge:
            action = "FADE"
        else:
            action = "PASS"
        values.append(MarketDecision(side, probability, odds, break_even, ev, edge, action))
    return values[0], values[1]


@dataclass(frozen=True)
class Pick:
    name: str
    probability: float
    odds: float
    event_id: str | None = None


@dataclass(frozen=True)
class CombinationResult:
    picks: tuple[str, ...]
    hit_probability: float
    decimal_odds: float
    expected_return: float
    profitable_trial_rate: float


def rank_combinations(
    picks: Sequence[Pick],
    legs: int = 2,
    simulations: int = 50000,
    seed: int = 7,
    sort_by: str = "expected_return",
) -> list[CombinationResult]:
    if legs < 1 or legs > len(picks):
        raise ValueError("legs must be between 1 and the number of picks")
    if sort_by not in {"expected_return", "survival"}:
        raise ValueError("sort_by must be 'expected_return' or 'survival'")
    rng = random.Random(seed)
    ranked = []
    for combo in itertools.combinations(picks, legs):
        event_ids = [pick.event_id for pick in combo if pick.event_id]
        if len(event_ids) != len(set(event_ids)):
            continue
        hit_probability = 1.0
        decimal_odds = 1.0
        for pick in combo:
            hit_probability *= pick.probability
            decimal_odds *= pick.odds
        wins = 0
        for _ in range(max(0, simulations)):
            if all(rng.random() < pick.probability for pick in combo):
                wins += 1
        ranked.append(
            CombinationResult(
                picks=tuple(pick.name for pick in combo),
                hit_probability=hit_probability,
                decimal_odds=decimal_odds,
                expected_return=hit_probability * decimal_odds - 1.0,
                profitable_trial_rate=wins / simulations if simulations else hit_probability,
            )
        )
    if sort_by == "survival":
        key = lambda row: (row.hit_probability, row.expected_return)
    else:
        key = lambda row: (row.expected_return, row.hit_probability)
    return sorted(ranked, key=key, reverse=True)


def portfolio_profit_rate(
    tickets: Iterable[tuple[Sequence[Pick], float]],
    simulations: int = 50000,
    seed: int = 11,
) -> tuple[float, float]:
    """Return probability of net profit and mean ROI for a multi-ticket portfolio."""
    tickets = list(tickets)
    rng = random.Random(seed)
    total_stake = sum(stake for _, stake in tickets)
    profitable = 0
    total_profit = 0.0
    for _ in range(simulations):
        outcomes: dict[str, bool] = {}
        profit = -total_stake
        for picks, stake in tickets:
            for pick in picks:
                outcomes.setdefault(pick.name, rng.random() < pick.probability)
            if all(outcomes[pick.name] for pick in picks):
                odds = 1.0
                for pick in picks:
                    odds *= pick.odds
                profit += stake * odds
        profitable += int(profit > 0)
        total_profit += profit
    return profitable / simulations, total_profit / simulations / max(total_stake, 1e-9)
