from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from .decision import evaluate_market
from .model import CalibratedLogisticModel, brier_score
from .train import read_training_csv


def _log_loss(probabilities: list[float], labels: list[int]) -> float:
    total = 0.0
    for probability, label in zip(probabilities, labels):
        p = min(1.0 - 1e-9, max(1e-9, probability))
        total -= label * math.log(p) + (1 - label) * math.log(1 - p)
    return total / max(1, len(labels))


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward MLB model backtest")
    parser.add_argument("training_csv")
    parser.add_argument("--min-train", type=int, default=300)
    parser.add_argument("--refit-every", type=int, default=50)
    parser.add_argument("--minimum-edge", type=float, default=0.025)
    args = parser.parse_args()

    rows, labels = read_training_csv(args.training_csv)
    with Path(args.training_csv).open("r", encoding="utf-8-sig", newline="") as handle:
        raw = list(csv.DictReader(handle))
    if len(rows) <= args.min_train:
        raise ValueError("Not enough games for the requested walk-forward window")

    probabilities: list[float] = []
    actuals: list[int] = []
    wager_profit = 0.0
    wager_count = 0
    model = None
    for index in range(args.min_train, len(rows)):
        if model is None or (index - args.min_train) % args.refit_every == 0:
            model = CalibratedLogisticModel().fit(rows[:index], labels[:index])
        probability = model.predict_home_win(rows[index])
        probabilities.append(probability)
        actuals.append(labels[index])
        home_odds = float(raw[index].get("home_odds") or 0.0)
        away_odds = float(raw[index].get("away_odds") or 0.0)
        if home_odds > 1.0 and away_odds > 1.0:
            decisions = evaluate_market(probability, home_odds, away_odds, args.minimum_edge)
            bets = [decision for decision in decisions if decision.action == "BET"]
            if bets:
                best = max(bets, key=lambda decision: decision.expected_return)
                won = labels[index] == (1 if best.side == "home" else 0)
                wager_profit += best.odds - 1.0 if won else -1.0
                wager_count += 1

    accuracy = sum((p >= 0.5) == bool(y) for p, y in zip(probabilities, actuals)) / len(actuals)
    print(f"games={len(actuals)}")
    print(f"accuracy={accuracy:.4f}")
    print(f"brier={brier_score(probabilities, actuals):.4f}")
    print(f"log_loss={_log_loss(probabilities, actuals):.4f}")
    print(f"bets={wager_count}")
    print(f"flat_bet_roi={(wager_profit / wager_count if wager_count else 0.0):.4f}")


if __name__ == "__main__":
    main()
