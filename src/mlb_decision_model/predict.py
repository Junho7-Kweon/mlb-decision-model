from __future__ import annotations

import argparse
import json
from pathlib import Path

from .decision import evaluate_market
from .features import build_game_features
from .model import CalibratedLogisticModel


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict and evaluate both MLB moneyline sides")
    parser.add_argument("model_json")
    parser.add_argument("game_json")
    parser.add_argument("--home-odds", type=float, required=True)
    parser.add_argument("--away-odds", type=float, required=True)
    args = parser.parse_args()
    game = json.loads(Path(args.game_json).read_text(encoding="utf-8"))
    model = CalibratedLogisticModel.load(args.model_json)
    features = build_game_features(game)
    probability = model.predict_home_win(features)
    home, away = evaluate_market(probability, args.home_odds, args.away_odds)
    payload = {
        "home_win_probability": probability,
        "away_win_probability": 1.0 - probability,
        "features": features,
        "market": [home.__dict__, away.__dict__],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
