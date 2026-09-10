from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .features import FEATURE_NAMES, FEATURE_VERSION
from .model import CalibratedLogisticModel, brier_score


def read_training_csv(path: str | Path) -> tuple[list[dict[str, float]], list[int]]:
    rows: list[dict[str, float]] = []
    labels: list[int] = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(FEATURE_NAMES).difference(reader.fieldnames or [])
        if "feature_version" not in (reader.fieldnames or []):
            missing.add("feature_version")
        if "home_win" not in (reader.fieldnames or []):
            missing.add("home_win")
        if missing:
            raise ValueError(f"Missing columns: {sorted(missing)}")
        for record in reader:
            if record["feature_version"] != FEATURE_VERSION:
                raise ValueError(
                    "Training row uses an incompatible feature schema; "
                    f"expected {FEATURE_VERSION}, got {record['feature_version']!r}"
                )
            rows.append({name: float(record[name]) for name in FEATURE_NAMES})
            labels.append(int(record["home_win"]))
    return rows, labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the MLB decision model")
    parser.add_argument("training_csv")
    parser.add_argument("model_json")
    args = parser.parse_args()
    rows, labels = read_training_csv(args.training_csv)
    model = CalibratedLogisticModel().fit(rows, labels)
    probabilities = [model.predict_home_win(row) for row in rows]
    model.save(args.model_json)
    print(f"games={len(rows)} brier={brier_score(probabilities, labels):.4f}")


if __name__ == "__main__":
    main()
