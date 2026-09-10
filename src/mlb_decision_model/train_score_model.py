from __future__ import annotations

"""Train TeamScoreModel from a retrosheet_etl.py training CSV.

retrosheet_etl.py가 이미 home_runs/away_runs/home_f5_runs/away_f5_runs를
정답값으로 CSV에 담아두므로, 이 파일은 그 CSV를 읽어서 정규경기용 모델
하나(선택적으로 F5용 모델도)를 학습·저장하기만 하면 된다.
"""

import argparse
import csv
import json
import math
from pathlib import Path

from .features import FEATURE_NAMES, FEATURE_VERSION
from .score_model import TeamScoreModel
from .score_model import moneyline_probability
from .model import CalibratedLogisticModel, brier_score


def read_score_training_csv(
    path: str | Path, *, segment: str = "full"
) -> tuple[list[dict[str, float]], list[int], list[int]]:
    """segment: 'full'(정규경기) 또는 'f5'(전반 5회)."""
    if segment not in {"full", "f5"}:
        raise ValueError("segment must be full or f5")
    home_col = "home_runs" if segment == "full" else "home_f5_runs"
    away_col = "away_runs" if segment == "full" else "away_f5_runs"
    rows: list[dict[str, float]] = []
    home_runs: list[int] = []
    away_runs: list[int] = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(FEATURE_NAMES).difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing feature columns: {sorted(missing)}")
        for record in reader:
            if record.get("feature_version") != FEATURE_VERSION:
                raise ValueError(
                    f"Row uses feature_version {record.get('feature_version')!r}, "
                    f"expected {FEATURE_VERSION}"
                )
            home_value, away_value = record.get(home_col), record.get(away_col)
            if not home_value or not away_value:
                continue  # F5 라벨이 없는 경기(teamstats 누락 등)는 건너뜀
            rows.append({name: float(record[name]) for name in FEATURE_NAMES})
            if not all(math.isfinite(value) for value in rows[-1].values()):
                raise ValueError("Non-finite training feature")
            if any(not math.isfinite(float(v)) or float(v) < 0 or not float(v).is_integer() for v in (home_value, away_value)):
                raise ValueError("Scores must be nonnegative integers")
            home_runs.append(int(float(home_value)))
            away_runs.append(int(float(away_value)))
    return rows, home_runs, away_runs


def evaluate_holdout(path: str | Path, *, baseline_csv: str | None = None) -> dict:
    """Fixed chronological 80/20 holdout; calibration stays inside training.

    This is a single holdout, not a rolling refit walk-forward. The final 20%
    never participates in scaling, fitting or calibration.
    """
    rows, home, away = read_score_training_csv(path)
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        records = list(csv.DictReader(handle))
    times = [r.get("as_of_timestamp", "") for r in records]
    dates = [value[:10] for value in times]
    if len(records) != len(rows) or not all(times) or dates != sorted(dates):
        raise ValueError("Evaluation requires complete rows sorted by as_of_timestamp")
    split = int(len(rows) * .8)
    if split < 20 or len(rows) - split < 10:
        raise ValueError("Not enough data for chronological holdout")
    # Keep all games on the split date on one side of the boundary.
    boundary = times[split][:10]
    while split > 0 and times[split-1][:10] == boundary:
        split -= 1
    labels = [int(h > a) for h, a in zip(home, away)]
    actual = labels[split:]
    def metrics(probabilities):
        return {
            "brier": brier_score(probabilities, actual),
            "log_loss": -sum(y*math.log(max(1e-12,p))+(1-y)*math.log(max(1e-12,1-p)) for p,y in zip(probabilities,actual))/len(actual),
            "accuracy": sum((p>=.5)==bool(y) for p,y in zip(probabilities,actual))/len(actual),
        }
    rate = sum(labels[:split])/split
    report = {"method":"chronological_80_20_holdout", "train_games":split,"test_games":len(actual),"test_start":times[split],"baseline":metrics([rate]*len(actual)),"models":{}}
    datasets = {"updated": rows}
    if baseline_csv:
        old, old_home, old_away = read_score_training_csv(baseline_csv)
        with Path(baseline_csv).open(encoding="utf-8-sig",newline="") as handle:
            old_ids=[r["event_id"] for r in csv.DictReader(handle)]
        if old_ids != [r["event_id"] for r in records] or old_home != home or old_away != away:
            raise ValueError("Baseline and updated CSV must have identical games and labels")
        datasets["previous"] = old
    for label, values in datasets.items():
        print(f"Evaluating {label}: logistic", flush=True)
        logistic = CalibratedLogisticModel().fit(values[:split], labels[:split])
        report["models"][label+"_logistic"] = metrics([logistic.predict_home_win(r) for r in values[split:]])
        print(f"Evaluating {label}: score distribution", flush=True)
        score = TeamScoreModel().fit(values[:split], home[:split], away[:split])
        report["models"][label+"_negative_binomial"] = metrics([moneyline_probability(score.score_grid(r),"승") for r in values[split:]])
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the joint score-distribution model")
    parser.add_argument("training_csv")
    parser.add_argument("model_json")
    parser.add_argument("--segment", choices=["full", "f5"], default="full")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate only; never overwrite the model")
    parser.add_argument("--baseline-csv")
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--exclude-features",
        nargs="*",
        default=["bvp_edge"],
        help="학습에서 0으로 고정할 피처. 기본값: bvp_edge",
    )
    args = parser.parse_args()
    unknown = set(args.exclude_features).difference(FEATURE_NAMES)
    if unknown:
        raise ValueError(f"Unknown excluded features: {sorted(unknown)}")
    if args.evaluate:
        report = evaluate_holdout(args.training_csv, baseline_csv=args.baseline_csv)
        content = json.dumps(report, indent=2, ensure_ascii=False)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(content, encoding="utf-8")
        print(content)
        return
    rows, home_runs, away_runs = read_score_training_csv(args.training_csv, segment=args.segment)
    for row in rows:
        for name in args.exclude_features:
            row[name] = 0.0
    if len(rows) < 20:
        raise SystemExit(
            f"학습 가능한 행이 {len(rows)}개뿐입니다(최소 20개 필요). "
            f"{'F5 라벨(teamstats.csv)이 부족한 것 같습니다.' if args.segment=='f5' else 'Retrosheet 기간을 넓혀보세요.'}"
        )
    model = TeamScoreModel(segment=args.segment).fit(rows, home_runs, away_runs)
    model.save(args.model_json)
    print(
        f"games={len(rows)} segment={args.segment} "
        f"distribution=negative_binomial excluded={','.join(args.exclude_features) or '-'} "
        f"dispersion(home={model.home_dispersion:.3f}, away={model.away_dispersion:.3f}) "
        f"-> {args.model_json}"
    )


if __name__ == "__main__":
    main()
