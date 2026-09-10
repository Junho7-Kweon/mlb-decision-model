from __future__ import annotations

"""오늘 경기 피처 입력 -> model_probability.py가 읽는 model_probabilities.json 생성.

이 파일이 바로 예전 로드맵 문서에서 "당일 정보를 구조화해서 직접 넣는 도구가
아직 없다"고 지적했던 그 빈 부분을 채우는 스크립트다. 오늘 경기의 선발·불펜·
라인업 등은 Retrosheet(과거 데이터)나 OCR(배당만 줌)로는 못 채우므로, 사람이
직접 조사해서 이 입력 JSON에 채워 넣어야 한다 - 이건 정책상 불가피한 부분이다.

*** event_id 설계 ***
GPT가 만든 model_probability.py의 event_id는 문자열 매칭(팀명 텍스트)이라
OCR이 뽑아낸 경기명과 정확히 일치해야 한다. 이건 실제로 계속 문제가
됐던 방식이라(팀명 표기가 매번 조금씩 다름), 이 스크립트는 event_id를
"팀명 문자열"이 아니라 사용자가 입력 JSON에서 직접 지정한 안정적인 키를
그대로 쓴다 - 다만 대시보드에서 실제 매칭이 되려면 그 키가 OCR로 뽑힌
경기명과 일치해야 하므로, 사용자가 OCR 결과의 "경기명"란을 이 스크립트의
event_id와 동일하게 맞춰서 입력하는 걸 권장한다(임시방편, 근본 해결은
canonical ID 도입 - docs/OCR_REVIEW_2026-09-04.md에 남겨둔 이슈와 같은 계열).
"""

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .score_model import (
    TeamScoreModel,
    handicap_probability,
    moneyline_probability,
    sum_probability,
    three_way_probability,
    total_probability,
    win_one_loss_probability,
)
from .features import FEATURE_NAMES


def build_predictions(games: list[dict[str, Any]], model: TeamScoreModel) -> list[dict[str, Any]]:
    period = "full" if model.segment == "full" else "first_five"
    predictions: list[dict[str, Any]] = []
    seen = set()
    for game in games:
        event_id = str(game["event_id"]).strip()
        if not event_id or event_id in seen:
            raise ValueError("event_id must be nonempty and unique (include date/game number)")
        seen.add(event_id)
        missing = set(FEATURE_NAMES) - set(game["features"])
        if missing or not all(math.isfinite(float(game["features"][key])) for key in FEATURE_NAMES):
            raise ValueError(f"Incomplete or non-finite pregame features: {sorted(missing)}")
        if game.get("selection_home_side", "home") != "home":
            raise ValueError("Confirm capture 승 and handicap sign refer to the home team")
        grid = model.score_grid(game["features"])
        if model.segment == "full":
            for selection in ("승", "패"):
                predictions.append({
                    "event_id": event_id, "period": period, "market": "moneyline",
                    "selection": selection, "line": None,
                    "probability": round(moneyline_probability(grid, selection), 4),
                })
            for selection in ("승", "1", "패"):
                predictions.append({
                    "event_id": event_id, "period": period, "market": "three_way",
                    "selection": selection, "line": None,
                    "probability": round(win_one_loss_probability(grid, selection), 4),
                })
            for selection in ("홀", "짝"):
                predictions.append({
                    "event_id": event_id, "period": period, "market": "sum",
                    "selection": selection, "line": None,
                    "probability": round(sum_probability(grid, selection), 4),
                })
        else:
            for selection in ("승", "무", "패"):
                predictions.append({
                    "event_id": event_id, "period": period, "market": "three_way",
                    "selection": selection, "line": None,
                    "probability": round(three_way_probability(grid, selection), 4),
                })
        for line in game.get("total_lines", []):
            for selection in ("언더", "오버"):
                predictions.append({
                    "event_id": event_id, "period": period, "market": "total",
                    "selection": selection, "line": line,
                    "probability": round(total_probability(grid, selection, line), 4),
                })
        for line in game.get("handicap_lines", []):
            for selection in ("승", "패"):
                predictions.append({
                    "event_id": event_id, "period": period, "market": "handicap",
                    "selection": selection, "line": line,
                    "probability": round(handicap_probability(grid, selection, line), 4),
                })
        for prediction in predictions:
            if prediction["event_id"] == event_id:
                prediction["event_aliases"] = list(game.get("event_aliases", []))
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="오늘 경기 피처(선발/불펜 등, 사람이 조사해서 입력)로 model_probabilities.json 생성"
    )
    parser.add_argument("model_json", help="train_score_model.py로 학습한 모델")
    parser.add_argument("games_json", help='{"games":[{"event_id":..,"features":{...},"total_lines":[7.5]}]}')
    parser.add_argument("--out", type=Path, default=Path("data/private/model_probabilities.json"))
    args = parser.parse_args()

    model = TeamScoreModel.load(args.model_json)
    games = json.loads(Path(args.games_json).read_text(encoding="utf-8"))["games"]
    now = datetime.now(timezone.utc)
    for game in games:
        if not game.get("as_of_timestamp") or not game.get("starts_at"):
            raise ValueError("Each game requires as_of_timestamp and starts_at with timezone")
        as_of = datetime.fromisoformat(game["as_of_timestamp"])
        starts = datetime.fromisoformat(game["starts_at"])
        if as_of.tzinfo is None or starts.tzinfo is None or not as_of <= now < starts:
            raise ValueError("Prediction requires a past pregame snapshot and a future game start")
    predictions = build_predictions(games, model)

    payload = {
        "model_version": Path(args.model_json).stem,
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "predictions": predictions,
        "expires_at": min(datetime.fromisoformat(game["starts_at"]) for game in games).isoformat() if games else now.isoformat(),
        "assumptions": ["independent_negative_binomial", "moneyline_conditions_on_non_tie", "home_side_selection"],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"games={len(games)} predictions={len(predictions)} -> {args.out}")


if __name__ == "__main__":
    main()
