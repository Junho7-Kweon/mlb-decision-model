from __future__ import annotations

"""Join verified market offers to probabilities produced by the model pipeline.

The dashboard never asks a user to type a model probability and never substitutes
bookmaker implied probability for a missing model result.  The future prediction
pipeline writes the ERD ``MARKET_PROBABILITY`` projection to the private JSON file
consumed here; the dashboard performs only a deterministic, read-only join.
"""

import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .features import FEATURE_NAMES
from .score_model import (
    TeamScoreModel,
    handicap_probability,
    moneyline_probability,
    sum_probability,
    three_way_probability,
    total_probability,
    win_one_loss_probability,
)


class ModelProbabilityUnavailable(ValueError):
    """Raised when one or more captured markets have no model probability."""

    def __init__(self, message: str, *, missing: list[str] | None = None):
        super().__init__(message)
        self.missing = missing or []


@dataclass(frozen=True)
class ModelProbabilityStatus:
    status: str
    model_version: str | None
    generated_at: str | None
    prediction_count: int
    message: str


def _text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


def _line(value: Any) -> str:
    if value in (None, ""):
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return _text(value)
    if not math.isfinite(number):
        return _text(value)
    return f"{number:+.2f}"


def probability_key(row: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        _text(row.get("event_id")),
        _text(row.get("period") or "full"),
        _text(row.get("market")),
        _text(row.get("selection")),
        _line(row.get("line")),
    )


def _read_projection(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ModelProbabilityUnavailable(
            "아직 예측 모델이 준비되지 않았습니다. 지금은 OCR로 경기와 배당을 "
            "확인할 수 있으며, 모델 학습이 완료되면 확률과 Top 5 추천이 자동으로 "
            "표시됩니다. 모델 확률을 직접 입력할 필요는 없습니다."
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelProbabilityUnavailable(
            "모델 확률 산출물을 읽을 수 없습니다. MARKET_PROBABILITY 파일을 확인하세요."
        ) from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("predictions"), list):
        raise ModelProbabilityUnavailable(
            "모델 확률 산출물 형식이 올바르지 않습니다. predictions 배열이 필요합니다."
        )
    if payload.get("expires_at"):
        try:
            expiry = datetime.fromisoformat(payload["expires_at"])
            if expiry.tzinfo is None or expiry <= datetime.now(timezone.utc):
                raise ValueError("expired")
        except (TypeError, ValueError) as exc:
            raise ModelProbabilityUnavailable("경기 시작 시각이 지난 예측입니다. 새 경기의 사전 데이터로 다시 예측하세요.") from exc
    return payload


def model_probability_status(path: Path) -> ModelProbabilityStatus:
    try:
        payload = _read_projection(path)
    except ModelProbabilityUnavailable as exc:
        return ModelProbabilityStatus("not_ready", None, None, 0, str(exc))
    predictions = payload["predictions"]
    valid = 0
    for row in predictions:
        try:
            probability = float(row.get("probability"))
        except (AttributeError, TypeError, ValueError):
            continue
        valid += int(0.001 <= probability <= 0.999)
    status = "ready" if valid else "not_ready"
    message = (
        f"모델 확률 {valid}건이 자동 연결 준비되었습니다."
        if valid
        else "유효한 모델 확률이 없습니다. 점수분포 예측을 먼저 생성하세요."
    )
    return ModelProbabilityStatus(
        status,
        str(payload.get("model_version") or "unknown"),
        str(payload.get("generated_at") or ""),
        valid,
        message,
    )


def attach_model_probabilities(
    raw_picks: list[dict[str, Any]], projection_path: Path
) -> tuple[list[dict[str, Any]], ModelProbabilityStatus]:
    """Attach model output to captured offers without using odds as probability."""
    unresolved = [row for row in raw_picks if row.get("probability") in (None, "", 0, 0.0)]
    if not unresolved:
        return raw_picks, ModelProbabilityStatus(
            "ready", "request", None, len(raw_picks), "요청에 모델 확률이 포함되었습니다."
        )

    payload = _read_projection(projection_path)
    index: dict[tuple[str, str, str, str, str], float] = {}
    for prediction in payload["predictions"]:
        if not isinstance(prediction, dict):
            continue
        try:
            probability = float(prediction.get("probability"))
        except (TypeError, ValueError):
            continue
        if 0.001 <= probability <= 0.999:
            for event in [prediction.get("event_id"), *prediction.get("event_aliases", [])]:
                key = probability_key({**prediction,"event_id":event})
                if key in index and index[key] != probability:
                    raise ModelProbabilityUnavailable("같은 경기·마켓에 서로 다른 확률이 있습니다. 경기 ID/별칭 중복을 확인하세요.")
                index[key] = probability

    resolved: list[dict[str, Any]] = []
    missing: list[str] = []
    for raw in raw_picks:
        row = dict(raw)
        if row.get("probability") in (None, "", 0, 0.0):
            probability = index.get(probability_key(row))
            if probability is None:
                missing.append(str(row.get("name") or row.get("event_id") or "알 수 없는 선택"))
            else:
                row["probability"] = probability
                row["probability_source"] = "model"
        resolved.append(row)

    if missing:
        preview = ", ".join(missing[:3])
        suffix = f" 외 {len(missing) - 3}건" if len(missing) > 3 else ""
        raise ModelProbabilityUnavailable(
            f"모델 확률을 찾지 못한 선택지가 있습니다: {preview}{suffix}. "
            "경기 ID·마켓·기준점 매핑을 확인하세요.",
            missing=missing,
        )

    return resolved, ModelProbabilityStatus(
        "ready",
        str(payload.get("model_version") or "unknown"),
        str(payload.get("generated_at") or ""),
        len(index),
        f"모델 확률 {len(resolved)}건을 자동 연결했습니다.",
    )


def _probability_from_grid(row: dict[str, Any], grid: list[list[float]]) -> float:
    market = _text(row.get("market"))
    period = _text(row.get("period") or "full")
    selection = str(row.get("selection") or "").strip()
    line = row.get("line")
    if market == "moneyline":
        return moneyline_probability(grid, selection)
    if market == "three_way":
        return (
            three_way_probability(grid, selection)
            if period == "first_five"
            else win_one_loss_probability(grid, selection)
        )
    if market == "sum":
        return sum_probability(grid, selection)
    if market in {"total", "handicap"}:
        if line in (None, ""):
            raise ValueError(f"{market} 기준점이 없습니다")
        value = float(line)
        return (
            total_probability(grid, selection, value)
            if market == "total"
            else handicap_probability(grid, selection, value)
        )
    raise ValueError(f"지원하지 않는 마켓입니다: {market or '미확인'}")


def attach_score_model_probabilities(
    raw_picks: list[dict[str, Any]],
    full_model_path: Path,
    first_five_model_path: Path | None = None,
) -> tuple[list[dict[str, Any]], ModelProbabilityStatus]:
    """Calculate a transparent neutral-game baseline when no daily projection exists.

    This is still a learned score distribution; bookmaker odds never enter the
    probability calculation.  All signed pregame edges are set to zero, so the
    result must be labelled as a baseline rather than a team-specific daily view.
    """
    if not full_model_path.is_file():
        raise ModelProbabilityUnavailable("학습된 정규경기 점수분포 모델이 없습니다.")
    models = {"full": TeamScoreModel.load(full_model_path)}
    if first_five_model_path and first_five_model_path.is_file():
        models["first_five"] = TeamScoreModel.load(first_five_model_path)
    neutral_features = {name: 0.0 for name in FEATURE_NAMES}
    grids: dict[tuple[str, str], list[list[float]]] = {}
    resolved: list[dict[str, Any]] = []
    missing: list[str] = []
    for raw in raw_picks:
        row = dict(raw)
        period = str(row.get("period") or "full").strip().casefold()
        model = models.get(period)
        if model is None:
            missing.append(f"{row.get('name') or row.get('event_id')} (전반 모델 없음)")
            resolved.append(row)
            continue
        event_id = _text(row.get("event_id"))
        key = (period, event_id)
        if key not in grids:
            grids[key] = model.score_grid(neutral_features)
        grid = grids[key]
        try:
            probability = _probability_from_grid(row, grid)
        except (TypeError, ValueError) as exc:
            missing.append(f"{row.get('name') or row.get('event_id')} ({exc})")
        else:
            row["probability"] = round(probability, 6)
            row["probability_source"] = "score_model_neutral_baseline"
        resolved.append(row)
    if missing:
        preview = ", ".join(missing[:3])
        suffix = f" 외 {len(missing) - 3}건" if len(missing) > 3 else ""
        raise ModelProbabilityUnavailable(
            f"점수분포에서 계산할 수 없는 선택지가 있습니다: {preview}{suffix}",
            missing=missing,
        )
    kinds = "/".join(sorted({model.model_kind for model in models.values()}))
    return resolved, ModelProbabilityStatus(
        "ready_baseline",
        kinds,
        None,
        len(resolved),
        "NB 학습 기준선으로 자동 계산했습니다. 배당은 확률 계산에 사용하지 않았으며, "
        "당일 선발·라인업·부상·날씨 피처는 아직 미반영입니다.",
    )
