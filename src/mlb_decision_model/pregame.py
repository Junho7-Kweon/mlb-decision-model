"""Validated pregame observations -> the Retrosheet v2 feature contract."""
from __future__ import annotations

import json
import math
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .model_probability import ModelProbabilityStatus, ModelProbabilityUnavailable, _probability_from_grid
from .retrosheet_etl import team_matchup_edge, MIN_GAMES_FOR_TEAM_PCT
from .score_model import TeamScoreModel
from .sources import require_approved_source


CONTRACT = "retrosheet_decay_v2"


def timestamp(value):
    result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("경기·자료 시각에 시간대가 필요합니다")
    return result


def number(record, name):
    value = record.get(name)
    if value is None or isinstance(value, bool):
        raise ValueError(f"누락된 경기 전 자료: {name}")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"잘못된 경기 전 자료: {name}")
    return value


def build_features(game):
    if game.get("feature_contract") != CONTRACT:
        raise ValueError("학습과 동일한 Retrosheet 감쇠 집계가 필요합니다")
    home, away = game["home"], game["away"]
    def win_pct(team):
        wins, losses = number(team, "wins"), number(team, "losses")
        return wins / (wins + losses) if wins + losses >= MIN_GAMES_FOR_TEAM_PCT else .5
    def rest(team):
        return max(-2, min(6, number(team, "days_since_last_game") - 4))
    return {
        "starter_edge": round((number(away, "starter_era") - number(home, "starter_era")) / 4, 4),
        "bullpen_edge": round((number(away, "bullpen_era") - number(home, "bullpen_era")) / 4, 4),
        "lineup_edge": round((number(home, "team_slg") - number(away, "team_slg")) * 10, 4),
        "team_matchup_edge": round(team_matchup_edge(win_pct(home), win_pct(away), number(game, "h2h_home_wins"), number(game, "h2h_away_wins")), 4),
        "defense_edge": round((number(away, "errors_per_game") - number(home, "errors_per_game")) * 2, 4),
        "rest_edge": round((rest(home) - rest(away)) / 4, 4),
        # These inputs have no trained effect in the deployed model.
        "bvp_edge": 0.0, "availability_edge": 0.0, "weather_edge": 0.0,
    }


def read_games(path: Path, now=None):
    now = now or datetime.now(timezone.utc)
    if not path.is_file():
        raise ModelProbabilityUnavailable("당일 경기 자료가 없습니다. 한국 기준 경기 날짜를 선택하고 당일 정보 갱신을 누르세요.")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        games, seen = [], set()
        for game in payload["games"]:
            require_approved_source(game["source_id"])
            if not game.get("source_url") or not game.get("source_version"):
                raise ValueError("자료 출처와 버전이 필요합니다")
            if not game.get("event_id") or game["event_id"] in seen:
                raise ValueError("경기 ID가 없거나 중복됐습니다")
            seen.add(game["event_id"])
            start, as_of, retrieved = map(timestamp, (game["starts_at"], game["as_of"], game["retrieved_at"]))
            if not as_of <= retrieved <= now or as_of >= start:
                raise ValueError("경기 전 자료 시각이 올바르지 않습니다")
            # Old games must not make unrelated upcoming games unusable.
            if start <= now or now - retrieved > timedelta(hours=6):
                continue
            if game.get("status") != "scheduled":
                continue
            if not game["home"].get("starter_id") or not game["away"].get("starter_id"):
                continue
            games.append({**game, "features": build_features(game)})
        return games
    except (KeyError, TypeError, ValueError, OSError) as exc:
        raise ModelProbabilityUnavailable(f"당일 자료를 검증하지 못했습니다: {exc}") from exc


def _key(value):
    # Explicit aliases only; no guessing when an OCR name is incomplete.
    return re.sub(r"[^a-z0-9가-힣]", "", str(value).casefold())


def attach_pregame_probabilities(picks, path, full_model_path, f5_model_path, now=None):
    games = read_games(path, now)
    models, grids, output, missing = {}, {}, [], []
    for raw in picks:
        event = _key(raw.get("event_id", ""))
        matches = [g for g in games if event in {_key(g["event_id"]), *(_key(a) for a in g.get("event_aliases", []))}]
        if raw.get("game_date"):
            matches = [g for g in matches if timestamp(g["starts_at"]).astimezone(timezone(timedelta(hours=9))).date().isoformat() == raw["game_date"]]
        if len(matches) != 1:
            missing.append(str(raw.get("event_id")) + (" (경기 ID로 더블헤더 구분 필요)" if matches else " (당일 자료·시작 시각·팀명 확인 필요)"))
            continue
        game = matches[0]
        period = raw.get("period") or "full"
        if period not in {"full", "first_five"}:
            raise ModelProbabilityUnavailable("지원하지 않는 경기 구간입니다")
        if period not in models:
            model = TeamScoreModel.load(full_model_path if period == "full" else f5_model_path)
            if model.segment != ("full" if period == "full" else "f5"):
                raise ModelProbabilityUnavailable("모델의 경기 구간이 일치하지 않습니다")
            if any(model.home.weights[i] or model.away.weights[i] for i, name in enumerate(model.feature_names) if name in {"bvp_edge", "availability_edge", "weather_edge"}):
                raise ModelProbabilityUnavailable("현재 당일 수집기가 제공하지 않는 피처를 사용하는 모델입니다")
            models[period] = model
        key = game["event_id"], period
        if key not in grids:
            grids[key] = models[period].score_grid(game["features"])
        row = {**raw, "event_id": game["event_id"], "probability": _probability_from_grid(raw, grids[key]), "probability_source": "pregame_score_model"}
        row["feature_snapshot"] = {k: game[k] for k in ("event_id", "source_id", "source_url", "source_version", "as_of", "retrieved_at", "starts_at", "features")}
        output.append(row)
    if missing:
        raise ModelProbabilityUnavailable(" / ".join(dict.fromkeys(missing)), missing=missing)
    return output, ModelProbabilityStatus("ready", "NB · 경기별 사전 데이터", None, len(output), "당일 선발과 최근 팀 기록으로 계산했습니다. 타자별 라인업·부상·날씨 효과는 재학습 전까지 미반영입니다.")
