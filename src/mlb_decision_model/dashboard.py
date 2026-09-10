from __future__ import annotations

import argparse
import json
import os
import threading
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .decision import Pick, rank_combinations
from .model_probability import (
    ModelProbabilityUnavailable,
    attach_model_probabilities,
    model_probability_status,
)
from .score_model import TeamScoreModel
from .pregame import attach_pregame_probabilities, read_games
from .mlb_daily_data import refresh
from .ocr_extract import OcrNotAvailableError, extract_odds
from .snapshot import decode_image, save_snapshot, validate_picks


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parents[1]
HTML_PATH = PACKAGE_ROOT / "static" / "dashboard.html"
PRIVATE_ROOT = PROJECT_ROOT / "data" / "private" / "snapshots"
MODEL_PROBABILITIES_PATH = PROJECT_ROOT / "data" / "private" / "model_probabilities.json"
SCORE_MODEL_PATH = PROJECT_ROOT / "models" / "score_model.json"
F5_SCORE_MODEL_PATH = PROJECT_ROOT / "models" / "score_model_f5.json"
PREGAME_PATH = PROJECT_ROOT / "data" / "private" / "pregame_observations.json"
SYNC_LOCK = threading.Lock()
SYNC_STATUS = {"state": "idle", "message": "당일 데이터 갱신 대기"}
MAX_REQUEST_BYTES = 12 * 1024 * 1024
DEFAULT_SIMULATIONS = 200_000


def resolve_model_picks(raw_picks: list[dict[str, Any]]):
    if PREGAME_PATH.is_file():
        return attach_pregame_probabilities(raw_picks, PREGAME_PATH, SCORE_MODEL_PATH, F5_SCORE_MODEL_PATH)
    try:
        return attach_model_probabilities(
            raw_picks, MODEL_PROBABILITIES_PATH
        )
    except ModelProbabilityUnavailable:
        raise ModelProbabilityUnavailable("경기별 사전 데이터가 없습니다. 당일 데이터 연결 상태를 확인하고 갱신하세요.")


def analyze_payload(
    payload: dict[str, Any],
    resolved: tuple[list[dict[str, Any]], Any] | None = None,
) -> dict[str, Any]:
    raw_picks = list(payload.get("picks", []))
    model_picks, probability_status = resolved or resolve_model_picks(raw_picks)
    validated = validate_picks(model_picks, minimum=1)
    legs = int(payload.get("legs", min(2, len(validated))))
    simulations = 0  # independent combination probabilities have an exact product
    top_n = payload.get("top_n")
    top_n = int(top_n) if top_n else None
    picks = [Pick(row.name, row.probability, row.odds, row.event_id) for row in validated]
    survival = rank_combinations(
        picks, legs=legs, simulations=simulations, sort_by="survival"
    )
    expected_value = sorted(survival, key=lambda row: (row.expected_return, row.hit_probability), reverse=True)
    combination_count = len(survival)
    worst_survival = asdict(survival[-1]) if survival else None
    if top_n:
        survival = survival[:top_n]
        expected_value = expected_value[:top_n]
    diagnostics = []
    for row in validated:
        break_even = 1.0 / row.odds
        edge = row.probability - break_even
        expected_return = row.probability * row.odds - 1.0
        action = "BET" if edge >= 0.025 else "FADE" if edge <= -0.025 else "PASS"
        diagnostics.append(
            {
                "event_id": row.event_id,
                "name": row.name,
                "probability": row.probability,
                "odds": row.odds,
                "break_even": break_even,
                "edge": edge,
                "expected_return": expected_return,
                "action": action,
                "probability_source": row.probability_source,
                "feature_snapshot": model_picks[len(diagnostics)].get("feature_snapshot"),
            }
        )
    return {
        "assumption": "independent-picks",
        "warning": f"{probability_status.message} 같은 경기의 복수 마켓은 자동으로 한 조합에서 제외합니다. 서로 다른 경기 사이의 상관관계는 아직 독립으로 가정합니다.",
        "top_n": top_n,
        "simulations": simulations,
        "combination_count": combination_count,
        "worst_survival": worst_survival,
        "model_probability": asdict(probability_status),
        "pick_diagnostics": diagnostics,
        "survival_rank": [asdict(row) for row in survival],
        "ev_rank": [asdict(row) for row in expected_value],
    }


def ocr_extract_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """캡처 이미지 -> 승패/언더오버 배당 초안. 저장은 하지 않고 미리보기만 반환한다.

    OCR은 100% 로컬(Tesseract)이며 어떤 외부 서버로도 이미지를 보내지 않는다.
    이 함수는 sources.py 정책 검사를 거치지 않는다 - 데이터 출처가 아니라
    사용자가 이미 로컬에 가진 이미지를 읽는 것뿐이기 때문이다.
    """
    data_url = str(payload.get("image_data_url", ""))
    if not data_url:
        raise ValueError("image_data_url is required")
    content, _suffix = decode_image(data_url)
    try:
        drafts = extract_odds(content)
    except OcrNotAvailableError as exc:
        raise ValueError(
            f"로컬 OCR을 사용할 수 없습니다 ({exc}). 수동으로 입력해 주세요."
        ) from exc
    return {
        "drafts": [
            {
                "market": d.market,
                "label_a": d.label_a,
                "odds_a": d.odds_a,
                "label_b": d.label_b,
                "odds_b": d.odds_b,
                "confidence": d.confidence,
                "event_name": d.event_name,
                "period": d.period,
                "market_line": d.market_line,
                "label_c": d.label_c,
                "odds_c": d.odds_c,
                "selections": [
                    {"label": label, "odds": odds}
                    for label, odds in d.selections()
                    if odds is not None
                ],
            }
            for d in drafts
        ],
        "note": "OCR 결과는 초안입니다. 제출 전 반드시 눈으로 확인·수정하세요.",
    }


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "MlbDecisionDashboard/0.1"

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        content = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/":
            content = HTML_PATH.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(content)
            return
        if self.path == "/api/source-status":
            self._json(
                HTTPStatus.OK,
                {
                    "official_results": {
                        "dataset_id": "15107776",
                        "status": "allowed",
                        "delay_days": 14,
                        "contains_odds": False,
                    },
                    "live_odds": {
                        "status": "manual_snapshot",
                        "reason": "공식 공공 API에 실시간 배당 필드가 없습니다.",
                    },
                },
            )
            return
        if self.path == "/api/model-status":
            status = asdict(model_probability_status(MODEL_PROBABILITIES_PATH))
            status["daily_data"] = {**SYNC_STATUS, "provider": "MLB Stats API", "requires_api_key": False}
            status["score_model_available"] = SCORE_MODEL_PATH.is_file()
            status["first_five_model_available"] = F5_SCORE_MODEL_PATH.is_file()
            status["supported_markets"] = ["moneyline", "three_way", "handicap", "total", "sum"]
            if status["status"] != "ready" and SCORE_MODEL_PATH.is_file():
                try:
                    model = TeamScoreModel.load(SCORE_MODEL_PATH)
                except (OSError, ValueError):
                    pass
                else:
                    status["status"] = "not_ready"
                    status["model_version"] = model.model_kind
                    status["message"] = "NB 모델 학습 완료 · 경기별 사전 데이터 연결 대기. 당일 자료가 있어야 확률과 추천을 계산합니다."
            if PREGAME_PATH.is_file():
                try:
                    games = read_games(PREGAME_PATH)
                except ModelProbabilityUnavailable as exc:
                    status.update(status="not_ready", message=str(exc))
                else:
                    status.update(status="ready" if games else "not_ready", prediction_count=0,
                                  message=f"경기별 사전 데이터 {len(games)}경기 준비. 캡처 팀명·마켓에 맞춰 확률을 계산합니다." if games else "유효한 시작 전 경기 자료가 없습니다. 당일 자료를 갱신하세요.")
                    status["games"] = [{"event_id": g["event_id"], "starts_at": g["starts_at"], "aliases": g.get("event_aliases", [])} for g in games]
            self._json(
                HTTPStatus.OK,
                status,
            )
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_OPTIONS(self) -> None:  # noqa: N802
        if self.path.startswith("/api/"):
            self.send_response(HTTPStatus.NO_CONTENT)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.end_headers()
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        if self.path not in {"/api/analyze", "/api/snapshots", "/api/ocr-extract", "/api/daily-refresh"}:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > MAX_REQUEST_BYTES:
                raise ValueError("request must be between 1 byte and 12 MB")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if self.path == "/api/daily-refresh":
                if not SYNC_LOCK.acquire(blocking=False):
                    self._json(HTTPStatus.ACCEPTED, dict(SYNC_STATUS))
                    return
                SYNC_STATUS.update(state="running", message="당일 자료를 수집 중입니다. 최초 실행은 과거 기록 동기화도 필요합니다.")
                def sync():
                    try:
                        result = refresh(PREGAME_PATH, date=payload.get("date"), progress=lambda message: SYNC_STATUS.update(message=message))
                        SYNC_STATUS.update(state="done", message=f"{result['ready_games']}경기 준비", result=result)
                    except Exception as exc:
                        SYNC_STATUS.update(state="error", message=f"수집 실패: {exc}")
                    finally:
                        SYNC_LOCK.release()
                threading.Thread(target=sync, daemon=True).start()
                self._json(HTTPStatus.ACCEPTED, dict(SYNC_STATUS))
                return
            if self.path == "/api/ocr-extract":
                self._json(HTTPStatus.OK, ocr_extract_payload(payload))
                return
            # Browser requests contain offers only; probabilities must come
            # from the server's generated projection, never client input.
            payload["picks"] = [
                {key:value for key,value in row.items() if key not in {"probability","probability_source"}}
                for row in payload.get("picks", [])
            ]
            resolved = None
            if self.path == "/api/snapshots":
                resolved = resolve_model_picks(list(payload.get("picks", [])))
                resolved_picks, _ = resolved
                payload = {**payload, "picks": resolved_picks}
            result = analyze_payload(payload, resolved=resolved)
            if self.path == "/api/snapshots":
                result["snapshot"] = save_snapshot(payload, PRIVATE_ROOT)
            self._json(HTTPStatus.OK, result)
        except ModelProbabilityUnavailable as exc:
            self._json(
                HTTPStatus.CONFLICT,
                {
                    "error": str(exc),
                    "code": "model_probability_unavailable",
                    "missing": exc.missing,
                },
            )
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except Exception as exc:  # noqa: BLE001 - 마지막 안전망: 예상 못한 에러도 조용히 죽지 않고 화면에 보이게
            self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"예상하지 못한 오류: {exc}"})

    def log_message(self, message: str, *args: object) -> None:
        print(f"[dashboard] {self.address_string()} - {message % args}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local MLB decision dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "localhost"}:
        raise ValueError("dashboard is intentionally restricted to localhost")
    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    print(f"Dashboard: http://{args.host}:{args.port}")
    print("Private snapshots stay under data/private and are ignored by Git.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
