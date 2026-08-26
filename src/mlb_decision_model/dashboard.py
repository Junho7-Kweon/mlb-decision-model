from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .decision import Pick, rank_combinations
from .ocr_extract import OcrNotAvailableError, extract_odds
from .snapshot import decode_image, save_snapshot, validate_picks


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parents[1]
HTML_PATH = PACKAGE_ROOT / "static" / "dashboard.html"
PRIVATE_ROOT = PROJECT_ROOT / "data" / "private" / "snapshots"
MAX_REQUEST_BYTES = 12 * 1024 * 1024
DEFAULT_TOP_N = 5


def analyze_payload(payload: dict[str, Any]) -> dict[str, Any]:
    validated = validate_picks(list(payload.get("picks", [])))
    legs = int(payload.get("legs", 2))
    simulations = min(100_000, max(1_000, int(payload.get("simulations", 50_000))))
    top_n = payload.get("top_n", DEFAULT_TOP_N)
    top_n = int(top_n) if top_n else None
    picks = [Pick(row.name, row.probability, row.odds, row.event_id) for row in validated]
    survival = rank_combinations(
        picks, legs=legs, simulations=simulations, sort_by="survival"
    )
    expected_value = rank_combinations(
        picks, legs=legs, simulations=simulations, sort_by="expected_return"
    )
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
            }
        )
    return {
        "assumption": "independent-picks",
        "warning": "같은 경기의 복수 마켓은 자동으로 한 조합에서 제외합니다. 서로 다른 경기 사이의 상관관계는 아직 독립으로 가정합니다.",
        "top_n": top_n,
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
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        if self.path not in {"/api/analyze", "/api/snapshots", "/api/ocr-extract"}:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > MAX_REQUEST_BYTES:
                raise ValueError("request must be between 1 byte and 12 MB")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if self.path == "/api/ocr-extract":
                self._json(HTTPStatus.OK, ocr_extract_payload(payload))
                return
            result = analyze_payload(payload)
            if self.path == "/api/snapshots":
                result["snapshot"] = save_snapshot(payload, PRIVATE_ROOT)
            self._json(HTTPStatus.OK, result)
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

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
