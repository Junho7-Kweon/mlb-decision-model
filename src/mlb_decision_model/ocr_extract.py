from __future__ import annotations

"""
로컬 OCR 보조 배당 추출 (외부 전송 없음, 데이터 출처 정책과 무관)
================================================================

이 모듈은 "데이터 출처"가 아닙니다. 사용자가 이미 로컬에 캡처해 둔 이미지 파일에서
텍스트를 읽어내는 것뿐이라 sources.py의 정책 검사 대상이 아닙니다 - 외부 서버에
아무것도 요청하지 않습니다 (import 자체에도 requests/urllib이 없습니다).

전제 조건 (전부 로컬 설치, 선택사항):
    apt-get install tesseract-ocr tesseract-ocr-kor
    pip install pytesseract pillow

Tesseract가 없으면 OcrNotAvailableError를 던지고, 대시보드는 자동으로 완전 수동
입력 모드로 넘어갑니다. 즉 이 기능은 100% 선택 사항입니다.

*** 실제 캡처 3장으로 직접 테스트해서 설계한 파서입니다 ***
프로토 캡처의 OCR 결과는 "라벨 줄 -> 숫자 줄"이 번갈아 나오는 구조였고(승패, 승1패,
핸디캡, 언더오버, SUM 순서 고정), "패"/"언더" 같은 라벨은 매번 정확히 읽혔지만
"승"/"홀" 같은 다른 라벨은 자주 깨졌습니다("Fy", "é", "=" 등으로). 그래서 라벨
텍스트가 아니라 **줄 순서와 "언더" 앵커 키워드**로 위치를 잡습니다:
  - 첫 번째로 숫자가 2개 이상 나오는 줄 = 승패(모든 캡처에서 항상 첫 번째 마켓)
  - "언더"가 포함된 줄 바로 다음 줄 = 언더오버

승1패/핸디캡/SUM은 두 번째 숫자가 자주 깨져서(예: "2.77" -> "277") 자동 추출
대상에서 제외했습니다. 이 두 마켓(승패, 언더오버)이 마침 조합 추천에 실제로
쓰는 마켓이기도 합니다.

*** 반드시 사람이 확인해야 합니다 ***
extract_odds()의 결과는 항상 "초안(draft)"입니다. 팀명·경기명은 OCR로 자동
추출하지 않고 사람이 입력합니다(오독 위험이 가장 큰 부분이라 일부러 뺐습니다).
"""

import re
from dataclasses import dataclass


class OcrNotAvailableError(RuntimeError):
    """pytesseract/Tesseract가 로컬에 설치되어 있지 않을 때 발생."""


@dataclass(frozen=True)
class DraftOdds:
    market: str             # "moneyline" 또는 "total"
    label_a: str             # "승" 또는 "언더"
    odds_a: float | None
    label_b: str             # "패" 또는 "오버"
    odds_b: float | None
    confidence: str           # "ok" | "partial" - 확인 UI에서 경고 표시용


_ODDS = re.compile(r"\d\.\d{2}")


def extract_text(image_bytes: bytes) -> str:
    """이미지 바이트에서 원본 텍스트를 뽑는다. Tesseract 미설치 시 예외."""
    try:
        import io
        import pytesseract
        from PIL import Image
    except ImportError as exc:
        raise OcrNotAvailableError(
            "pytesseract/Pillow가 없습니다. "
            "pip install pytesseract pillow 후 시스템에 tesseract-ocr(+ tesseract-ocr-kor)을 설치하세요."
        ) from exc
    image = Image.open(io.BytesIO(image_bytes))
    return pytesseract.image_to_string(image, lang="kor+eng")


def _numbers_on_line(line: str) -> list[float]:
    return [float(m) for m in _ODDS.findall(line)]


def _make(market: str, label_a: str, label_b: str, nums: list[float]) -> DraftOdds:
    confidence = "ok" if len(nums) == 2 else "partial"
    return DraftOdds(
        market, label_a, nums[0] if len(nums) > 0 else None,
        label_b, nums[1] if len(nums) > 1 else None, confidence,
    )


def parse_draft_odds(raw_text: str) -> list[DraftOdds]:
    """OCR 원본 텍스트에서 승패·언더오버 배당만 뽑는다 (실측 검증된 두 마켓).

    라벨 텍스트가 아니라 줄 순서 기반으로 찾는다 - "승/홀" 같은 라벨은 OCR로
    자주 깨지지만 숫자 줄 순서와 "언더" 키워드는 실측 3건에서 전부 안정적이었다.
    """
    lines = raw_text.splitlines()
    results: list[DraftOdds] = []

    # 1) 승패: 숫자가 2개 이상 있는 첫 번째 줄 (레이아웃상 항상 첫 마켓)
    for line in lines:
        nums = _numbers_on_line(line)
        if len(nums) >= 2:
            results.append(_make("moneyline", "승", "패", nums[:2]))
            break

    # 2) 언더오버: "언더"가 포함된 '순수 라벨 줄' 바로 다음 줄.
    #    "야구 언더오버 U/O 7.5" 같은 마켓 헤더 줄도 "언더"를 포함하므로,
    #    "야구"가 없는(=헤더가 아닌) 줄만 라벨 줄로 인정한다.
    for i, line in enumerate(lines):
        if "언더" in line and "야구" not in line and i + 1 < len(lines):
            nums = _numbers_on_line(lines[i + 1])
            results.append(_make("total", "언더", "오버", nums[:2]))
            break

    return results


def extract_odds(image_bytes: bytes) -> list[DraftOdds]:
    """이미지 -> 승패/언더오버 배당 초안. 항상 사람 확인 후 제출해야 한다."""
    return parse_draft_odds(extract_text(image_bytes))
