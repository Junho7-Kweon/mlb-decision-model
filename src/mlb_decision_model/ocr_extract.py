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

프로토 전체 마켓표는 전체 화면 OCR과 행별 숫자 전용 OCR을 함께 사용합니다.
승패, 승1패, 핸디캡, 언더오버, SUM, 전반 마켓 순서를 식별한 뒤 작은 소수점은
업스케일·고대비 숫자 패스로 다시 읽습니다. 알려진 전체 마켓표가 아니면 기존의
보수적인 승패/언더오버 파서로 되돌아갑니다.

*** 반드시 사람이 확인해야 합니다 ***
extract_odds()의 결과는 항상 "초안(draft)"입니다. 팀명·기준점·배당을 자동으로
읽더라도 사용자가 원본과 대조한 뒤에만 확정해야 합니다.
"""

import re
import statistics
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
    event_name: str | None = None
    period: str = "full"
    market_line: float | None = None
    label_c: str | None = None
    odds_c: float | None = None

    def selections(self) -> list[tuple[str, float | None]]:
        values = [(self.label_a, self.odds_a), (self.label_b, self.odds_b)]
        if self.label_c is not None:
            values.append((self.label_c, self.odds_c))
        return values


_ODDS = re.compile(r"(?<!\d)(\d{1,2})[.,](\d{2,3})(?!\d)")


def _configure_tesseract(pytesseract: object) -> None:
    import shutil
    from pathlib import Path

    if shutil.which("tesseract") is not None:
        return
    for candidate in (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ):
        if Path(candidate).is_file():
            pytesseract.pytesseract.tesseract_cmd = candidate
            return


def extract_text(image_bytes: bytes) -> str:
    """이미지 바이트에서 원본 텍스트를 뽑는다. Tesseract 미설치/PATH 문제 시 예외."""
    try:
        import io
        import pytesseract
        from PIL import Image
    except ImportError as exc:
        raise OcrNotAvailableError(
            "pytesseract/Pillow가 없습니다. "
            "pip install pytesseract pillow 후 시스템에 tesseract-ocr(+ tesseract-ocr-kor)을 설치하세요."
        ) from exc

    # Windows에서 흔한 문제: Tesseract는 설치했지만 tesseract.exe가 PATH에 없는 경우.
    # PATH에 없으면 흔히 쓰는 기본 설치 경로를 직접 찾아본다.
    _configure_tesseract(pytesseract)

    image = Image.open(io.BytesIO(image_bytes))
    try:
        return pytesseract.image_to_string(image, lang="kor+eng")
    except pytesseract.pytesseract.TesseractNotFoundError as exc:
        raise OcrNotAvailableError(
            "Tesseract 실행 파일을 찾지 못했습니다. 설치는 됐는데 PATH에 안 잡혔을 가능성이 큽니다 - "
            "설치 후 새로 연 터미널(PowerShell)에서 다시 시도해보세요. 그래도 안 되면 "
            "'C:\\Program Files\\Tesseract-OCR'이 실제로 존재하는지, 그 안에 tesseract.exe가 있는지 확인하세요."
        ) from exc
    except pytesseract.pytesseract.TesseractError as exc:
        raise OcrNotAvailableError(
            f"Tesseract 처리 중 오류가 발생했습니다: {exc}. "
            "한국어 언어팩(kor)이 제대로 설치됐는지 확인하세요."
        ) from exc


def _numbers_on_line(line: str) -> list[float]:
    values: list[float] = []
    for whole, fraction in _ODDS.findall(line):
        # Upscaled OCR occasionally adds a third decimal digit (1.584).  The
        # source UI always displays decimal odds to two places, so retain the
        # first two digits rather than rounding to a number that never appeared.
        values.append(float(f"{whole}.{fraction[:2]}"))
    return values


def _valid_decimal_odds(value: float | None) -> bool:
    return value is not None and 1.01 <= value <= 100.0


def _plausible_market_odds(values: list[float | None]) -> bool:
    """Reject a complete OCR row whose combined implied margin is implausible.

    Individual values such as 1.07 are legal decimal odds, so a range check
    cannot catch an OCR substitution like 1.67 -> 1.07.  A complete two- or
    three-way market provides an additional invariant: the sum of implied
    probabilities should remain near one, including the bookmaker margin.
    Values are retained for manual correction, but the row is marked partial.
    """
    if len(values) not in {2, 3} or not all(_valid_decimal_odds(value) for value in values):
        return False
    implied_sum = sum(1.0 / float(value) for value in values if value is not None)
    upper = 1.35 if len(values) == 2 else 1.50
    return 0.85 <= implied_sum <= upper


def _clean_event_name(value: str) -> str:
    cleaned = " ".join(value.split()).strip()
    # The tiny blue `vs` glyph is often read as `ys`, `v5`, or only its
    # trailing `s`.  A standalone Latin token between Korean team names is a
    # separator in this header, so normalise all of those variants.
    cleaned = re.sub(r"(?<=[가-힣])\s+(?:[yYvV]?[sS5])\s+(?=[가-힣])", " vs ", cleaned)
    cleaned = re.sub(r"\s*[:：]\s*", " vs ", cleaned)
    if "vs" in cleaned.casefold():
        first_hangul = re.search(r"[가-힣]", cleaned)
        if first_hangul:
            cleaned = cleaned[first_hangul.start() :]
    return cleaned[:100]


def _event_name_before_odds(lines: list[str], odds_index: int) -> str | None:
    """Best-effort event label; the user can still correct it in the dashboard."""
    blocked = ("야구", "조합", "마감", "승패", "승1패", "핸디캡", "언더오버", "SUM")
    for line in reversed(lines[:odds_index]):
        cleaned = " ".join(line.split()).strip()
        hangul = re.findall(r"[가-힣]", cleaned)
        if len(hangul) >= 5 and not any(word in cleaned for word in blocked):
            return _clean_event_name(cleaned)
    return None


def _make(
    market: str,
    label_a: str,
    label_b: str,
    nums: list[float],
    event_name: str | None = None,
) -> DraftOdds:
    safe = [value if _valid_decimal_odds(value) else None for value in nums[:2]]
    confidence = "ok" if _plausible_market_odds(safe) else "partial"
    return DraftOdds(
        market, label_a, safe[0] if len(safe) > 0 else None,
        label_b, safe[1] if len(safe) > 1 else None, confidence, event_name,
    )


def parse_draft_odds(raw_text: str) -> list[DraftOdds]:
    """OCR 원본 텍스트에서 승패·언더오버 배당만 뽑는다 (실측 검증된 두 마켓).

    이 함수는 이미지 좌표를 사용할 수 없을 때의 보수적인 폴백이다. 승패 헤더가
    실제로 확인된 경우에만 첫 배당행을 사용하며, 소수점이 사라진 배당 후보가 먼저
    보이면 뒤의 SUM 숫자를 승패로 승격하지 않는다.
    """
    lines = raw_text.splitlines()
    results: list[DraftOdds] = []
    event_name = None

    compact_lines = [re.sub(r"\s+", "", line) for line in lines]
    moneyline_anchor = next(
        (
            index
            for index, line in enumerate(compact_lines)
            if "승패" in line and "승1패" not in line
        ),
        None,
    )
    if moneyline_anchor is not None:
        suspicious_compact_odds = False
        for index in range(moneyline_anchor + 1, len(lines)):
            line = lines[index]
            compact_numbers = re.findall(r"(?<![\d.,])\d{3,4}(?![\d.,])", line)
            if compact_numbers and (len(compact_numbers) + len(_numbers_on_line(line)) >= 2):
                suspicious_compact_odds = True
            nums = _numbers_on_line(line)
            if len(nums) < 2:
                continue
            event_name = _event_name_before_odds(lines, index)
            nearby = " ".join(lines[max(moneyline_anchor, index - 2) : index]).upper()
            if (suspicious_compact_odds or "SUM" in nearby or "홀" in nearby or "짝" in nearby
                or "언더" in nearby or "오버" in nearby or len(nums) != 2
                or not re.search(r"패", nearby)):
                # There is evidence that an earlier odds row lost its decimal
                # point.  Returning no moneyline is safer than labelling a later
                # market (usually SUM) as a high-confidence win/loss offer.
                break
            results.append(_make("moneyline", "승", "패", nums[:2], event_name))
            break

    # 2) 언더오버: "언더"가 포함된 '순수 라벨 줄' 바로 다음 줄.
    #    "야구 언더오버 U/O 7.5" 같은 마켓 헤더 줄도 "언더"를 포함하므로,
    #    "야구"가 없는(=헤더가 아닌) 줄만 라벨 줄로 인정한다.
    for i, line in enumerate(lines):
        if "언더" in line and "야구" not in line and i + 1 < len(lines):
            nums = _numbers_on_line(lines[i + 1])
            results.append(_make("total", "언더", "오버", nums[:2], event_name))
            break

    return results


def _market_spec(label_text: str) -> tuple[str, str, tuple[str, ...], int, bool] | None:
    """Classify one visual market row from its left-hand label crop."""
    compact = re.sub(r"\s+", "", label_text).upper()
    period = "first_five" if "전반" in compact else "full"
    if "SUM" in compact:
        return "sum", period, ("홀", "짝"), 2, False
    if "언더" in compact or "오버" in compact:
        return "total", period, ("언더", "오버"), 2, True
    if "핸디" in compact:
        return "handicap", period, ("승", "패"), 2, True
    if "승1패" in compact or "1패" in compact or "승무패" in compact:
        labels = ("승", "무", "패") if period == "first_five" else ("승", "1", "패")
        return "three_way", period, labels, 3, False
    if "승패" in compact:
        return "moneyline", period, ("승", "패"), 2, False
    return None


def _header_starts(image: object, pytesseract: object) -> list[int]:
    """Locate repeated Proto game headers using OCR token coordinates.

    Multi-game screenshots do not have a useful global row count.  Team names,
    however, form a wide Hangul band in the centre of every game header.  This
    detector groups those tokens by y-coordinate and returns each header start.
    """
    data = pytesseract.image_to_data(
        image,
        lang="kor+eng",
        config="--psm 11",
        output_type=pytesseract.Output.DICT,
    )
    width, _height = image.size
    blocked = ("야구", "조합", "한경기", "마감", "승패", "승1패", "핸디", "언더", "오버", "SUM")
    tokens: list[tuple[int, int, int, str]] = []
    for index, raw in enumerate(data.get("text", [])):
        text = str(raw).strip()
        hangul = len(re.findall(r"[가-힣]", text))
        left = int(data["left"][index])
        if not hangul or left < width * 0.28 or left > width * 0.88:
            continue
        if any(word in text.upper() for word in blocked):
            continue
        tokens.append((int(data["top"][index]), left, hangul, text))

    bands: list[list[tuple[int, int, int, str]]] = []
    for token in sorted(tokens):
        if not bands or token[0] - max(item[0] for item in bands[-1]) > 15:
            bands.append([token])
        else:
            bands[-1].append(token)

    starts: list[int] = []
    for band in bands:
        hangul = sum(item[2] for item in band)
        lefts = [item[1] for item in band]
        wide_team_line = len(band) >= 2 and max(lefts) - min(lefts) >= width * 0.10
        if hangul < 5 or not wide_team_line:
            continue
        center_y = int(statistics.median(item[0] for item in band))
        start = max(0, center_y - 21)
        if not starts or start - starts[-1] >= 80:
            starts.append(start)
    return starts


def _odds_by_cell(
    image: object,
    top: int,
    bottom: int,
    expected: int,
    pytesseract: object,
    image_ops: object,
) -> list[float | None]:
    """OCR each visual odds button separately so missing sides keep position."""
    def cell_value(text: str) -> float | None:
        decimals = _numbers_on_line(text)
        if decimals and _valid_decimal_odds(decimals[0]):
            return decimals[0]
        return None  # Ambiguous decimal-less tokens must be manually verified.

    width, _height = image.size
    region_left = int(width * 0.49)
    region_right = int(width * 0.86)
    cell_width = (region_right - region_left) / expected
    config = "--psm 6 -c tessedit_char_whitelist=0123456789.,"
    values: list[float | None] = []
    for index in range(expected):
        left = int(region_left + cell_width * index)
        right = int(region_left + cell_width * (index + 1))
        crop = image.crop((left + 4, top + 5, right - 4, bottom - 5))
        gray = image_ops.autocontrast(
            image_ops.grayscale(crop).resize((crop.width * 5, crop.height * 5))
        )
        threshold = gray.point(lambda pixel: 255 if pixel > 190 else 0)
        value = cell_value(
            pytesseract.image_to_string(threshold, lang="eng", config=config)
        )
        if value is None:
            value = cell_value(
                pytesseract.image_to_string(gray, lang="eng", config=config)
            )
        if value is None:
            original = image.crop((left, top, right, bottom))
            original = image_ops.autocontrast(image_ops.grayscale(original).resize((original.width*3,original.height*3)))
            value = cell_value(pytesseract.image_to_string(original, lang="eng", config=config))
        values.append(value)
    if any(value is None for value in values):
        row = image.crop((int(width*.47),top,int(width*.86),bottom))
        row = image_ops.autocontrast(image_ops.grayscale(row).resize((row.width*3,row.height*3)))
        for variant in (row.point(lambda px:255 if px>190 else 0), row):
            numbers = _numbers_on_line(pytesseract.image_to_string(variant,lang="eng",config=config))
            if (len(numbers) == expected and all(_valid_decimal_odds(value) for value in numbers)
                    and all(v is None or v == numbers[i] for i, v in enumerate(values))):
                values = numbers
                break
    return values


def _market_line(text: str, *, handicap: bool) -> float | None:
    signed = re.search(r"([+-])\s*(\d{1,2}(?:[.,]\d+)?)", text)
    if signed:
        value = float(signed.group(2).replace(",", "."))
        return value if signed.group(1) == "+" else -value
    decimals = [float(value.replace(",", ".")) for value in re.findall(r"\d{1,2}[.,]\d", text)]
    if not decimals:
        return None
    value = decimals[-1]
    # In one verified capture Tesseract read the orange "H +2.5" chip as
    # "42.5".  The digit 4 is the merged H/+ glyph, not a forty-run line.
    if handicap and value > 20 and str(value).startswith("4"):
        return float(str(value)[1:])
    return value


def _scaled_row_bounds(
    header_bottom: int,
    block_end: int,
    row_count: int,
    row_index: int,
) -> tuple[int, int]:
    """Return one market row using the block's actual rendered height."""
    row_height = (block_end - header_bottom) / row_count
    top = round(header_bottom + row_height * row_index)
    bottom = round(header_bottom + row_height * (row_index + 1))
    return top, min(block_end, bottom)


def _layout_market_drafts(image_bytes: bytes, raw_text: str) -> list[DraftOdds]:
    """Read one or more Proto game blocks row by row.

    A screenshot can contain one expanded game or several consecutive games.
    We first locate every team-name header, split the image into game blocks,
    and then apply label OCR and digit-only OCR inside each block.  No global
    image-height/row-count assumption is used for multi-game captures.
    """
    try:
        import io
        import pytesseract
        from PIL import Image, ImageOps
    except ImportError:
        return []
    _configure_tesseract(pytesseract)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    width, height = image.size
    if width < 760 or height < 110:
        return []
    starts = _header_starts(image, pytesseract)
    if not starts:
        # Retain support for tightly cropped single-game tables when sparse OCR
        # cannot produce header coordinates.
        legacy_rows = round((height - 57) / 71)
        if legacy_rows not in {5, 6, 7, 8}:
            return []
        starts = [0]

    fixed_specs = [
        ("moneyline", "full", ("승", "패"), 2, False),
        ("three_way", "full", ("승", "1", "패"), 3, False),
        ("handicap", "full", ("승", "패"), 2, True),
        ("total", "full", ("언더", "오버"), 2, True),
        ("sum", "full", ("홀", "짝"), 2, False),
        ("three_way", "first_five", ("승", "무", "패"), 3, False),
        ("handicap", "first_five", ("승", "패"), 2, True),
        ("total", "first_five", ("언더", "오버"), 2, True),
    ]
    drafts: list[DraftOdds] = []
    boundaries = [*starts, height]
    for block_index, header_start in enumerate(starts):
        block_end = boundaries[block_index + 1]
        header_bottom = min(block_end, header_start + 57)
        available = block_end - header_bottom
        row_count = min(8, max(0, int((available + 10) // 71)))
        if row_count < 1:
            continue
        # Proto screenshots are rendered at different browser zoom/scaling
        # levels.  Recent 1072x455 captures have ~79 px market rows, while the
        # older fixtures use ~71 px.  Advancing by a hard-coded 71 px made the
        # fourth-row total crop drift upward until it missed the odds digits;
        # Tesseract then hallucinated the same 1.07 value for 1.89 and 1.87.
        # Once the number of rows is known, distribute the actual block height
        # across them so every row stays aligned with its visible cell.

        header_crop = image.crop(
            (int(width * 0.28), header_start, int(width * 0.82), min(block_end, header_start + 66))
        )
        header_gray = ImageOps.autocontrast(
            ImageOps.grayscale(header_crop).resize((header_crop.width * 3, header_crop.height * 3))
        )
        event_name = _clean_event_name(
            pytesseract.image_to_string(header_gray, lang="kor+eng", config="--psm 11")
        )
        # Dense and sparse passes differ on the small team-name glyphs. Prefer
        # the complete Korean header over a candidate with ASCII OCR debris.
        alternatives = [event_name]
        header_line = image.crop((int(width*.28),header_start,int(width*.82),min(block_end,header_start+50)))
        header_line = ImageOps.autocontrast(ImageOps.grayscale(header_line).resize((header_line.width*3,header_line.height*3)))
        alternatives.append(_clean_event_name(pytesseract.image_to_string(header_line,lang="kor+eng",config="--psm 6")))
        event_name = max(alternatives, key=lambda name: len(re.findall(r"[가-힣]",name))*2-len(re.findall(r"[A-Z=“]",name)))
        if len(re.findall(r"[가-힣]", event_name)) < 5:
            continue

        label_texts: list[str] = []
        for row_index in range(row_count):
            top, bottom = _scaled_row_bounds(
                header_bottom, block_end, row_count, row_index
            )
            label_crop = image.crop((int(width * 0.10), top, int(width * 0.47), bottom))
            label_gray = ImageOps.autocontrast(
                ImageOps.grayscale(label_crop).resize((label_crop.width * 2, label_crop.height * 2))
            )
            label_texts.append(
                pytesseract.image_to_string(label_gray, lang="kor+eng", config="--psm 6")
            )

        classified = [_market_spec(text) for text in label_texts]
        prefix_matches = sum(
            classified[index] is not None and classified[index][0] == fixed_specs[index][0]
            for index in range(min(5, row_count))
        )
        fixed_prefix = row_count >= 5 and prefix_matches >= 4

        for row_index in range(row_count):
            spec = classified[row_index]
            if spec is None and fixed_prefix:
                spec = fixed_specs[row_index]
            if spec is None:
                continue
            market, period, labels, expected, needs_line = spec
            top, bottom = _scaled_row_bounds(
                header_bottom, block_end, row_count, row_index
            )
            odds = _odds_by_cell(
                image, top, bottom, expected, pytesseract, ImageOps
            )

            line_value = None
            if needs_line:
                line_value = _market_line(
                    label_texts[row_index], handicap=market == "handicap"
                )
            padded: list[float | None] = [*odds, *([None] * (3 - len(odds)))]
            recognized = sum(value is not None for value in odds)
            drafts.append(
                DraftOdds(
                    market=market,
                    label_a=labels[0],
                    odds_a=padded[0],
                    label_b=labels[1],
                    odds_b=padded[1],
                    confidence=(
                        "ok"
                        if (
                            recognized == expected
                            and _plausible_market_odds(odds)
                            and (not needs_line or line_value is not None)
                        )
                        else "partial"
                    ),
                    event_name=event_name,
                    period=period,
                    market_line=line_value,
                    label_c=labels[2] if len(labels) == 3 else None,
                    odds_c=padded[2] if len(labels) == 3 else None,
                )
            )
    return drafts


def extract_odds(image_bytes: bytes) -> list[DraftOdds]:
    """이미지 -> 지원되는 마켓 배당 초안. 항상 사람 확인 후 제출해야 한다."""
    raw_text = extract_text(image_bytes)
    layout_drafts = _layout_market_drafts(image_bytes, raw_text)
    return layout_drafts or parse_draft_odds(raw_text)
