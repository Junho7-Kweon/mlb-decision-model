from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .sources import require_approved_source


DATASET_ID = "15107776"
ENDPOINT = (
    "https://apis.data.go.kr/B551014/SRVC_OD_API_TB_SOSFO_MATCH_MGMT/"
    "todz_api_tb_match_mgmt_i"
)


@dataclass(frozen=True)
class KspoMatchResult:
    match_date: str
    match_time: str
    home_team: str
    away_team: str
    league: str
    stadium: str
    sport: str
    result: str
    product_name: str


def _item_list(payload: dict[str, Any]) -> list[dict[str, Any]]:
    body = payload.get("body", {}) or {}
    items = body.get("items", {}) or {}
    raw = items.get("item", []) if isinstance(items, dict) else []
    if isinstance(raw, dict):
        return [raw]
    return list(raw or [])


def parse_results(payload: dict[str, Any]) -> list[KspoMatchResult]:
    header = payload.get("header", {}) or {}
    code = str(header.get("resultCode", ""))
    if code not in {"0", "00"}:
        raise ValueError(f"data.go.kr API error {code}: {header.get('resultMsg', '')}")
    return [
        KspoMatchResult(
            match_date=str(item.get("match_ymd", "")),
            match_time=str(item.get("match_tm", "")),
            home_team=str(item.get("hteam_han_nm", "")),
            away_team=str(item.get("ateam_han_nm", "")),
            league=str(item.get("leag_han_nm", "")),
            stadium=str(item.get("stdm_han_nm", "")),
            sport=str(item.get("match_sport_han_nm", "")),
            result=str(item.get("match_end_val", "")),
            product_name=str(item.get("obj_prod_nm", "")),
        )
        for item in _item_list(payload)
    ]


def fetch_results(
    *,
    service_key: str | None = None,
    match_date: str | None = None,
    home_team: str | None = None,
    away_team: str | None = None,
    sport: str | None = "야구",
    page: int = 1,
    rows: int = 100,
    timeout: float = 20.0,
) -> list[KspoMatchResult]:
    """Fetch delayed official results; this endpoint does not provide odds."""
    require_approved_source("data_go_kr_kspo_results", automated=True)
    key = service_key or os.environ.get("DATA_GO_KR_SERVICE_KEY", "")
    if not key:
        raise ValueError("DATA_GO_KR_SERVICE_KEY is required")
    if not 1 <= rows <= 1000:
        raise ValueError("rows must be between 1 and 1000")
    params = {
        "serviceKey": key,
        "pageNo": str(page),
        "numOfRows": str(rows),
        "resultType": "json",
    }
    optional = {
        "match_ymd": match_date,
        "hteam_han_nm": home_team,
        "ateam_han_nm": away_team,
        "match_sport_han_nm": sport,
    }
    params.update({name: value for name, value in optional.items() if value})
    request = Request(
        f"{ENDPOINT}?{urlencode(params)}",
        headers={"User-Agent": "mlb-decision-model/0.1"},
    )
    with urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return parse_results(payload)


def as_rows(results: list[KspoMatchResult]) -> list[dict[str, str]]:
    return [asdict(result) for result in results]
