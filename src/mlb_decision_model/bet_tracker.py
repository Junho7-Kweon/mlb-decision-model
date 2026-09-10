from __future__ import annotations

"""
개인 베팅 기록 추적기 (모델 학습과 완전히 별개)
================================================================

*** 이 모듈은 model.py/retrosheet_etl.py가 학습하는 데이터와 아무 관련이
없다. *** 학습 모델은 Retrosheet 과거 데이터로만 학습하고, 경기 1개
결과로 즉시 재학습하지 않는다는 원칙(ERD/블루프린트에 명시)을 지킨다.

이 모듈은 순수하게 "내가 실제로 어떤 근거로, 어떤 픽을, 얼마의 확률로
평가해서 걸었고, 결과가 어땠는지"를 개인 기록으로 남기는 용도다. 시간이
지나면서 이 CSV 자체가 쌓이면, "불펜을 확인했을 때와 안 했을 때 적중률
차이" 같은 개인 패턴 분석에 쓸 수 있다 - 이것도 모델 재학습이 아니라
사람이 직접 보는 리포트다.

데이터는 data/private/에만 저장되고 Git에 커밋되지 않는다(.gitignore로
이미 막혀 있음).
"""

import csv
from dataclasses import dataclass, asdict
from datetime import date as _date
from pathlib import Path


DEFAULT_LOG_PATH = Path("data/private/bet_tracker.csv")

FIELDS = (
    "date", "ticket_id", "leg_number", "game", "market", "pick",
    "odds", "my_probability", "break_even", "edge", "ev",
    "factors_considered", "actual_outcome", "final_score", "notes",
)


@dataclass
class BetLeg:
    date: str                    # YYYY-MM-DD
    ticket_id: str                # 같은 티켓의 다리들을 묶는 ID (예: "2026-09-02-T1")
    leg_number: int
    game: str                      # 예: "NYY@LAA"
    market: str                     # 예: "moneyline", "total"
    pick: str                        # 예: "NYY 승", "언더7.5"
    odds: float
    my_probability: float
    factors_considered: str            # 콤마로 구분: "선발폼,불펜,부상자,핫스트릭" 등
    actual_outcome: str = ""             # "WIN" | "LOSS" | "PUSH" | "" (미정)
    final_score: str = ""
    notes: str = ""

    @property
    def break_even(self) -> float | None:
        if self.odds <= 0:
            return None
        return round(1.0 / self.odds, 4)

    @property
    def edge(self) -> float | None:
        break_even = self.break_even
        if break_even is None:
            return None
        return round(self.my_probability - break_even, 4)

    @property
    def ev(self) -> float | None:
        if self.odds <= 0:
            return None
        return round(self.my_probability * self.odds - 1.0, 4)

    def to_row(self) -> dict[str, object]:
        row = asdict(self)
        row["break_even"] = self.break_even
        row["edge"] = self.edge
        row["ev"] = self.ev
        return row


def append_legs(legs: list[BetLeg], path: Path = DEFAULT_LOG_PATH) -> None:
    """새 다리들을 CSV 끝에 추가한다. 파일이 없으면 헤더부터 만든다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()
        for leg in legs:
            writer.writerow({field: leg.to_row()[field] for field in FIELDS})


def load_legs(path: Path = DEFAULT_LOG_PATH) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def record_outcome(
    ticket_id: str, leg_number: int, outcome: str, final_score: str = "",
    path: Path = DEFAULT_LOG_PATH,
) -> bool:
    """이미 기록된 다리 하나의 결과를 채운다. 찾아서 갱신하면 True."""
    rows = load_legs(path)
    found = False
    for row in rows:
        if row["ticket_id"] == ticket_id and row["leg_number"] == str(leg_number):
            row["actual_outcome"] = outcome
            row["final_score"] = final_score
            found = True
    if found:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
    return found


def summarize(path: Path = DEFAULT_LOG_PATH) -> dict[str, object]:
    """개인 적중 패턴을 요약한다. 모델 성능 지표가 아니라 개인 리포트용."""
    rows = [r for r in load_legs(path) if r["actual_outcome"] in ("WIN", "LOSS")]
    if not rows:
        return {"total_legs": 0}

    wins = sum(1 for r in rows if r["actual_outcome"] == "WIN")
    total = len(rows)

    # 고려한 요소별 적중률 - "불펜을 확인했을 때 vs 안 했을 때" 같은 패턴을 보려는 용도
    by_factor: dict[str, dict[str, int]] = {}
    for row in rows:
        factors = [f.strip() for f in row["factors_considered"].split(",") if f.strip()]
        for factor in factors:
            bucket = by_factor.setdefault(factor, {"wins": 0, "total": 0})
            bucket["total"] += 1
            if row["actual_outcome"] == "WIN":
                bucket["wins"] += 1

    return {
        "total_legs": total,
        "win_rate": round(wins / total, 4),
        "avg_my_probability": round(sum(float(r["my_probability"]) for r in rows) / total, 4),
        "avg_edge": round(sum(float(r["edge"]) for r in rows) / total, 4),
        "by_factor": {
            factor: {
                "win_rate": round(b["wins"] / b["total"], 4),
                "sample": b["total"],
            }
            for factor, b in by_factor.items()
        },
    }
