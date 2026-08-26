from __future__ import annotations

"""
Retrosheet CSV -> 학습용 CSV 변환 (시점 안전, point-in-time-safe)
================================================================

`retrosheet.py`의 extract_bundle()로 이미 검증·추출된 CSV 폴더를 입력으로 받는다.
이 파일 자체는 아무것도 다운로드하지 않는다 - retrosheet.py가 이미 official host
검사와 필수 파일 확인을 끝낸 로컬 CSV만 읽는다.

컬럼 정의 출처(2026-08-26 확인):
https://www.retrosheet.org/downloads/csvcontents.html

*** 데이터 누수 방지 (as_of < first_pitch) ***
경기를 (date, gid) 순으로 정렬한 뒤, 각 경기의 피처를 계산할 때는 그 경기
"이전까지" 누적된 선발/불펜/휴식 상태만 사용한다. 이 경기 자체의 결과는
피처 계산이 끝난 "다음"에만 누적 상태에 반영한다 - 코드 순서 자체가 이
원칙을 강제하도록 짰다(먼저 읽고, 나중에 갱신).

*** v1 범위: 9개 피처 중 실제로 구현한 것 ***
- starter_edge, bullpen_edge, closer_edge(불펜으로 대체), rest_edge: 구현됨
- lineup_edge, bvp_edge, bench_edge, availability_edge: **아직 0.0(중립)**.
  타자별 좌우 스플릿·상대전적은 plays.csv(타석 단위 원자료, 1경기당 수십 행)가
  필요한데 이번엔 손대지 않았다. model.py는 분산이 0인 피처를 안전하게 무시하므로
  (표준화 시 1e-6 하한 처리) 학습이 깨지진 않지만, 이 4개 피처는 지금 학습에
  아무 기여도 못 한다. 다음 단계로 남겨둔다.
- defense_edge: teamstats.csv에 팀 실책(d_e)이 있지만 이번 v1엔 미포함, 0.0.

*** 실행 방법 ***
    python -m mlb_decision_model.retrosheet_etl \
        data/raw/retrosheet --start-date 20230101 --end-date 20251231 \
        --out data/processed/training_games.csv

*** 검증 ***
실제 725MB 번들을 이 환경에서 받을 수 없어서(용량·네트워크 제약), 이 파일은
Retrosheet의 실제 컬럼 스키마와 정확히 일치하는 합성(synthetic) 픽스처로만
검증했다 - tests/test_retrosheet_etl.py 참고. 진짜 번들로는 사용자가 직접
실행해서 검증해야 한다.
"""

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date as _date
from pathlib import Path


LEAGUE_AVERAGE_ERA = 4.30


def _to_float(value: str | None, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: str | None, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _parse_date(yyyymmdd: str) -> _date:
    return _date(int(yyyymmdd[0:4]), int(yyyymmdd[4:6]), int(yyyymmdd[6:8]))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


@dataclass
class PitcherForm:
    """이 투수(또는 팀 불펜 전체)의 '지금까지' 누적 투구 성적."""

    ipouts: int = 0
    er: int = 0
    bb: int = 0
    k: int = 0
    bfp: int = 0

    @property
    def has_sample(self) -> bool:
        return self.ipouts > 0

    def era(self) -> float:
        if self.ipouts == 0:
            return LEAGUE_AVERAGE_ERA
        return self.er * 27.0 / self.ipouts

    def k_rate(self) -> float:
        return self.k / self.bfp if self.bfp else 0.20

    def bb_rate(self) -> float:
        return self.bb / self.bfp if self.bfp else 0.08

    def add(self, ipouts: int, er: int, bb: int, k: int, bfp: int) -> None:
        self.ipouts += ipouts
        self.er += er
        self.bb += bb
        self.k += k
        self.bfp += bfp


def build_training_rows(
    gameinfo_rows: list[dict[str, str]],
    pitching_rows: list[dict[str, str]],
    start_date: str,
    end_date: str,
) -> list[dict[str, object]]:
    """시점 안전한 학습 행을 만든다. train.py의 9개 피처 + home_win 스키마를 따른다."""

    games = [
        g for g in gameinfo_rows
        if g.get("date") and start_date <= g["date"] <= end_date
        and g.get("wteam") and g.get("lteam")  # 결과 없는(취소 등) 경기 제외
    ]
    games.sort(key=lambda g: (g["date"], g["gid"]))

    pitching_by_game: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in pitching_rows:
        pitching_by_game[row["gid"]].append(row)

    starter_form: dict[str, PitcherForm] = defaultdict(PitcherForm)
    bullpen_form: dict[str, PitcherForm] = defaultdict(PitcherForm)  # 팀ID 기준
    last_game_date: dict[str, _date] = {}

    output: list[dict[str, object]] = []

    for game in games:
        gid = game["gid"]
        home, away = game["hometeam"], game["visteam"]
        game_date = _parse_date(game["date"])
        pitchers = pitching_by_game.get(gid, [])

        home_starter = next((p for p in pitchers if p.get("team") == home and p.get("p_seq") == "1"), None)
        away_starter = next((p for p in pitchers if p.get("team") == away and p.get("p_seq") == "1"), None)
        if home_starter is None or away_starter is None:
            continue  # 선발을 식별 못 하면 안전하게 이 경기는 건너뛴다

        # --- 1) 먼저 '이 경기 이전까지'의 상태만 읽어서 피처를 만든다 ---
        home_starter_era = starter_form[home_starter["id"]].era()
        away_starter_era = starter_form[away_starter["id"]].era()
        home_bullpen_era = bullpen_form[home].era()
        away_bullpen_era = bullpen_form[away].era()

        home_rest = (game_date - last_game_date[home]).days if home in last_game_date else 4
        away_rest = (game_date - last_game_date[away]).days if away in last_game_date else 4
        home_rest = max(-2, min(6, home_rest - 4))
        away_rest = max(-2, min(6, away_rest - 4))

        # ERA는 낮을수록 좋으므로 부호를 뒤집어서 "높을수록 유리한" edge로 통일
        output.append({
            "starter_edge": round((away_starter_era - home_starter_era) / 4.0, 4),
            "bullpen_edge": round((away_bullpen_era - home_bullpen_era) / 4.0, 4),
            "closer_edge": round((away_bullpen_era - home_bullpen_era) / 4.0, 4),  # 불펜으로 대체(v1)
            "lineup_edge": 0.0,       # TODO(v2): plays.csv 기반 좌우 wOBA
            "bvp_edge": 0.0,           # TODO(v2): plays.csv 기반 상대전적
            "bench_edge": 0.0,         # TODO(v2)
            "availability_edge": 0.0,  # TODO(v2)
            "defense_edge": 0.0,       # TODO(v2): teamstats.csv의 d_e 롤링
            "rest_edge": round((home_rest - away_rest) / 4.0, 4),
            "home_win": 1 if game.get("wteam") == home else 0,
            "_gid": gid,
            "_date": game["date"],
        })

        # --- 2) 그다음에야 이 경기 결과를 상태에 반영 (다음 경기부터 영향) ---
        for p in pitchers:
            ipouts, er, bb, k, bfp = (
                _to_int(p.get("p_ipouts")), _to_int(p.get("p_er")),
                _to_int(p.get("p_w")), _to_int(p.get("p_k")), _to_int(p.get("p_bfp")),
            )
            if p.get("p_seq") == "1":
                starter_form[p["id"]].add(ipouts, er, bb, k, bfp)
            else:
                bullpen_form[p["team"]].add(ipouts, er, bb, k, bfp)
        last_game_date[home] = game_date
        last_game_date[away] = game_date

    return output


FEATURE_COLUMNS = (
    "starter_edge", "bullpen_edge", "closer_edge", "lineup_edge", "bvp_edge",
    "bench_edge", "availability_edge", "defense_edge", "rest_edge", "home_win",
)


def write_training_csv(rows: list[dict[str, object]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FEATURE_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in FEATURE_COLUMNS})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert extracted Retrosheet CSVs into a train.py-compatible training CSV"
    )
    parser.add_argument("csv_dir", type=Path, help="retrosheet.py extract_bundle()가 만든 폴더")
    parser.add_argument("--start-date", required=True, help="YYYYMMDD")
    parser.add_argument("--end-date", required=True, help="YYYYMMDD")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    gameinfo = read_csv(args.csv_dir / "gameinfo.csv")
    pitching = read_csv(args.csv_dir / "pitching.csv")
    rows = build_training_rows(gameinfo, pitching, args.start_date, args.end_date)
    write_training_csv(rows, args.out)
    print(f"games={len(rows)} -> {args.out}")
    print("주의: lineup_edge/bvp_edge/bench_edge/availability_edge/defense_edge는 "
          "아직 0.0(중립)입니다 - v2에서 plays.csv 기반으로 채워야 합니다.")


if __name__ == "__main__":
    main()
