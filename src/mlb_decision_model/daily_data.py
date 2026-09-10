"""On-demand licensed daily-data import; no odds endpoints are used."""
from __future__ import annotations

import argparse
import json
import os
import math
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.parse import urlencode
from urllib.error import HTTPError

from .sources import require_approved_source
from .pregame import CONTRACT, timestamp, build_features
from .retrosheet_etl import PitcherForm, BattingForm, DefenseForm

BASE = "https://api.balldontlie.io/mlb/v1/"
# Reviewed display aliases, provider team IDs remain authoritative.
KOREAN = {
    "ARI": "애리조나 다이아몬드백스", "ATL": "애틀랜타 브레이브스", "BAL": "볼티모어 오리올스",
    "BOS": "보스턴 레드삭스", "CHC": "시카고 컵스", "CWS": "시카고 화이트삭스",
    "CIN": "신시내티 레즈", "CLE": "클리블랜드 가디언스", "COL": "콜로라도 로키스",
    "DET": "디트로이트 타이거스", "HOU": "휴스턴 애스트로스", "KC": "캔자스시티 로열스",
    "LAA": "LA 에인절스", "LAD": "LA 다저스", "MIA": "마이애미 말린스",
    "MIL": "밀워키 브루어스", "MIN": "미네소타 트윈스", "NYM": "뉴욕 메츠",
    "NYY": "뉴욕 양키스", "ATH": "애슬레틱스", "PHI": "필라델피아 필리스",
    "PIT": "피츠버그 파이리츠", "SD": "샌디에이고 파드리스", "SF": "샌프란시스코 자이언츠",
    "SEA": "시애틀 매리너스", "STL": "세인트루이스 카디널스", "TB": "탬파베이 레이스",
    "TEX": "텍사스 레인저스", "TOR": "토론토 블루제이스", "WSH": "워싱턴 내셔널스",
}


class DailyDataUnavailable(ValueError):
    pass


class Client:
    def __init__(self, key=None):
        require_approved_source("balldontlie", automated=True)
        self.key = key or os.environ.get("BALLDONTLIE_API_KEY", "")
        if not self.key:
            raise DailyDataUnavailable("BALLDONTLIE_API_KEY가 없습니다. 선발·라인업 API 이용 권한이 있는 키가 필요합니다.")
        self.last_call = 0.0
        self.rpm = float(os.environ.get("BALLDONTLIE_REQUESTS_PER_MINUTE", "5"))
        if not math.isfinite(self.rpm) or not 1 <= self.rpm <= 600:
            raise DailyDataUnavailable("호출 한도 설정은 1~600 범위여야 합니다")

    def rows(self, endpoint, params):
        if endpoint not in {"games", "stats", "lineups"}:
            raise ValueError("Unsupported daily-data endpoint")
        cursor, seen, result = None, set(), []
        for _ in range(10000):
            # Conservative default also supports the documented trial limit.
            time.sleep(max(0, 60 / self.rpm + .1 - (time.monotonic() - self.last_call)))
            query = {**params, "per_page": 100}
            if cursor is not None:
                query["cursor"] = cursor
            request = Request(BASE + endpoint + "?" + urlencode(query, doseq=True), headers={"Authorization": self.key, "User-Agent": "MLBDecisionModel/0.1"})
            self.last_call = time.monotonic()
            try:
                with urlopen(request, timeout=30) as response:
                    payload = json.load(response)
            except HTTPError as exc:
                raise DailyDataUnavailable(f"데이터 API 응답 {exc.code}: 이용 권한·호출 한도를 확인하세요.") from None
            result.extend(payload["data"])
            cursor = payload.get("meta", {}).get("next_cursor")
            if cursor is None:
                return result
            if cursor in seen:
                raise DailyDataUnavailable("API 페이지가 반복돼 수집을 중단했습니다")
            seen.add(cursor)
        raise DailyDataUnavailable("API 페이지 제한에 도달했습니다. 불완전한 자료는 예측에 사용하지 않습니다.")


def _count(row, key):
    value = row.get(key)
    if value is None or isinstance(value, bool):
        raise DailyDataUnavailable(f"상세 경기 기록 누락: {key}")
    value = float(value)
    if not value.is_integer() or value < 0:
        raise DailyDataUnavailable(f"잘못된 경기 기록: {key}")
    return int(value)


def observations(target, history, stats, lineups, now):
    """Replay final box scores through the same decay classes used in training.

    History coverage must start at 2021, like the deployed training CSV. Only
    games on earlier UTC dates are included; same-day outcomes are deliberately
    excluded until reliable finalization timestamps are available.
    """
    starters, bullpen, batting, defense = (defaultdict(cls) for cls in (PitcherForm, PitcherForm, BattingForm, DefenseForm))
    wins, losses, h2h, last = defaultdict(int), defaultdict(int), defaultdict(int), {}
    by_game = defaultdict(list)
    seen = set()
    for row in stats:
        key = (row["game_id"], row["team"]["id"], row["player"]["id"])
        if key in seen:
            raise DailyDataUnavailable("중복된 선수 경기 기록")
        seen.add(key)
        by_game[row["game_id"]].append(row)
    for game in sorted(history, key=lambda g: (timestamp(g["date"]), g["id"])):
        if game.get("status_state") != "final" or timestamp(game["date"]).date() >= now.date():
            continue
        home, away = game["home_team"]["id"], game["away_team"]["id"]
        for side, team in (("home", home), ("away", away)):
            records = [r for r in by_game[game["id"]] if r["team"]["id"] == team]
            if not records:
                raise DailyDataUnavailable(f"완료 경기 상세 기록 누락: {game['id']}")
            arms = [r for r in records if r.get("ip") is not None or r.get("pitching_outs") is not None]
            if not arms:
                raise DailyDataUnavailable("투수 기록 누락")
            relief_outs = relief_er = 0
            for arm in arms:
                outs, er = _count(arm, "pitching_outs"), _count(arm, "er")
                if _count(arm, "games_started") == 1:
                    starters[arm["player"]["id"]].add(outs, er, 0, 0, 0)
                else:
                    relief_outs += outs
                    relief_er += er
            if any(_count(a, "games_started") == 0 for a in arms):
                bullpen[team].add(relief_outs, relief_er, 0, 0, 0)
            bats = [r for r in records if r.get("at_bats") is not None]
            if not bats:
                raise DailyDataUnavailable("타격 기록 누락")
            batting[team].add(*(sum(_count(r, field) for r in bats) for field in ("at_bats", "hits", "doubles", "triples", "hr")))
            defense[team].add(_count(game[side + "_team_data"], "errors"))
            last[team] = timestamp(game["date"]).date()
        hr, ar = _count(game["home_team_data"], "runs"), _count(game["away_team_data"], "runs")
        if hr != ar:
            winner, loser = (home, away) if hr > ar else (away, home)
            wins[winner] += 1
            losses[loser] += 1
            h2h[winner, loser] += 1
    result = {"event_id": f"bdl:{target['id']}", "starts_at": target["date"], "status": target["status_state"], "feature_contract": CONTRACT,
              "source_id": "balldontlie", "source_url": BASE, "source_version": "mlb/v1", "as_of": now.isoformat(), "retrieved_at": now.isoformat()}
    for side in ("home", "away"):
        team = target[side + "_team"]["id"]
        candidates = [r for r in lineups if r["game_id"] == target["id"] and r["team"]["id"] == team and r.get("is_probable_pitcher") is True]
        if len(candidates) != 1:
            raise DailyDataUnavailable(f"{side} 선발투수 발표 대기")
        player = candidates[0]["player"]["id"]
        if player not in starters or not starters[player].has_sample or team not in last:
            raise DailyDataUnavailable("선발투수 또는 팀 과거 표본 부족")
        if not bullpen[team].has_sample or batting[team].ab <= 0 or defense[team].games <= 0:
            raise DailyDataUnavailable("팀 타격·불펜·수비 표본 부족")
        result[side] = {"team_id": team, "starter_id": player, "starter_era": starters[player].era(), "bullpen_era": bullpen[team].era(),
                        "team_slg": batting[team].slg(), "errors_per_game": defense[team].error_rate(),
                        "wins": wins[team], "losses": losses[team], "days_since_last_game": (timestamp(target["date"]).date() - last[team]).days}
    h, a = target["home_team"], target["away_team"]
    result["h2h_home_wins"], result["h2h_away_wins"] = h2h[h["id"], a["id"]], h2h[a["id"], h["id"]]
    def names(team):
        code = {"OAK": "ATH", "AZ": "ARI", "CHW": "CWS", "KCR": "KC", "SDP": "SD", "SFG": "SF", "TBR": "TB", "WSN": "WSH"}.get(team["abbreviation"], team["abbreviation"])
        values = {team["display_name"], team["abbreviation"]}
        if code in KOREAN:
            values.add(KOREAN[code])
        if code == "LAD":
            values.add("다저스")
        return values
    result["event_aliases"] = [f"{hn}{sep}{an}" for hn in names(h) for an in names(a) for sep in (" vs ", " ")]
    result["coverage"] = {"history_start": "2021-01-01", "history_excludes_same_utc_day": True, "lineup_feature": "team_slg_proxy", "untrained": ["injuries", "individual_lineup", "weather"]}
    build_features(result)
    return result


def refresh(output: Path, date=None, client=None):
    client = client or Client()
    now = datetime.now(timezone.utc)
    days = [date] if date else [(now + timedelta(days=i)).date().isoformat() for i in (0, 1)]
    targets = [g for g in client.rows("games", {"dates[]": days}) if g.get("status_state") == "scheduled" and timestamp(g["date"]) > now]
    if not targets:
        raise DailyDataUnavailable("선택 날짜에 시작 전 경기가 없습니다")
    lineups = client.rows("lineups", {"game_ids[]": [g["id"] for g in targets]})
    # Historical data are reusable; keep endpoint provenance with the cached copy.
    cache = output.parent / "daily_history_cache.json"
    history_data = json.loads(cache.read_text(encoding="utf-8")) if cache.is_file() else None
    if not history_data or history_data.get("through") != now.date().isoformat():
        history, stats = [], []
        for season in range(2021, now.year + 1):
            season_path = output.parent / f"daily_history_{season}.json"
            archived = json.loads(season_path.read_text(encoding="utf-8")) if season_path.is_file() else None
            if archived is None or (season == now.year and archived.get("through") != now.date().isoformat()):
                print(f"Syncing historical season {season}", flush=True)
                archived = {"through": now.date().isoformat(), "source_url": BASE,
                            "history": client.rows("games", {"seasons[]": [season], "season_type": "regular"}),
                            "stats": client.rows("stats", {"seasons[]": [season]})}
                output.parent.mkdir(parents=True, exist_ok=True)
                season_temp = season_path.with_suffix(".tmp")
                season_temp.write_text(json.dumps(archived), encoding="utf-8")
                season_temp.replace(season_path)
            history.extend(archived["history"])
            stats.extend(archived["stats"])
        history_data = {"source_url": BASE, "through": now.date().isoformat(), "history": history, "stats": stats}
        output.parent.mkdir(parents=True, exist_ok=True)
        temp = cache.with_suffix(".tmp")
        temp.write_text(json.dumps(history_data), encoding="utf-8")
        temp.replace(cache)
    games, pending = [], []
    for target in targets:
        try:
            # Keep the actual lineup retrieval cutoff; a long history bootstrap
            # must never make an old lineup look freshly collected.
            games.append(observations(target, history_data["history"], history_data["stats"], lineups, now))
        except DailyDataUnavailable as exc:
            pending.append({"event_id": f"bdl:{target['id']}", "reason": str(exc)})
    payload = {"games": games, "pending": pending}
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    temp.replace(output)
    return {"ready_games": len(games), "pending": pending}


def main():
    parser = argparse.ArgumentParser(description="공식 데이터 API에서 당일 경기 자료 갱신")
    parser.add_argument("--date")
    parser.add_argument("--out", type=Path, default=Path("data/private/pregame_observations.json"))
    args = parser.parse_args()
    print(json.dumps(refresh(args.out, args.date), ensure_ascii=False))


if __name__ == "__main__":
    main()
