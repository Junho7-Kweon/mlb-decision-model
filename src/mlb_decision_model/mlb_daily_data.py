"""Cached MLB pregame inputs. Odds never enter this adapter.

Project permission is not a grant of rights from MLB. No authentication bypass,
automatic retry on denial, or redistribution is implemented here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from datetime import date as Date, datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .daily_data import KOREAN, DailyDataUnavailable, _count
from .pregame import CONTRACT, build_features, timestamp
from .retrosheet_etl import PitcherForm, BattingForm, DefenseForm
from .sources import require_approved_source

BASE = "https://statsapi.mlb.com/api/v1/"
KST = timezone(timedelta(hours=9))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    temp.replace(path)


class Client:
    def __init__(self, cache):
        require_approved_source("mlb_statsapi", automated=True)
        self.cache, self.last, self.calls = Path(cache), 0., 0

    def get(self, endpoint, params=None, ttl=21600):
        if not (endpoint == "schedule" or endpoint == "teams" or endpoint == "people" or
                endpoint.startswith("teams/") and endpoint.endswith(("/stats", "/roster")) or
                endpoint.startswith("game/") and endpoint.endswith("/boxscore")):
            raise ValueError("Unsupported MLB endpoint")
        url = BASE + endpoint + "?" + urlencode(params or {})
        path = self.cache / (hashlib.sha256(url.encode()).hexdigest() + ".json")
        if path.exists():
            cached = json.loads(path.read_text(encoding="utf-8"))
            if 0 <= time.time() - cached["at"] < ttl:
                return cached["data"]
        if self.calls >= 500:
            raise DailyDataUnavailable("갱신 호출 상한 도달: 캐시는 보존됩니다. 다시 갱신하세요.")
        time.sleep(max(0., .5 - (time.monotonic() - self.last)))
        self.calls += 1
        self.last = time.monotonic()
        with urlopen(Request(url, headers={"User-Agent": "MLBDecisionModel/0.1 (personal research)"}), timeout=45) as response:
            data = json.load(response)
        save(path, {"at": time.time(), "source_url": url, "data": data})
        return data


def splits(payload, group):
    result = []
    for block in payload.get("stats", []):
        if block.get("group", {}).get("displayName") == group:
            rows = block.get("splits", [])
            if block.get("totalSplits", len(rows)) > len(rows):
                raise DailyDataUnavailable("MLB 경기 로그가 잘렸습니다")
            result.extend(rows)
    return result


def completed_game(game, now):
    # MLB also labels postponed games abstractGameState=Final. They have no result.
    return (game["status"].get("codedGameState") in {"F", "O"}
            and game["status"].get("abstractGameState") == "Final"
            and timestamp(game["gameDate"]).date() < now.date()
            and all("score" in game["teams"][s] for s in ("home", "away")))


def team_form(team, logs, arms, final_ids, repair=None):
    """Replay the same decay and 2021 start as training; verify pitching totals."""
    bat, pen, defense = BattingForm(), PitcherForm(), DefenseForm()
    groups = {g: {} for g in ("hitting", "pitching", "fielding")}
    for payload in logs:
        for group in groups:
            for row in splits(payload, group):
                gid = row["game"]["gamePk"]
                if gid in final_ids:
                    if gid in groups[group]:
                        raise DailyDataUnavailable("중복 팀 경기 로그")
                    groups[group][gid] = row
    pitching = defaultdict(dict)
    for player in arms:
        for row in splits(player, "pitching"):
            gid = row["game"]["gamePk"]
            if gid in final_ids and row["team"]["id"] == team:
                pid = player["id"]
                if pid in pitching[gid] and pitching[gid][pid] != row["stat"]:
                    raise DailyDataUnavailable("상충하는 투수 경기 로그")
                pitching[gid][pid] = row["stat"]
    expected = {gid for gid, g in final_ids.items() if any(g["teams"][s]["team"]["id"] == team for s in ("home", "away"))}
    if not expected or any(set(g) != expected for g in groups.values()):
        raise DailyDataUnavailable("완료 경기와 팀 투타·수비 로그가 일치하지 않습니다")
    wins = losses = 0
    h2h = defaultdict(int)
    ordered = sorted(expected, key=lambda gid: (final_ids[gid]["gameDate"], gid))
    for gid in ordered:
        row = groups["hitting"][gid]
        b, p, f = (groups[g][gid]["stat"] for g in ("hitting", "pitching", "fielding"))
        ps = list(pitching[gid].values())
        # Team ER is not necessarily the sum of pitcher ER (reliever scoring).
        def complete(records):
            return (bool(records) and sum(_count(s, "gamesStarted") for s in records) == 1
                    and all(sum(_count(s, k) for s in records) == _count(p, k) for k in ("outs", "battersFaced")))
        if not complete(ps) and repair:
            # Full-season rosters can omit departed players. Use the actual
            # boxscore, not a league-average substitute, to repair that game.
            box = repair(gid)
            matches = [s for s in box["teams"].values() if s["team"]["id"] == team]
            if len(matches) == 1:
                side = matches[0]
                ps = [side["players"][f"ID{pid}"]["stats"]["pitching"] for pid in side["pitchers"]]
        if not complete(ps):
            raise DailyDataUnavailable(f"불완전한 투수 기록: {gid}")
        relief = [s for s in ps if _count(s, "gamesStarted") == 0]
        if relief:
            pen.add(sum(_count(s, "outs") for s in relief), sum(_count(s, "earnedRuns") for s in relief), 0, 0, 0)
        bat.add(*(_count(b, k) for k in ("atBats", "hits", "doubles", "triples", "homeRuns")))
        defense.add(_count(f, "errors"))
        if not isinstance(row.get("isWin"), bool):
            raise DailyDataUnavailable("승패 결과 누락")
        wins += int(row["isWin"])
        losses += int(not row["isWin"])
        if row["isWin"]:
            h2h[row["opponent"]["id"]] += 1
    if not pen.has_sample or bat.ab <= 0:
        raise DailyDataUnavailable("팀 기록 표본 부족")
    return {"team_id": team, "bullpen_era": pen.era(), "team_slg": bat.slg(),
            "errors_per_game": defense.error_rate(), "wins": wins, "losses": losses}, h2h, timestamp(final_ids[ordered[-1]]["gameDate"]).date()


def pitcher_form(player, payloads, final_ids):
    rows = {}
    for payload in payloads:
        for person in payload.get("people", []):
            if person["id"] == player:
                for row in splits(person, "pitching"):
                    gid = row["game"]["gamePk"]
                    if gid in final_ids and _count(row["stat"], "gamesStarted") == 1:
                        rows[gid] = row["stat"]
    form = PitcherForm()
    for gid in sorted(rows, key=lambda g: (final_ids[g]["gameDate"], g)):
        form.add(_count(rows[gid], "outs"), _count(rows[gid], "earnedRuns"), 0, 0, 0)
    if not form.has_sample:
        raise DailyDataUnavailable("선발투수 과거 선발 등판 표본 없음")
    return form.era()


def refresh(output, date=None, client=None, game_id=None, progress=None):
    output = Path(output)
    client = client or Client(output.parent / "mlb_cache")
    now = datetime.now(timezone.utc)
    day = Date.fromisoformat(date) if date else now.astimezone(KST).date()
    if day < now.astimezone(KST).date():
        raise DailyDataUnavailable("현재 API 자료를 과거 시점 예측에 사용할 수 없습니다")
    if day > now.astimezone(KST).date() + timedelta(days=7):
        raise DailyDataUnavailable("당일 연결은 향후 7일 이내 경기만 지원합니다")
    schedule = client.get("schedule", {"sportId": 1, "startDate": (day - timedelta(days=1)).isoformat(), "endDate": day.isoformat(), "hydrate": "probablePitcher"}, ttl=0)
    targets = [g for d in schedule.get("dates", []) for g in d["games"] if g.get("gameType") == "R" and g["status"]["detailedState"] == "Scheduled" and timestamp(g["gameDate"]) > now and timestamp(g["gameDate"]).astimezone(KST).date() == day and (game_id is None or g["gamePk"] == game_id)]
    if not targets:
        raise DailyDataUnavailable("선택한 한국 날짜에 시작 전 정규시즌 경기가 없습니다")
    teams = {t["id"]: t for t in client.get("teams", {"sportId": 1})["teams"]}
    final_ids = {}
    seasons = range(2021, now.year + 1)
    for year in seasons:
        data = client.get("schedule", {"sportId": 1, "season": year, "gameType": "R"})
        final_ids.update({g["gamePk"]: g for d in data.get("dates", []) for g in d["games"] if completed_game(g, now)})
    pitchers = sorted({s["probablePitcher"]["id"] for g in targets for s in g["teams"].values() if s.get("probablePitcher")})
    pitcher_data = [client.get("people", {"personIds": ",".join(map(str, pitchers)), "hydrate": f"stats(type=gameLog,group=pitching,season={year})"}) for year in seasons] if pitchers else []
    forms, failures = {}, {}
    for team in sorted({s["team"]["id"] for g in targets for s in g["teams"].values()}):
        print(f"MLB team {team}: replay 2021-{now.year}", flush=True)
        if progress:
            progress(f"{teams[team]['name']} 과거 기록 확인 중 (캐시 재사용)")
        logs, arms = [], []
        for year in seasons:
            logs.append(client.get(f"teams/{team}/stats", {"stats": "gameLog", "group": "hitting,pitching,fielding", "season": year}))
            roster = client.get(f"teams/{team}/roster", {"rosterType": "fullSeason", "season": year, "hydrate": f"person(stats(type=gameLog,group=pitching,season={year}))"})
            arms.extend(r["person"] for r in roster.get("roster", []))
        try:
            forms[team] = team_form(team, logs, arms, final_ids, repair=lambda gid: client.get(f"game/{gid}/boxscore"))
        except DailyDataUnavailable as exc:
            failures[team] = str(exc)
    games, pending = [], []
    completed_at = datetime.now(timezone.utc)
    for target in targets:
        gid = f"mlb:{target['gamePk']}"
        try:
            if timestamp(target["gameDate"]) <= completed_at or completed_at - now > timedelta(hours=6):
                raise DailyDataUnavailable("수집 중 경기 시작 또는 선발 자료 유효시간 초과")
            result = dict(event_id=gid, starts_at=target["gameDate"], status="scheduled", feature_contract=CONTRACT,
                          source_id="mlb_statsapi", source_url=BASE, source_version="v1-gameLog-decay-2021",
                          as_of=completed_at.isoformat(), retrieved_at=completed_at.isoformat(), schedule_retrieved_at=now.isoformat())
            aliases = {}
            for side in ("home", "away"):
                team = target["teams"][side]["team"]["id"]
                if team in failures:
                    raise DailyDataUnavailable(failures[team])
                starter = target["teams"][side].get("probablePitcher", {}).get("id")
                if not starter:
                    raise DailyDataUnavailable("예정 선발 발표 대기")
                form, _, last = forms[team]
                result[side] = {**form, "starter_id": starter, "starter_era": pitcher_form(starter, pitcher_data, final_ids), "days_since_last_game": (timestamp(target["gameDate"]).date() - last).days}
                t = teams[team]
                code = {"AZ": "ARI", "OAK": "ATH"}.get(t["abbreviation"], t["abbreviation"])
                aliases[side] = {t["name"], t["abbreviation"], KOREAN.get(code, t["name"])}
                if code == "LAD":
                    aliases[side].add("다저스")
            h, a = result["home"]["team_id"], result["away"]["team_id"]
            result.update(h2h_home_wins=forms[h][1][a], h2h_away_wins=forms[a][1][h],
                          event_aliases=[f"{hn}{sep}{an}" for hn in aliases["home"] for an in aliases["away"] for sep in (" vs ", " ")],
                          coverage={"history_start": "2021-01-01", "history_game_types": ["R"], "history_excludes_same_utc_day": True, "lineup_feature": "team_slg_proxy", "untrained": ["individual_lineup", "injuries", "weather"]})
            build_features(result)
            games.append(result)
        except DailyDataUnavailable as exc:
            pending.append({"event_id": gid, "reason": str(exc)})
    save(output, {"games": games, "pending": pending})
    return {"ready_games": len(games), "pending": pending}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", help="Korean calendar date")
    parser.add_argument("--game-id", type=int)
    parser.add_argument("--out", type=Path, default=Path("data/private/pregame_observations.json"))
    args = parser.parse_args()
    print(json.dumps(refresh(args.out, args.date, game_id=args.game_id), ensure_ascii=False))
