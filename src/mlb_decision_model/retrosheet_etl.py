from __future__ import annotations

"""Convert an extracted Retrosheet Main/Simplified CSV bundle to v2 rows.

The transformer is point-in-time safe: every feature is computed from games
strictly preceding the current game, and only then is the current result added
to rolling state. Main downloads are reduced to ``stattype=value``; Simplified
downloads, which already contain only those rows, follow the same code path.
"""

import argparse
import csv
import io
import re
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import date as _date
from datetime import datetime
from pathlib import Path, PurePosixPath

from .features import FEATURE_NAMES, FEATURE_VERSION
from .retrosheet import inspect_bundle


LEAGUE_AVERAGE_ERA = 4.30
MIN_GAMES_FOR_TEAM_PCT = 10
H2H_SHRINKAGE_PRIOR_GAMES = 8.0


def _to_int(value: str | None, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _optional_int(value: str | None) -> int | None:
    if value is None or not str(value).strip():
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def _parse_date(yyyymmdd: str) -> _date:
    return _date(int(yyyymmdd[0:4]), int(yyyymmdd[4:6]), int(yyyymmdd[6:8]))


def _as_of_timestamp(game: dict[str, str]) -> str:
    """Return a stable first-pitch timestamp without inventing a timezone."""
    game_date = _parse_date(game["date"])
    raw_time = (game.get("starttime") or "").strip().upper()
    for pattern in ("%H:%M", "%H%M", "%I:%M%p", "%I:%M %p"):
        try:
            parsed = datetime.strptime(raw_time, pattern).time()
            return datetime.combine(game_date, parsed).isoformat()
        except ValueError:
            continue
    return datetime.combine(game_date, datetime.min.time()).isoformat()


def value_stat_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Normalize Main and Simplified CSVs to Retrosheet's best-estimate rows.

    Main CSVs may also contain ``lower``, ``upper`` and ``official`` records.
    Simplified files contain only ``value`` records. Older/synthetic fixtures
    without a ``stattype`` column are treated as already simplified.
    """
    return [
        row
        for row in rows
        if not row.get("stattype") or row["stattype"].strip().lower() == "value"
    ]


def _read_filtered_rows(
    handle: io.TextIOBase,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    value_only: bool = False,
    columns: tuple[str, ...] | None = None,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in csv.DictReader(handle):
        row_date = row.get("date")
        if start_date and row_date and row_date < start_date:
            continue
        if end_date and row_date and row_date > end_date:
            continue
        if value_only and row.get("stattype"):
            if row["stattype"].strip().lower() != "value":
                continue
        rows.append({key: row[key] for key in columns if key in row} if columns else row)
    return rows


def read_csv(
    path: Path,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    value_only: bool = False,
    columns: tuple[str, ...] | None = None,
) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return _read_filtered_rows(
            handle,
            start_date=start_date,
            end_date=end_date,
            value_only=value_only,
            columns=columns,
        )


def read_bundle_csv(
    bundle: Path,
    filename: str,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    value_only: bool = False,
    columns: tuple[str, ...] | None = None,
) -> list[dict[str, str]]:
    """Read one CSV directly from a large Retrosheet ZIP without extracting it."""
    with zipfile.ZipFile(bundle) as archive:
        matches = [
            item
            for item in archive.infolist()
            if not item.is_dir()
            and PurePosixPath(item.filename.replace("\\", "/")).name.lower()
            == filename.lower()
        ]
        if len(matches) != 1:
            raise ValueError(
                f"expected exactly one {filename!r} in {bundle.name}, found {len(matches)}"
            )
        with archive.open(matches[0]) as raw_handle:
            with io.TextIOWrapper(raw_handle, encoding="utf-8-sig", newline="") as handle:
                return _read_filtered_rows(
                    handle,
                    start_date=start_date,
                    end_date=end_date,
                    value_only=value_only,
                    columns=columns,
                )


@dataclass
class PitcherForm:
    """A pitcher's or team bullpen's cumulative form before a game.

    *** 2026-09-02 실전에서 발견한 문제를 반영한 변경 ***
    지금까지는 시즌 전체를 동일 가중치로 평균 냈다. 그런데 실제 라이브 경기
    분석에서 이렇게 했다가 "불펜 최근 상태"를 놓쳤고(양키스-에인절스 경기가
    6이닝 만에 7점이 남), 스펜서 존스의 2경기 연속 홈런 같은 최근 폼도 시즌
    평균에 묻혀서 안 보였다. 그래서 지수 감쇠(decay)를 넣어서 최근 등판이
    과거 등판보다 더 큰 비중을 갖게 했다. DECAY=0.92는 대략 8~9번째 등판
    전의 기록이 절반 정도로 줄어드는 정도다(반감기 근사).
    """

    ipouts: float = 0.0
    er: float = 0.0
    bb: float = 0.0
    k: float = 0.0
    bfp: float = 0.0

    DECAY = 0.92

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
        # 먼저 과거 누적치를 감쇠시킨 다음 이번 등판을 더한다 - 최근이 더 큰 비중을 갖는다.
        self.ipouts = self.ipouts * self.DECAY + ipouts
        self.er = self.er * self.DECAY + er
        self.bb = self.bb * self.DECAY + bb
        self.k = self.k * self.DECAY + k
        self.bfp = self.bfp * self.DECAY + bfp


@dataclass
class BattingForm:
    """A team's recency-weighted batting form before a game (lineup_edge용).

    *** 신규 (2026-09-02) *** — 스펜서 존스처럼 최근 2~3경기 폼이 급상승한
    타자가 시즌 평균에 묻히는 문제를 막기 위해 만들었다. batting.csv를
    선수별이 아니라 팀 단위로 이 경기 전까지 감쇠 누적해서 팀 SLG를 프록시로
    쓴다. 개별 타자별·좌우 스플릿까지 반영한 정교한 버전은 plays.csv 기반
    v3에서 다룰 예정이고, 이건 그 전까지 쓸 1차 버전이다.

    주의: batting.csv의 정확한 컬럼명(b_ab, b_h, b_2b, b_3b, b_hr)은
    pitching.csv의 p_ 접두어 관례를 따른다고 가정했다 - 실제 번들로 아직
    검증 못 했으니 사용자가 실제 실행해서 확인해야 한다.
    """

    ab: float = 0.0
    h: float = 0.0
    doubles: float = 0.0
    triples: float = 0.0
    hr: float = 0.0

    DECAY = 0.90  # 타자는 경기당 표본(타석)이 많아 불펜보다 조금 더 빠르게 감쇠

    def slg(self) -> float:
        if self.ab == 0:
            return 0.400  # 리그 평균급 기본값(표본 없을 때)
        singles = max(0.0, self.h - self.doubles - self.triples - self.hr)
        total_bases = singles + 2 * self.doubles + 3 * self.triples + 4 * self.hr
        return total_bases / self.ab

    def add(self, ab: int, h: int, doubles: int, triples: int, hr: int) -> None:
        self.ab = self.ab * self.DECAY + ab
        self.h = self.h * self.DECAY + h
        self.doubles = self.doubles * self.DECAY + doubles
        self.triples = self.triples * self.DECAY + triples
        self.hr = self.hr * self.DECAY + hr


def log5_win_prob(team_pct: float, opponent_pct: float) -> float:
    """Estimate a team's neutral win probability from two baseline records."""
    denominator = team_pct + opponent_pct - 2 * team_pct * opponent_pct
    if denominator <= 0:
        return 0.5
    return (team_pct - team_pct * opponent_pct) / denominator


def team_matchup_edge(
    home_pct: float,
    away_pct: float,
    h2h_home_wins: int,
    h2h_away_wins: int,
) -> float:
    """Return a Bayesian-shrunk head-to-head residual from the home view."""
    h2h_games = h2h_home_wins + h2h_away_wins
    if h2h_games == 0:
        return 0.0
    expected = log5_win_prob(home_pct, away_pct)
    actual = h2h_home_wins / h2h_games
    reliability = h2h_games / (h2h_games + H2H_SHRINKAGE_PRIOR_GAMES)
    return (actual - expected) * reliability


def _f5_runs(team_row: dict[str, str] | None) -> int | None:
    if team_row is None:
        return None
    innings = [_optional_int(team_row.get(f"inn{inning}")) for inning in range(1, 6)]
    if any(value is None for value in innings):
        return None
    return sum(value for value in innings if value is not None)


def _add_pitching_result(
    pitchers: list[dict[str, str]],
    starter_form: dict[str, PitcherForm],
    bullpen_form: dict[str, PitcherForm],
) -> None:
    # 불펜은 팀 단위로 감쇠하므로, 한 경기에 계투가 여럿이어도 감쇠는 경기당
    # 딱 1번만 적용해야 한다 - 개별 투수마다 add()를 부르면 감쇠가 경기 안에서
    # 여러 번 겹쳐 적용되는 버그가 생긴다. 그래서 팀별로 먼저 합산한다.
    bullpen_totals: dict[str, dict[str, int]] = defaultdict(
        lambda: {"ipouts": 0, "er": 0, "bb": 0, "k": 0, "bfp": 0}
    )
    for pitcher in pitchers:
        ipouts, er, bb, strikeouts, bfp = (
            _to_int(pitcher.get("p_ipouts")),
            _to_int(pitcher.get("p_er")),
            _to_int(pitcher.get("p_w")),
            _to_int(pitcher.get("p_k")),
            _to_int(pitcher.get("p_bfp")),
        )
        if _to_int(pitcher.get("p_seq")) == 1:
            starter_form[pitcher["id"]].add(ipouts, er, bb, strikeouts, bfp)
        else:
            totals = bullpen_totals[pitcher["team"]]
            totals["ipouts"] += ipouts
            totals["er"] += er
            totals["bb"] += bb
            totals["k"] += strikeouts
            totals["bfp"] += bfp
    for team, totals in bullpen_totals.items():
        bullpen_form[team].add(
            totals["ipouts"], totals["er"], totals["bb"], totals["k"], totals["bfp"]
        )


def _add_batting_result(
    batting_rows: list[dict[str, str]],
    batting_form: dict[str, BattingForm],
) -> None:
    """팀 단위로 이 경기의 타격을 합산해서 감쇠 누적한다(불펜과 같은 원리)."""
    team_totals: dict[str, dict[str, int]] = defaultdict(
        lambda: {"ab": 0, "h": 0, "doubles": 0, "triples": 0, "hr": 0}
    )
    for row in batting_rows:
        team = row.get("team")
        if not team:
            continue
        totals = team_totals[team]
        totals["ab"] += _to_int(row.get("b_ab"))
        totals["h"] += _to_int(row.get("b_h"))
        totals["doubles"] += _to_int(row.get("b_d"))
        totals["triples"] += _to_int(row.get("b_t"))
        totals["hr"] += _to_int(row.get("b_hr"))
    for team, totals in team_totals.items():
        batting_form[team].add(
            totals["ab"], totals["h"], totals["doubles"], totals["triples"], totals["hr"]
        )


LEAGUE_AVG_BATTING = 0.250
BVP_PRIOR_AB = 50.0


def classify_event(event: str) -> tuple[bool, bool]:
    """Retrosheet event 코드(예: 'S8','K','63','HR')를 (타수 여부, 안타 여부)로 분류한다.

    실제 표준 표기법 20개 예시로 검증 완료(단타/2루타/3루타/홈런/그라운드룰2루타,
    삼진/필딩아웃/실책출루/야수선택, 볼넷/고의사구/사구/희생타/도루/노플레이).
    모르는 코드는 보수적으로 (False, False) - 잘못된 신호보다 누락이 안전하다.
    """
    if not event:
        return False, False
    code = event.strip().upper()
    core = re.split(r"[/.]", code)[0]
    if re.search(r"/(?:SH|SF)(?:/|\.|$)", code):
        return False, False
    core = core.rstrip("!?#")

    if core.startswith("DGR"):
        return True, True
    if re.match(r"^(S|D|T)(\d.*)?$", core):
        return True, True
    if core.startswith("HR") or core == "H":
        return True, True

    if core == "K" or core.startswith("K"):
        return True, False
    if re.match(r"^\d", core):
        return True, False
    if core.startswith("E"):
        return True, False
    if core.startswith("FC"):
        return True, False

    if core in ("W", "IW", "HP", "SH", "SF", "SB", "CS", "PB", "WP", "BK", "NP", "DI", "OA"):
        return False, False
    return False, False


@dataclass
class BvpForm:
    """팀 대 특정 투수 감쇠가중 상대전적(bvp_edge용). plays.csv 기반."""

    ab: float = 0.0
    h: float = 0.0

    DECAY = 0.94  # 맞대결 자체가 이미 드문 사건이라 배팅/불펜보다 감쇠를 느리게

    def shrunk_avg(self) -> float:
        return (self.h + BVP_PRIOR_AB * LEAGUE_AVG_BATTING) / (self.ab + BVP_PRIOR_AB)

    def add(self, ab: int, h: int) -> None:
        self.ab = self.ab * self.DECAY + ab
        self.h = self.h * self.DECAY + h


def _add_bvp_result(
    plays_for_game: list[tuple[str, str, str]],
    bvp_form: dict[tuple[str, str], BvpForm],
) -> None:
    """이 경기의 타석 결과를 (타격팀, 상대투수) 단위로 합산해서 감쇠 누적한다."""
    totals: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: {"ab": 0, "h": 0})
    for batteam, pitcher, event in plays_for_game:
        if not batteam or not pitcher:
            continue
        is_ab, is_hit = classify_event(event)
        if not is_ab:
            continue
        key = (batteam, pitcher)
        totals[key]["ab"] += 1
        totals[key]["h"] += int(is_hit)
    for key, stat in totals.items():
        bvp_form[key].add(stat["ab"], stat["h"])


@dataclass
class DefenseForm:
    """팀의 감쇠가중 실책률(defense_edge용). teamstats.csv의 d_e(실책 수)를 쓴다."""

    games: float = 0.0
    errors: float = 0.0

    DECAY = 0.90

    def error_rate(self) -> float:
        if self.games == 0:
            return 0.60  # 리그 평균급 경기당 실책 수(대략치)
        return self.errors / self.games

    def add(self, errors: int) -> None:
        self.games = self.games * self.DECAY + 1
        self.errors = self.errors * self.DECAY + errors


def _add_defense_result(
    teamstats_rows_for_game: list[dict[str, str]],
    defense_form: dict[str, DefenseForm],
) -> None:
    """teamstats.csv는 이미 팀당 한 줄이라 batting처럼 합산할 필요가 없다."""
    for row in teamstats_rows_for_game:
        team = row.get("team")
        if not team:
            continue
        defense_form[team].add(_to_int(row.get("d_e")))


def build_training_rows(
    gameinfo_rows: list[dict[str, str]],
    pitching_rows: list[dict[str, str]],
    start_date: str,
    end_date: str,
    teamstats_rows: list[dict[str, str]] | None = None,
    batting_rows: list[dict[str, str]] | None = None,
    plays_rows: list[dict[str, str]] | None = None,
) -> list[dict[str, object]]:
    """Build v2 feature snapshots plus full-game/F5 score labels.

    ``home_win`` remains as a temporary compatibility label for the existing
    logistic baseline. The run labels are the contract needed by the upcoming
    full-game and first-five joint score-distribution models.
    """
    games = [
        game
        for game in gameinfo_rows
        if game.get("date")
        and start_date <= game["date"] <= end_date
        and game.get("gid")
        and game.get("hometeam")
        and game.get("visteam")
        and game.get("wteam")
        and game.get("lteam")
    ]
    games.sort(key=lambda game: (game["date"], game["gid"]))

    pitching_by_game: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in value_stat_rows(pitching_rows):
        if row.get("gid"):
            pitching_by_game[row["gid"]].append(row)

    batting_by_game: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in value_stat_rows(batting_rows or []):
        if row.get("gid"):
            batting_by_game[row["gid"]].append(row)

    # plays.csv에는 stattype 컬럼이 없다(공식 문서 확인) - value_stat_rows 필터 안 씀
    plays_by_game: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for row in plays_rows or []:
        if row.get("gid"):
            # Simplified CSV supplies official counting fields. Prefer them to
            # reinterpreting the event notation (sacrifices, double plays, etc.).
            event = row.get("event", "")
            if row.get("ab") not in (None, ""):
                hit = sum(_to_int(row.get(key)) for key in ("single", "double", "triple", "hr"))
                event = "S" if hit else "63" if _to_int(row["ab"]) else "NP"
            plays_by_game[row["gid"]].append(
                (row.get("batteam", ""), row.get("pitcher", ""), event)
            )

    teamstats_by_game_team: dict[tuple[str, str], dict[str, str]] = {}
    for row in value_stat_rows(teamstats_rows or []):
        if row.get("gid") and row.get("team"):
            teamstats_by_game_team[(row["gid"], row["team"])] = row

    starter_form: dict[str, PitcherForm] = defaultdict(PitcherForm)
    bullpen_form: dict[str, PitcherForm] = defaultdict(PitcherForm)
    batting_form: dict[str, BattingForm] = defaultdict(BattingForm)
    defense_form: dict[str, DefenseForm] = defaultdict(DefenseForm)
    bvp_form: dict[tuple[str, str], BvpForm] = defaultdict(BvpForm)
    last_game_date: dict[str, _date] = {}
    wins: dict[str, int] = defaultdict(int)
    losses: dict[str, int] = defaultdict(int)
    h2h_wins: dict[tuple[str, str], int] = defaultdict(int)
    output: list[dict[str, object]] = []

    for game in games:
        gid = game["gid"]
        home, away = game["hometeam"], game["visteam"]
        game_date = _parse_date(game["date"])
        pitchers = pitching_by_game.get(gid, [])
        batters = batting_by_game.get(gid, [])
        plays = plays_by_game.get(gid, [])
        home_starter = next(
            (
                pitcher
                for pitcher in pitchers
                if pitcher.get("team") == home and _to_int(pitcher.get("p_seq")) == 1
            ),
            None,
        )
        away_starter = next(
            (
                pitcher
                for pitcher in pitchers
                if pitcher.get("team") == away and _to_int(pitcher.get("p_seq")) == 1
            ),
            None,
        )

        if home_starter is not None and away_starter is not None:
            home_starter_era = starter_form[home_starter["id"]].era()
            away_starter_era = starter_form[away_starter["id"]].era()
            home_bullpen_era = bullpen_form[home].era()
            away_bullpen_era = bullpen_form[away].era()
            home_slg = batting_form[home].slg()
            away_slg = batting_form[away].slg()

            home_rest = (
                (game_date - last_game_date[home]).days if home in last_game_date else 4
            )
            away_rest = (
                (game_date - last_game_date[away]).days if away in last_game_date else 4
            )
            home_rest = max(-2, min(6, home_rest - 4))
            away_rest = max(-2, min(6, away_rest - 4))

            home_played = wins[home] + losses[home]
            away_played = wins[away] + losses[away]
            home_pct = wins[home] / home_played if home_played >= MIN_GAMES_FOR_TEAM_PCT else 0.5
            away_pct = wins[away] / away_played if away_played >= MIN_GAMES_FOR_TEAM_PCT else 0.5
            matchup = team_matchup_edge(
                home_pct,
                away_pct,
                h2h_wins[(home, away)],
                h2h_wins[(away, home)],
            )
            home_error_rate = defense_form[home].error_rate()
            away_error_rate = defense_form[away].error_rate()
            # bvp_edge: 홈타선 vs 원정선발 평균 상대전적 - 원정타선 vs 홈선발 평균 상대전적
            home_vs_away_starter = bvp_form[(home, away_starter["id"])].shrunk_avg()
            away_vs_home_starter = bvp_form[(away, home_starter["id"])].shrunk_avg()

            output.append(
                {
                    "feature_snapshot_id": f"retrosheet:{gid}:{FEATURE_VERSION}",
                    "event_id": gid,
                    "as_of_timestamp": _as_of_timestamp(game),
                    "feature_version": FEATURE_VERSION,
                    "starter_edge": round((away_starter_era - home_starter_era) / 4.0, 4),
                    "bullpen_edge": round((away_bullpen_era - home_bullpen_era) / 4.0, 4),
                    "lineup_edge": round((home_slg - away_slg) * 10.0, 4),
                    "team_matchup_edge": round(matchup, 4),
                    "bvp_edge": round((home_vs_away_starter - away_vs_home_starter) * 10.0, 4),
                    "availability_edge": 0.0,
                    "defense_edge": round((away_error_rate - home_error_rate) * 2.0, 4),
                    "weather_edge": 0.0,
                    "rest_edge": round((home_rest - away_rest) / 4.0, 4),
                    "away_runs": _optional_int(game.get("vruns")),
                    "home_runs": _optional_int(game.get("hruns")),
                    "away_f5_runs": _f5_runs(teamstats_by_game_team.get((gid, away))),
                    "home_f5_runs": _f5_runs(teamstats_by_game_team.get((gid, home))),
                    "home_win": 1 if game.get("wteam") == home else 0,
                }
            )

        # Games that cannot emit a feature row still belong in future state.
        _add_pitching_result(pitchers, starter_form, bullpen_form)
        _add_batting_result(batters, batting_form)
        _add_defense_result(
            [r for r in (teamstats_by_game_team.get((gid, home)), teamstats_by_game_team.get((gid, away))) if r],
            defense_form,
        )
        _add_bvp_result(plays, bvp_form)
        last_game_date[home] = game_date
        last_game_date[away] = game_date
        if game.get("wteam") == home:
            wins[home] += 1
            losses[away] += 1
            h2h_wins[(home, away)] += 1
        else:
            wins[away] += 1
            losses[home] += 1
            h2h_wins[(away, home)] += 1

    return output


FEATURE_COLUMNS = (
    "feature_snapshot_id",
    "event_id",
    "as_of_timestamp",
    "feature_version",
    *FEATURE_NAMES,
    "away_runs",
    "home_runs",
    "away_f5_runs",
    "home_f5_runs",
    "home_win",
)


def write_training_csv(rows: list[dict[str, object]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FEATURE_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in FEATURE_COLUMNS})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert an extracted Retrosheet Main/Simplified CSV bundle to v2 rows"
    )
    parser.add_argument(
        "source",
        type=Path,
        help="Retrosheet ZIP or a directory previously created by retrosheet.py",
    )
    parser.add_argument(
        "--source-url",
        help="Official Retrosheet URL; required when source is a ZIP",
    )
    parser.add_argument("--start-date", required=True, help="YYYYMMDD")
    parser.add_argument("--end-date", required=True, help="YYYYMMDD")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if args.source.is_file():
        if not args.source_url:
            parser.error("--source-url is required when source is a ZIP")
        inspect_bundle(args.source, args.source_url)
        reader = lambda filename, value_only=False, columns=None: read_bundle_csv(  # noqa: E731
            args.source,
            filename,
            start_date=args.start_date,
            end_date=args.end_date,
            value_only=value_only,
            columns=columns,
        )
    elif args.source.is_dir():
        reader = lambda filename, value_only=False, columns=None: read_csv(  # noqa: E731
            args.source / filename,
            start_date=args.start_date,
            end_date=args.end_date,
            value_only=value_only,
            columns=columns,
        )
    else:
        parser.error(f"source does not exist: {args.source}")

    gameinfo = reader("gameinfo.csv")
    pitching = reader("pitching.csv", value_only=True)
    teamstats = reader("teamstats.csv", value_only=True)
    batting = reader("batting.csv", value_only=True)
    print("plays.csv 읽는 중 (9GB+ 압축해제 대상이라 다른 파일보다 훨씬 오래 걸립니다)...")
    plays = reader("plays.csv", columns=("gid", "batteam", "pitcher", "event", "ab", "single", "double", "triple", "hr"))
    rows = build_training_rows(
        gameinfo,
        pitching,
        args.start_date,
        args.end_date,
        teamstats_rows=teamstats,
        batting_rows=batting,
        plays_rows=plays,
    )
    write_training_csv(rows, args.out)
    print(f"games={len(rows)} feature_version={FEATURE_VERSION} -> {args.out}")
    print(
        "pending neutral features: availability_edge, weather_edge"
    )


if __name__ == "__main__":
    main()
