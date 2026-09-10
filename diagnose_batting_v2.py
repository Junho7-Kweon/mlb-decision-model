import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from mlb_decision_model.retrosheet_etl import read_bundle_csv, value_stat_rows, _add_batting_result, BattingForm
from collections import defaultdict

if len(sys.argv) != 2:
    raise SystemExit("사용법: python diagnose_batting_v2.py <csvdownloads.zip>")
ZIP_PATH = sys.argv[1]

print("1) gameinfo.csv에서 2022년 4월 첫 20경기 gid 추출...")
gameinfo = read_bundle_csv(ZIP_PATH, "gameinfo.csv", start_date="20220401", end_date="20220410")
gameinfo.sort(key=lambda g: (g.get("date",""), g.get("gid","")))
sample_gids = [g["gid"] for g in gameinfo[:20] if g.get("gid")]
print(f"   샘플 gid 20개: {sample_gids[:5]} ...")

print("\n2) batting.csv에서 같은 기간 읽기...")
batting = read_bundle_csv(ZIP_PATH, "batting.csv", start_date="20220401", end_date="20220410", value_only=True)
print(f"   읽힌 batting 행 수: {len(batting)}")

print("\n3) 샘플 gid들에 대해 실제로 batting 행이 매칭되는지 확인...")
batting_by_gid = defaultdict(list)
for row in value_stat_rows(batting):
    if row.get("gid"):
        batting_by_gid[row["gid"]].append(row)

matched = 0
for gid in sample_gids:
    rows = batting_by_gid.get(gid, [])
    if rows:
        matched += 1
        if matched <= 2:
            print(f"   gid={gid}: {len(rows)}행 매칭됨. 첫 행 team={rows[0].get('team')} b_ab={rows[0].get('b_ab')} b_h={rows[0].get('b_h')}")
print(f"   샘플 20개 gid 중 batting 데이터 매칭된 것: {matched}개")

print("\n4) _add_batting_result()로 실제 누적까지 해보기...")
batting_form = defaultdict(BattingForm)
for gid in sample_gids:
    _add_batting_result(batting_by_gid.get(gid, []), batting_form)
for team, form in list(batting_form.items())[:3]:
    print(f"   team={team}: ab={form.ab:.1f} h={form.h:.1f} hr={form.hr:.1f} slg={form.slg():.3f}")
if not batting_form:
    print("   ❌ batting_form이 완전히 비어있음 - 여기가 문제 지점")
