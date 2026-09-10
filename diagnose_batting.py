import zipfile, csv, io, sys
from collections import Counter

if len(sys.argv) != 2:
    raise SystemExit("사용법: python diagnose_batting.py <csvdownloads.zip>")
ZIP_PATH = sys.argv[1]

with zipfile.ZipFile(ZIP_PATH) as z:
    name = [n for n in z.namelist() if n.lower().endswith("batting.csv")][0]
    with z.open(name) as raw, io.TextIOWrapper(raw, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        stattypes = Counter()
        total = 0
        sample_rows = []
        first_gid = None
        for row in reader:
            total += 1
            stattypes[row.get("stattype", "<없음>")] += 1
            if row.get("date", "").startswith("2022040"):  # 2022년 4월 초 아무 경기나
                if first_gid is None:
                    first_gid = row.get("gid")
                if row.get("gid") == first_gid and len(sample_rows) < 5:
                    sample_rows.append(row)
            if total >= 2_000_000:  # 안전장치
                break

print(f"총 batting.csv 행 수(스캔한 만큼): {total}")
print(f"stattype 분포: {dict(stattypes)}")
print(f"\n샘플 경기({first_gid}) 첫 5행:")
for r in sample_rows:
    print(f"  team={r.get('team')} b_ab={r.get('b_ab')} b_h={r.get('b_h')} b_hr={r.get('b_hr')} stattype={r.get('stattype')}")
