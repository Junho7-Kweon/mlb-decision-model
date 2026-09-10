# Claude 교차검증 결과

## 1. 전체 판정: **조건부 승인**

방향성(01·02) 자체는 건전하고, 특히 "모델 확률 레이어"와 "캡처 배당 레이어"를
분리한 것은 실제 코드에서 이미 확인된 버그(`market_no_vig`가 같은 배당에서
확률을 역산해서 EV가 구조적으로 항상 마이너스가 되는 문제)를 정확히 겨냥한
올바른 수정입니다. 다만 아래 P0 3건을 먼저 해소하지 않으면, 승인해도 구현
단계에서 다시 막힙니다.

---

## 2. 심각도별 발견사항

### P0

**P0-1. "과거 배당" 데이터 소스가 정책과 정면으로 모순됩니다.**
- `01_ARCHITECTURE_BLUEPRINT` §04 데이터 도메인 표: "과거 배당" = **필수**,
  §08 학습·평가 순서 3단계는 "당시 배당 시장 오퍼와 결합"을 요구합니다.
- `04_DATA_SOURCES_POLICY.md`의 허용 출처는 Retrosheet/Chadwick/NWS/KSPO/
  사용자 직접입력뿐입니다. 배당 이력 제공자는 아예 없고, `user_input`은
  "자동수집 불가"입니다.
- 즉 대량 과거 배당을 소급 확보할 정책상 경로가 없습니다. 가능한 건 오늘부터
  스냅샷을 organic하게 쌓는 것뿐이고, 이러면 §08의 ROI/드로다운 검증과 P8의
  `MODEL_EVALUATION.roi` 채우기는 실질적으로 **수개월~시즌 단위**로 지연됩니다.
  문서 어디에도 이 제약이 적혀 있지 않습니다 — 명시적으로 못박아야 합니다.

**P0-2. P1·P2가 P3·P4보다 먼저인데, 그 구간의 `model_probability` 출처가 안 정해졌습니다.**
- 로드맵(§12): P1(OCR·후보풀) → P2(조합엔진) → P3(데이터) → P4(베이스라인 모델).
- 그런데 `CANDIDATE.model_probability`는 P4에서 나오는 `SCORE_PROBABILITY`에
  의존합니다. P1~P2 구간엔 넣을 확률이 없습니다.
- 이 공백을 지금 코드의 `market_no_vig`(캡처 배당 자체에서 역산)로 메우면,
  "캡처 배당을 모델 확률로 오인하지 않는다"는 §01의 원칙을 P1~P2 내내
  스스로 어기게 되고, **지금 실제로 겪고 있는 그 버그가 그대로 재현**됩니다.
  이건 가정이 아니라 어제 실측으로 확인된 사실입니다.

**P0-3. `02_DATA_MODEL_ERD.md`의 `GAME_FEATURE_SNAPSHOT`이 현재 코드 스키마와 다릅니다.**
- ERD: `starter_edge, bullpen_edge, lineup_edge, team_matchup_edge, bvp_edge, availability_edge, defense_edge, weather_edge, rest_edge` (9개)
- 현재 코드(`features.py FEATURE_NAMES`, `model.py`, `retrosheet_etl.py`):
  `starter_edge, bullpen_edge, closer_edge, lineup_edge, bvp_edge, bench_edge, availability_edge, defense_edge, rest_edge`
- `closer_edge`, `bench_edge`가 빠지고 `team_matchup_edge`, `weather_edge`가
  새로 생겼습니다. 자동 마이그레이션되는 변경이 아니라, `model.json`도
  `FEATURE_NAMES`도 전부 다시 만들어야 하는 breaking change입니다.

### P1

**P1-1. Blueprint의 "18×11=198개 2폴" 워크드 예제가 실측 OCR 신뢰도를 넘어섭니다.**
- 제가 직접 캡처 3장을 Tesseract로 테스트한 결과: 승패·언더오버 2개 마켓만
  숫자가 안정적으로 읽혔고, 승1패·핸디캡은 소수점이 자주 깨졌습니다
  (`2.77` → `277`).
- `06_DASHBOARD_FEATURE_SPEC.md`(제가 작성)는 정확히 이 이유로 2마켓만
  지원한다고 명시했는데, 새 blueprint의 예제는 경기당 9개 마켓(18개
  선택지)을 전부 OCR로 뽑는다고 가정합니다. 이 간극을 어디서도 인정하지
  않습니다.

**P1-2. F5·핸디캡·홀짝처럼 세분화된 마켓의 캘리브레이션 표본 문제.**
- 권장 3시즌(~7,000경기)로도 이런 세부 마켓은 슬라이스당 표본이 얇습니다.
- P5 완료 기준이 "마켓별 calibration 통과"인데, 못 넘는 마켓은 어떻게
  되는지(후보 풀에서 계속 제외?) 문서에 없습니다.

### P2

- `MODEL_EVALUATION`에 `roi`, `max_drawdown`은 있는데 표본 수·신뢰구간
  필드가 없어서, 작은 표본에서 우연히 좋은 ROI가 나온 걸 승격 기준으로
  오인할 위험이 있습니다.
- ERD A에서 `PLAYER ||--o{ PLAY_EVENT`(batter/pitcher) 관계선이
  빠졌습니다 (PITCHING/BATTING/FIELDING_APPEARANCE는 그려져 있는데
  PLAY_EVENT만 빠짐) — §8 "FK 없이 문자열로 임시 조인하지 않는다"는
  스스로 정한 규칙과 대조하면 사소하지만 실제 누락입니다.
- `LEG_OUTCOME.outcome`이 ERD C엔 단순 string인데, `03_PIPELINE_DRAFT.md`엔
  WIN/LOSS/PUSH/HALF_WIN/HALF_LOSS/VOID enum이 명시돼 있습니다. ERD에도
  옮겨 적어야 문서 간 정합이 맞습니다.

---

## 3. 문서별 모순 요약표

| 문서 A | 문서 B | 모순 내용 |
|---|---|---|
| `01_BLUEPRINT` §04 (과거배당=필수) | `04_DATA_SOURCES_POLICY` | 필수 데이터의 획득 경로가 정책상 없음 |
| `01_BLUEPRINT` §02 (18×11 예제) | `06_DASHBOARD_FEATURE_SPEC` (2마켓 한정) | OCR 커버리지 가정 불일치 |
| `02_DATA_MODEL_ERD` GAME_FEATURE_SNAPSHOT | 현재 `features.py`/`model.py` | 피처 9개 구성 자체가 다름 |
| `05_PROJECT_README` "반영 요소"/학습CSV 스키마 절 | `01_BLUEPRINT`의 결합분포 구조 | README가 옛 단일 로지스틱 회귀 구조를 여전히 "확정"처럼 서술 (문서 상단엔 이미 새 blueprint 다이어그램을 넣어놔서 문서 **내부적으로도** 앞뒤가 안 맞음) |

---

## 4. ERD 수정 제안

1. `GAME_FEATURE_SNAPSHOT`: `closer_edge` 복원 여부를 결정하고(불펜에 통합할
   거면 §8 변경규칙에 그 결정을 한 줄 남길 것), `team_matchup_edge`·
   `weather_edge` 추가는 승인하되 현재 코드 마이그레이션 계획을 §6 표에
   같이 적을 것.
2. ERD A에 `PLAYER ||--o{ PLAY_EVENT : "batter_id"` / `"pitcher_id"` 관계선 추가.
3. `LEG_OUTCOME.outcome`을 enum(WIN/LOSS/PUSH/HALF_WIN/HALF_LOSS/VOID)으로 명시.
4. `MODEL_EVALUATION`에 `sample_size`, `roi_ci_low`, `roi_ci_high` 추가 권고.
5. `PLAYER.bats`/`throws`는 시점 스냅샷이 없는 유일한 핵심 필드입니다 —
   의도적(시즌 내 불변)이라면 주석으로 그 이유를 명시.

## 5. 파이프라인 수정 제안

1. `CANDIDATE.model_probability`에 `is_placeholder: bool` 필드를 추가하고,
   P4(SCORE_PROBABILITY) 완성 전까지는 이 값이 true인 카드는 EV·BET 배지를
   숨기고 "참고용 순위(모델 미완성)"라고만 표시할 것. 이러면 P1~P2를
   먼저 만들어도 P0-2의 재발을 막을 수 있습니다.
2. `04_DATA_SOURCES_POLICY.md`에 "과거 배당은 user_input 스냅샷의 organic
   누적만 허용되며, 이 때문에 P8 ROI 검증은 스냅샷이 충분히 쌓일 때까지
   지연된다"는 문장을 명시적으로 추가.
3. OCR 마켓 확장(승1패·핸디캡 등)은 "마켓별 실측 정확도가 기준을 넘을
   때만 후보 풀에 추가"하는 게이트를 §11 구현 순서에 명시.

## 6. 권장 구현 순서 (원안 대비 재배치 제안)

원안의 P0→P1→P2→...→P8 순서는 대체로 합리적이지만, **P3+P4를 P1+P2보다
먼저** 두는 걸 권합니다. 이유는 추상적이지 않습니다 — 지금 코드가 정확히
반대 순서(예쁜 UI 먼저, 실체 있는 확률은 나중)로 만들어졌고, 그 결과가
지금 실제로 겪고 있는 "그럴듯해 보이지만 EV가 항상 마이너스인 카드"입니다.
산 증거가 이미 있는 상태에서 같은 순서를 또 반복할 이유가 없습니다.

```
P0(스키마·계약)
→ P3(Retrosheet 실데이터로 GAME_FEATURE_SNAPSHOT까지)
→ P4(베이스라인 — 지금 있는 로지스틱 회귀부터 실제 Brier로 먼저 검증,
      NB+GBDT 앙상블은 베이스라인이 최소한 동전던지기보다 나은 걸 확인한 다음)
→ P1(OCR 후보풀 — 이미 검증된 2마켓부터, 위 게이트 적용)
→ P2(조합엔진 — 이 시점엔 진짜 model_probability가 있음)
→ P5 → P6 → P7 → P8
```

## 7. 첫 번째로 수정할 파일 하나

**`retrosheet_etl.py`**

이유: 새 ERD의 `GAME_FEATURE_SNAPSHOT`이 지금 코드의 학습 CSV 스키마와
필드 구성 자체가 다릅니다(P0-3). 이 파일이 학습 데이터를 만드는 유일한
진입점이라, 스키마 불일치를 여기서 먼저 해소해야 이후 모델 학습·대시보드
표시가 전부 같은 기준으로 움직입니다. 그리고 이 파일은 지금까지 실제
Retrosheet 데이터로 한 번도 검증된 적이 없다고 스스로 문서(`05_PROJECT_README`)
에 적혀 있으니, "새 스키마로 다시 짜는 동시에 처음으로 실제 데이터 검증까지"
같이 끝내는 게 지금 시점에서 레버리지가 가장 큽니다.
