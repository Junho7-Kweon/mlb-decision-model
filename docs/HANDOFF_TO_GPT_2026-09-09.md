# GPT 인수인계 — 클로드 작업 전체 요약 (2026-09-09 기준)

이 zip이 지금까지 클로드랑 작업한 **확정 최종본**입니다. 이 위에서 이어서 작업해주세요 —
새 파일을 만들거나 구조를 다시 짜지 말고, 아래 내용을 참고해서 수정해주시면 됩니다.

## 1. 피처 스키마 (v2.0, 9개 — 순서·이름 고정)

```
starter_edge, bullpen_edge, lineup_edge, team_matchup_edge, bvp_edge,
availability_edge, defense_edge, weather_edge, rest_edge
```

| 피처 | 상태 | 데이터 출처 |
|---|---|---|
| starter_edge | ✅ 구현 | pitching.csv, 선발 ERA 감쇠가중 |
| bullpen_edge | ✅ 구현 | pitching.csv, 불펜 ERA 감쇠가중(계투는 경기당 1회만 감쇠) |
| lineup_edge | ✅ 구현 | batting.csv (`b_ab,b_h,b_d,b_t,b_hr`), 팀 SLG 프록시 |
| team_matchup_edge | ✅ 구현 | 로그5 기대승률 대비 실제 맞대결 잔차, Bayesian shrinkage |
| bvp_edge | ✅ 구현(신규) | plays.csv, 팀-투수 상대전적, event 코드 파서 포함 |
| defense_edge | ✅ 구현(신규) | teamstats.csv의 `d_e`(실책), 감쇠가중 실책률 |
| rest_edge | ✅ 구현 | gameinfo.csv 날짜 간격 |
| availability_edge | ⬜ 미구현 | Retrosheet에 부상자 명단 없음 - 자동화 경로 없음 |
| weather_edge | ⬜ 미구현(학습용) | 과거 날씨 데이터 없음. 당일 예측(predict_today.py)엔 NWS API로 연결 가능(미착수) |

**중요**: batting.csv 컬럼은 `b_2b`/`b_3b`가 아니라 **`b_d`/`b_t`**입니다(공식 문서로 확인 완료,
실제 헤더로 재확인함). plays.csv에는 `stattype` 컬럼이 없습니다(다른 파일과 다름) - 읽을 때
`value_only` 필터 적용하면 안 됩니다.

## 2. 실제 데이터 검증 결과 (합성 아님, 진짜 Retrosheet 9,893경기)

시간순 80/20 walk-forward 백테스트:

| 모델 | Brier | LogLoss | 정확도 |
|---|---|---|---|
| 기존 로지스틱 | 0.2452 | 0.6835 | 56.0% |
| 새 포아송(득점분포) | 0.2460 | 0.6850 | 54.5% |
| 기준선(항상 홈52.8%) | 0.2490 | — | — |

이건 **lineup_edge/defense_edge/bvp_edge가 전부 0이었을 때**(4/9 피처만 채워진 상태) 결과입니다.
이번에 3개를 새로 구현했으니, `retrosheet_etl` 재실행 후 같은 백테스트를 다시 돌려서
Brier score가 개선되는지 확인하는 게 다음 최우선 작업입니다.

## 3. 새로 생긴 파일들

- `score_model.py` — 포아송 득점분포 모델(P(원정득점=x, 홈득점=y)). 홈·원정 득점을 각각
  회귀하고, 그리드에서 승패·언더오버·핸디캡 확률을 계산합니다.
- `train_score_model.py` — 위 모델을 `training_games.csv`로 학습하는 CLI.
- `predict_today.py` — 오늘 경기 피처(사람이 조사해서 입력) → `model_probabilities.json` 생성.
- `model_probability.py` — 이건 GPT가 만든 파일인데, 그대로 살려뒀습니다. 캡처 배당에
  모델 확률을 연결하고, 모델이 없으면 market_no_vig로 대체하지 않고 명확히 차단합니다
  (예전 "EV가 항상 마이너스" 버그의 근본 해결책).
- `bet_tracker.py` — 모델 학습과 완전히 별개인 개인 베팅 기록 도구. `data/private/bet_tracker.csv`에
  누적되며, 요소별(선발폼/불펜/핫스트릭 등) 적중률을 집계합니다.

## 4. 대시보드 (dashboard.html)

MLB 네이비(#041e42)+레드(#bf0d3e) 스타일로 리디자인. 5개 마켓(승패·승1패·핸디캡·언더오버·홀짝)
전부 픽 가능. **캡처 한 장에 경기 2개 이상 있으면 경기별로 먼저 나누고 마켓별로 표시**하도록
수정함(원래 다 섞여 나오던 버그 있었음). 실제 브라우저(Playwright)로 끝까지 검증 완료.

## 5. 알려진 남은 문제 (docs/OCR_REVIEW_2026-09-04.md 참고)

- `predict_today.py`의 `event_id`가 여전히 팀명 문자열 매칭이라, OCR 경기명이랑 정확히
  일치해야 함 - canonical ID 도입이 근본 해결책이지만 아직 안 함.
- OCR이 캡처 이미지 소수점 인식 실패할 때가 있음(예: "2.37"→"2374") - 폴백 파서가
  이 경우 다른 마켓 값을 잘못 가져올 위험 있음.

## 6. 교훈 (data/private/LESSONS.md에 상세)

- 시즌 평균 스탯을 그날 확률로 바로 쓰면 과신하게 됨 - 실전에서 여러 번 확인됨.
- "선발 둘 다 좋음 + 불펜 둘 다 약함" 조합은 연장전 리스크가 커서 총점 예측에 위험 요인.
- 컬럼명은 절대 추측하지 말고 항상 실제 헤더로 확인할 것(2번 실수함).

## 7. GPT 적용·검증 결과 (2026-09-09)

- 확정 ZIP 50개 파일을 현재 프로젝트에 반영. 기존 겹치는 파일은
  `tmp/before-handoff-20260909-094934.zip`으로 백업했다.
- 실제 Retrosheet CSV 헤더를 확인하고 타격·수비·팀 대 투수 피처를 포함한
  9,893경기를 `data/processed/training_games_updated.csv`로 재생성했다.
  plays.csv의 공식 ab/single/double/triple/hr 필드를 우선 사용한다.
- 동일 경기의 고정 시간순 80/20 holdout(학습 7,914 / 평가 1,979) 비교:
  기존 포아송 Brier 0.24598, 새 피처 포아송 0.24695.
  기존 로지스틱 0.24522, 새 피처 로지스틱 0.24563.
  새 후보는 성능이 악화되어 `models/score_model_candidate.json`에만 저장했다.
  기존 `models/score_model.json`을 유지하며, 결과는
  `data/processed/model_evaluation_2026-09-09.json`에 있다.
- OCR은 셀 내부 여백·확대율·행 재인식을 보강했다. 소수점 없는 불명확한 값을
  추정 확정하지 않는다. SUM/언더오버를 승패로 잘못 넘기는 폴백을 차단했다.
- 대시보드는 캡처당 하나가 아니라 활성 후보 전체를 전달한다. 경기·기간·기준점별
  표시, Top 5 카드, 전체 후보 중 실제 최저 확률을 연결했다. file://도 로컬 API를 쓴다.
- 모델 상한 밖 확률 보존, full/F5 artifact 검사, 적특 미정 정수 기준점 차단,
  당일 예측 만료·사전 시각 검사, 서버 확률 조인 검사를 추가했다.
- 자동 테스트 81개와 실제 이미지 HTTP OCR→후보→Top 5 스모크 테스트 통과.
  HTTP 검사의 합성 확률은 임시 폴더에서만 사용하고 제거했다.

남은 경계: 당일 선발·라인업 등 사전 피처 자동 수집, canonical 경기 ID 자동 매핑,
승1패·홀짝·F5 정산 연결은 아직 완료되지 않았다. 경기명 별칭은 명시적으로 확인해
등록할 수 있다. 포아송의 독립 득점·비동점 조건부 승패는 baseline 근사이며,
승패 holdout 결과를 언더오버·핸디캡 수익성 검증으로 해석하지 않는다.

## 8. 2026-09-10 MLB 공식 API 연결 작업 — 이 절이 최신 상태

### 구현 완료

- `score_model.py`의 정규경기/F5 모델은 포아송에서 **음이항 득점분포**로 변경됐다.
  현재 배포 artifact는 `score_distribution_nb_v1`이며, 과산포를 별도로 추정한다.
- `mlb_daily_data.py`를 추가해 MLB Stats API에서 API 키 없이 아래 자료를 가져온다.
  배당은 이 API에서 가져오지 않으며 기존처럼 OCR 캡처 입력만 사용한다.
  - 한국 날짜 기준 시작 전 정규시즌 경기와 canonical `gamePk`
  - 홈/원정 팀과 예정 선발
  - 2021년 이후 정규시즌 팀 경기별 타격·투구·수비 기록
  - 투수별 경기 로그와, 시즌 로스터에서 빠진 투수의 공식 경기 boxscore 보충
- API 기록은 학습 때 사용한 `retrosheet_decay_v2` 규칙으로 다시 재생한다.
  선발 ERA, 팀 불펜 ERA, 팀 SLG, 실책률, 승패·상대전적, 휴식을 경기별 피처로 만든다.
- 팀 전체 투구 아웃/상대한 타자 수와 개별 투수 합계를 대조한다. 기록이 빠졌는데
  boxscore로도 복구하지 못하거나 예정 선발이 발표되지 않으면 평균값을 넣지 않고
  해당 경기를 `pending`으로 보류한다.
- MLB API의 연기 경기가 `abstractGameState=Final`로 표시되는 사례를 확인해,
  실제 완료 상태 코드와 득점 존재 여부를 함께 검사한다.
- `dashboard.py`의 `/api/daily-refresh`가 MLB 수집기를 호출하도록 변경됐다.
  브라우저의 경기 날짜는 한국시간으로 서버에 전달되며 API 키 입력 UI는 제거했다.
- `pregame.py`가 OCR 팀명/별칭을 canonical `mlb:{gamePk}`에 연결한 뒤 학습 모델의
  점수분포에서 승패·승1패·핸디캡·언더오버·홀짝 확률을 각각 계산한다.
  브라우저가 보낸 `probability` 필드는 서버에서 제거하므로 사용자가 모델 확률을
  입력하거나 조작할 수 없다.
- `DATA_SOURCES.md`, `sources.py`, `config/data_sources.json`의 출처 정책도 갱신했다.
  사용자가 개인 연구용 연결을 요청해 프로젝트 내부에서 허용한 것이며, 이것은
  MLB의 오픈 라이선스나 자동수집 허가를 확인했다는 뜻이 아니다. 로컬 캐시,
  호출 간격, 호출 상한을 적용하고 접근 제한 우회는 하지 않는다.

### 실제 연결 검증 (2026-09-10)

- MLB 일정 응답에서 `gamePk`, 시작 시각, 양 팀, 예정 선발을 실제 확인했다.
- 한국시간 2026-09-11 시작 예정 경기 5개를 조회했다.
  - 자동 피처 생성 완료: 4경기
  - 예정 선발 미발표로 보류: 1경기 (`mlb:824550`)
- 실제 애틀랜타 브레이브스-탬파베이 레이스 경기(`mlb:824872`)의 자동 피처로
  정규경기 승 확률 0.519718, 패 확률 0.480282를 생성했다. 두 확률은 배당과 무관하다.
- 준비된 2경기에서 승패와 U/O 8.5 후보 8개를 넣어 2폴 전체 16개 조합과 Top 5를
  생성했다. 이 검사의 1.90 배당은 **파이프라인 검증용 합성값**이며 추천 성과 자료가 아니다.
- 준비된 4경기에서 F5 승/무/패와 U/O 4.5 확률 16건도 생성했다.
- 브라우저가 고의로 `probability=0.9999`를 전송한 HTTP 검사에서 해당 값이 무시되고
  서버 모델 확률로 교체되는 것을 확인했다.
- Python 테스트 **103개 통과**, 대시보드 JavaScript 구문 검사 통과.

검증 명령:

```powershell
$env:PYTHONPATH='src'
python -m unittest discover -s tests
python -m mlb_decision_model.mlb_daily_data --date 2026-09-11
python -m mlb_decision_model.dashboard --port 8765
```

### 파일별 핵심 변경

- `src/mlb_decision_model/mlb_daily_data.py`: MLB 일정·기록 수집, 캐시, 검증, 피처 생성
- `src/mlb_decision_model/dashboard.py`: MLB 갱신 API 및 상태 연결
- `src/mlb_decision_model/static/dashboard.html`: 한국 날짜·갱신 UI, 키 입력 제거
- `src/mlb_decision_model/pregame.py`: 사전자료 검증과 모델 확률 결합
- `src/mlb_decision_model/sources.py`, `config/data_sources.json`, `DATA_SOURCES.md`: 출처 정책
- `tests/test_mlb_daily_data.py`: 기록 완전성, boxscore 복구, 연기 상태 등 회귀 테스트
- `tests/test_mlb_decision_model.py`: 변경된 출처 정책과 라이브 캐시 격리

### “최종” 판정과 남은 검증

**현재 범위의 코드 파이프라인은 완료**다:

`MLB 경기·선발·과거 기록 → 학습과 같은 피처 → NB 점수분포 → 마켓별 확률 → OCR 배당 → 2폴 전체 조합 → Top 5`

다만 **모델 자체의 최종 성능이 확정된 것은 아니다.** 다음 항목은 별도의 모델 고도화다.

1. 현재 `lineup_edge`는 당일 출전 타자 명단이 아니라 최근 팀 SLG 프록시다.
2. 부상, 당일 실제 라인업 교체, 날씨는 현재 모델 가중치가 0이며 미반영이다.
3. 서로 다른 경기의 조합 확률은 독립을 가정한다.
4. Brier 0.2460은 과거 승패 holdout 결과다. 언더오버·핸디캡·승1패·홀짝의
   캘리브레이션과 실제 배당 수익성은 각각 별도 백테스트가 필요하다.
5. 실제 당일 캡처로 OCR부터 Top 5까지 반복 운영하며 팀명 별칭과 소수점 오류를
   계속 기록해야 한다.

따라서 클로드 교차검증에서는 이번 코드를 “실사용 가능한 1차 완성본”으로 보고,
위 5개를 성능·운영 고도화 항목으로 분리해 검토하면 된다. 이전 7절의 “당일 피처 미연결,
승1패·홀짝·F5 미완료” 문장은 8절 구현으로 대체됐다.
