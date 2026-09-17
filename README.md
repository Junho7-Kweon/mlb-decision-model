# MLB Decision Model

과거 야구 데이터로 계산한 모델 확률과 캡처에서 읽은 실제 배당을 결합해,
서로 다른 경기의 가능한 조합 중 Top 5를 비교하는 로컬 도구입니다.

## 확정 기준

- 인수인계: [HANDOFF_TO_GPT_2026-09-09.md](docs/HANDOFF_TO_GPT_2026-09-09.md)
- 데이터 구조: [02_DATA_MODEL_ERD.md](docs/02_DATA_MODEL_ERD.md)
- 데이터 출처: [DATA_SOURCES.md](DATA_SOURCES.md), [NOTICE.md](NOTICE.md)

9개 피처 이름과 순서는 v2.0 계약을 유지합니다. 경기 데이터 → 점수분포 →
마켓 확률을 만든 뒤, 별도로 읽은 배당과 결합합니다. 사용자는 모델 확률을
입력하지 않습니다. 캡처·개인 기록은 로컬에서 처리합니다.

## 현재 상태 (2026-09-10)

| 부분 | 상태 |
|---|---|
| Retrosheet ETL | 실제 9,893경기 재생성. 선발·불펜·타격·팀 상대전적·팀 대 투수·수비·휴식 계산 |
| 미수집 피처 | availability_edge, weather_edge는 0. 실제 정보가 반영된 것으로 해석하지 않음 |
| 로지스틱 | 홈 승률 비교 기준 모델. 보정 구간은 학습 구간 내부에 배치 |
| 음이항 점수분포 | 홈·원정 득점의 과산포 추정. 꼬리 정규화, full/F5 artifact 분리 |
| 모델 선택 | 기존 피처 음이항 운영 모델과 별도 F5 모델 사용 |
| 마켓 예측 | 승패·승1패·핸디캡·언더오버·홀짝과 F5 승무패/언더오버 연결 |
| 당일 입력 | MLB Stats API에서 일정·예정 선발·경기별 기록을 자동 수집해 동일 피처로 변환 |
| OCR | 여러 경기 분리, 행/셀 단위 추출. 불명확한 값은 확인필요 및 초기 비활성 |
| 조합 화면 | 활성 후보 전체 조합에서 Top 5 카드·차트·표·개별 진단. 동일 경기 중복 제외 |

모델 파일의 존재와 당일 예측 준비는 다른 상태입니다. 한국 날짜를 선택하고
경기 시작 전에 `당일 정보 갱신`을 실행해야 합니다. MLB `gamePk`를 canonical ID로
사용하고 확인된 팀 별칭을 OCR 경기명과 연결합니다. 시작한 경기, 예정 선발 미발표,
기록 누락, 6시간이 지난 자료는 확률을 임의로 채우지 않고 차단합니다.

## 실제 비교 결과 (2026-09-09 후보 비교)

같은 9,893경기 중 앞 7,914경기를 학습하고 마지막 1,979경기를 평가했습니다.
검증 시작일은 2025-05-04입니다. 고정 시간순 holdout이며, 반복 재학습하는
walk-forward와 구분합니다. 아래는 승패 확률 평가이며 U/O·핸디캡 성능이나
실전 수익성을 입증하지 않습니다.

| 모델/피처 | Brier (낮을수록 좋음) | LogLoss | 정확도 |
|---|---:|---:|---:|
| 기존 피처 로지스틱 | 0.24522 | 0.68348 | 56.14% |
| 새 피처 로지스틱 | 0.24563 | 0.68430 | 55.48% |
| 기존 피처 포아송 | 0.24598 | 0.68504 | 54.47% |
| 새 피처 포아송 | 0.24695 | 0.68728 | 55.53% |

원시 결과: [model_evaluation_2026-09-09.json](data/processed/model_evaluation_2026-09-09.json).
새 후보 artifact는 models/score_model_candidate.json이며 운영 모델로 승격하지 않았습니다.
이 비교 이후 운영 점수분포는 과산포를 반영하는 음이항 모델로 변경됐습니다.
승패 Brier 0.2460 수준을 회복했지만, U/O·핸디캡 등 마켓별 성능과 실제 수익성은
별도 시간순 백테스트가 필요합니다.

## 실행

프로젝트 폴더의 PowerShell에서 실행합니다.

```powershell
$env:PYTHONPATH="src"
python -m mlb_decision_model.dashboard --port 8765
```

브라우저에서 http://127.0.0.1:8765 를 엽니다. HTML 파일 직접 열기도 같은
로컬 서버를 사용합니다. 이미지에서 읽은 선택지의 활성 여부를 클릭으로 조절하고,
확인필요 값은 원본과 비교 후 수정합니다. 한 경기의 여러 후보를 활성화할 수 있습니다.
경기 시작 전에 한국 날짜를 고르고 `당일 정보 갱신`을 누르면 MLB 경기별 모델
확률이 자동 생성됩니다. 모델 확률을 사용자가 입력하는 칸은 없습니다.

기존 로컬 Retrosheet ZIP에서 재생성:

```powershell
python -m mlb_decision_model.retrosheet_etl "C:\경로\csvdownloads.zip" --source-url "https://retrosheet.org/downloads/csvdownloads.html" --start-date 20220101 --end-date 20251231 --out data/processed/training_games_updated.csv
```

plays.csv는 크므로 처리에 시간이 걸립니다. 실제 CSV 헤더의 b_d/b_t,
ab/single/double/triple/hr를 사용하며 plays.csv에 stattype 필터를 적용하지 않습니다.

평가 및 후보 학습:

```powershell
python -m mlb_decision_model.train_score_model data/processed/training_games_updated.csv models/score_model.json --evaluate --baseline-csv data/processed/training_games.csv --report data/processed/model_evaluation_2026-09-09.json
python -m mlb_decision_model.train_score_model data/processed/training_games_updated.csv models/score_model_candidate.json
```

대용량 학습은 선택 의존성 numpy가 설치돼 있으면 동일한 경사 계산을 가속합니다.
`pip install -e ".[training,ocr]"`로 Python 의존성을 설치할 수 있고,
로컬 OCR에는 Tesseract와 한국어 언어팩이 추가로 필요합니다.

## 당일 예측

대시보드가 사용하는 자동 연결:

```powershell
python -m mlb_decision_model.mlb_daily_data --date 2026-09-11
```

MLB Stats API의 배당은 사용하지 않습니다. 모델은 일정·예정 선발·2021년 이후
정규시즌 경기 로그만 가져오며, 배당은 사용자가 제공한 OCR 캡처에서 분리해 읽습니다.
API 응답은 `data/private/mlb_cache`에 저장되어 Git에 포함되지 않습니다.

경기 종료 후에는 한국 날짜 기준으로 정규시즌 전체 확정 경기와 핵심 승부처를
별도 저장할 수 있습니다. 결승 리드를 만든 플레이, 최대 득점 플레이, 빅이닝,
역전 여부, 선취 5이닝/7회 이후 득점 등을 공식 play-by-play에서 구조화합니다.

```powershell
python -m mlb_decision_model.postgame_review --date 2026-09-17
```

결과는 `data/private/game_reviews/YYYY-MM-DD.json`에 저장되어 Git에 포함되지
않습니다. 원시 경기 결과는 MLB 출처 데이터이며, 프로젝트 고유 데이터는 그 위에
만든 승부처 라벨·예측 오차·실패 요인입니다. 전체 경기 결과는 다음 당일 피처의
팀 폼에 반영되지만, 새 승부처 라벨은 역사 데이터 백필과 시간순 검증을 통과하기
전에는 모델 가중치에 즉시 넣지 않습니다.

아래 `predict_today`는 사람이 검증한 피처 파일을 사용하는 보조 경로입니다.

games JSON의 각 경기는 event_id, features(v2.0의 9개 값),
as_of_timestamp, starts_at(시간대 포함), total_lines, handicap_lines를 가집니다.
선택적으로 event_aliases에 확인한 OCR 경기명을 등록할 수 있습니다.
features는 과거 학습과 같은 정의·척도로 계산해야 합니다.

```powershell
python -m mlb_decision_model.predict_today models/score_model.json data/private/today_games.json --out data/private/model_probabilities.json
```

today_games.json은 실제 경기 전 자료로 준비해야 합니다. 미래 시각의 피처와
이미 시작한 경기는 차단합니다. 예측 파일은 가장 빠른 경기 시작 시각에 만료됩니다.
전반용 모델을 정규경기 예측에 잘못 사용하거나 적특 규칙이 없는 정수 기준점을
입력하면 오류로 알립니다.

현재 음이항 모델은 홈·원정 득점 상관과 연장전 과정을 직접 모델링하지 않습니다.
현재 승패 확률은 비동점 조건부 근사이며, 다른 마켓과 완전히 일관된 정산 분포를
구성하는 개선이 필요합니다. 수익성은 실제 사전 배당과 결과가 쌓인 뒤 평가합니다.

## 검증

```powershell
python -m unittest discover -s tests -p "test_*.py"
python tests/manual_e2e_smoke_test.py "캡처1.png" "캡처2.png"
```

단위 테스트는 실제 사용자 캡처·API 캐시·개인 베팅 기록을 수정하지 않습니다.
실제 연결 검증 내용은 인수인계 문서의 8절에 기록되어 있습니다.
