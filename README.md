# MLB Decision Model

> 현재 상태: 모델 구조와 테스트용 입력 형식까지 구현된 검토 버전이다.
> 실제 MLB 과거 데이터 학습과 수익성 검증 전에는 실전 예측기로 간주하지 않는다.

이 모형은 사용자가 고른 방향만 확인하지 않고 홈/원정, 오버/언더 양쪽을 모두 비교한다.

## 반영 요소

- 선발: ERA, FIP, xERA, K%, BB%, 땅볼률, 휴식, 최근 구속 변화
- 불펜: 투수별 기량, 최근 1일/3일 투구 수, 연투 여부, 레버리지 역할
- 마무리: 별도 평가와 당일 가용성
- 선발 타선: 상대 선발의 좌우 유형별 wOBA
- 상대전적(BvP): 타율을 그대로 사용하지 않고 타석 수에 따라 좌우 스플릿으로 축소 보정
- 후보 선수: 상위 대타 후보 3명의 상대 손잡이별 공격력
- 수비, 휴식 차이
- 양쪽 배당의 손익분기 확률과 기대수익

## 데이터 원칙

경기 시작 전 시점에 알 수 있었던 데이터만 학습 행에 넣는다. 경기 후에 확정된 정보가 과거 행에 들어가면 데이터 누수가 발생한다. 시간순으로 앞 80%를 학습하고 뒤 20%를 확률 보정에 사용한다. 최소 수백 경기, 권장 3시즌 이상의 정규시즌 경기가 필요하다.

과거 학습자료는 Retrosheet, 선수 ID 연결은 Chadwick Register, 날씨는 NWS만
자동 처리한다. MLB StatsAPI·Baseball Savant·스포츠북 웹페이지는 자동수집하지
않는다. 세부 허용범위와 필수 고지는 `DATA_SOURCES.md`와 `NOTICE.md`를 따른다.
등록되지 않은 데이터 출처는 기본적으로 차단한다.

학습 CSV는 `home_win`과 다음 열을 가진다.

```text
starter_edge,bullpen_edge,closer_edge,lineup_edge,bvp_edge,bench_edge,availability_edge,defense_edge,rest_edge,home_win
```

학습:

```powershell
$env:PYTHONPATH="src"
python -m mlb_decision_model.train data/processed/training_games.csv models/model.json
```

시간 순서 백테스트:

```powershell
python -m mlb_decision_model.backtest data/processed/training_games.csv --min-train 300
```

한 경기의 양방향 배당 판정:

```powershell
python -m mlb_decision_model.predict models/model.json data/samples/example_game.json --home-odds 1.85 --away-odds 2.05
```

테스트:

```powershell
$env:PYTHONPATH="src"
python -m unittest discover -s tests -v
```

Retrosheet 원본 준비는 [공식 CSV 다운로드 페이지](https://retrosheet.org/downloads/csvdownloads.html)에서
ZIP을 사용자가 직접 내려받은 뒤 수행한다. 가져오기 명령은 공식 도메인, 필수 CSV 7개,
ZIP 경로 안전성을 검사하고 SHA-256 및 취득 시각을 `provenance.json`으로 남긴다.

```powershell
$env:PYTHONPATH="src"
python -m mlb_decision_model.retrosheet C:\다운로드\retrosheet.zip `
  --source-url "공식 다운로드 링크" `
  --extract-to data\raw\retrosheet
```

이 명령은 웹사이트를 크롤링하거나 자동 다운로드하지 않는다.

## Retrosheet -> 학습 CSV 변환 (시점 안전)

추출된 CSV 폴더를 실제 학습 CSV로 바꾼다. 컬럼 정의는
[retrosheet.org 공식 문서](https://www.retrosheet.org/downloads/csvcontents.html) 기준.

```powershell
python -m mlb_decision_model.retrosheet_etl data\raw\retrosheet `
  --start-date 20220101 --end-date 20251231 `
  --out data\processed\training_games.csv
```

**v1 구현 범위** (합성 데이터로 시점 안전성 검증 완료 — `tests/test_retrosheet_etl.py`.
실제 대용량 번들로는 미검증이므로 사용자가 직접 실행해서 확인해야 함):

- ✅ `starter_edge`, `bullpen_edge`, `closer_edge`(불펜으로 대체), `rest_edge`
- ⬜ `lineup_edge`, `bvp_edge`, `bench_edge`, `availability_edge`, `defense_edge` —
  전부 0.0(중립). 타자별 좌우 스플릿·상대전적은 `plays.csv`(타석 단위 원자료)가
  있어야 정확한데 이번 v1에서는 손대지 않았다. 학습 자체는 안 깨지지만(분산 0인
  피처는 표준화 시 안전하게 무시됨) 이 5개는 아직 예측에 기여를 못 한다.

`build_training_rows()`는 경기를 (date, gid) 순으로 정렬한 뒤, 각 경기의 피처를
계산할 때 그 경기 **이전까지** 누적된 상태만 읽고, 계산이 끝난 뒤에야 그 경기
결과를 상태에 반영한다 - 코드 순서 자체가 데이터 누수를 막도록 짜여 있다.

## 캡처 이미지에서 배당 자동 추출 (OCR, 완전 선택 사항)

```powershell
pip install -e ".[ocr]"
```

그리고 시스템에 Tesseract OCR 바이너리를 설치해야 한다(예: `apt-get install
tesseract-ocr tesseract-ocr-kor`). 대시보드에서 캡처를 올리면 "이미지에서 배당
추출" 버튼으로 **승패·언더오버 두 마켓의 배당 숫자만** 로컬에서 자동으로 읽어
픽 후보로 제안한다. 실제 캡처 3장으로 직접 테스트해 확인한 범위이며, 그 이유로
이 두 마켓만 지원한다 - 승1패(3자택일)·핸디캡 마켓은 두 번째 숫자의 소수점이
OCR로 종종 깨져서(`2.77` → `277`) 자동 추출 대상에서 제외했다.

이 기능은 **데이터 출처 정책과 무관**하다 - 사용자가 이미 로컬에 가진 이미지
파일을 읽는 것뿐이고, 이미지를 어떤 외부 서버로도 전송하지 않는다(`ocr_extract.py`
는 `requests`/`urllib`을 import조차 하지 않는다). Tesseract가 없으면 자동으로
수동 입력 모드로 넘어간다.

**중요**: OCR 결과는 항상 "초안"이며 경기명·모델 승률은 자동으로 채워지지
않는다. 팀명은 OCR로 자동 추출하지 않는다(오독 위험이 가장 큰 부분이라
일부러 뺐다) - 사람이 직접 입력해야 한다.

### 아직 자동화하지 않은 것: 모델 승률

**"캡처를 올리면 승률까지 자동으로 나온다"는 아직 구현하지 않았고, 의도적으로
보류 중이다.** 이유는 정책 때문이다 - 실시간으로 그럴듯한 승률을 만들려면
오늘 선발투수·최근 폼 같은 데이터가 필요한데, 이걸 자동으로 가져올 수 있는
**허용된** 출처가 지금 없다. Retrosheet는 과거 데이터라 오늘 경기엔 못 쓰고,
MLB StatsAPI는 individual·non-bulk만 허용되어 자동 파이프라인에 넣을 수 없다.
정식 해법은 Retrosheet ETL을 완성해서 학습된 `model.json`을 갖추는 것이다
(README 하단 TODO 참고). 그 전까지 모델 승률은 사람이 직접 조사해서 입력해야
한다.

## 로컬 조합 분석 대시보드

공공데이터포털 데이터셋 `15107776`은 무료 공식 경기결과 API지만 경기 종료
14일 후 제공되고 배당 필드가 없다. 결과 라벨 검증에만 사용하며, 당일 배당은
스냅샷과 확인 입력으로 기록한다.

```powershell
$env:PYTHONPATH="src"
python -m mlb_decision_model.dashboard
```

브라우저에서 `http://127.0.0.1:8765`를 연다. PNG/JPEG/WebP 배당 캡처와
경기별 실제 배당·모델 승률을 입력하면 2~4폴 조합을 생존확률 우선으로 정렬한다.
스냅샷은 `data/private/`에만 보관되어 Git에서 제외된다. 현재 계산은 경기 간
독립을 가정하므로 같은 경기의 승패·핸디캡·언더오버처럼 상관된 선택은 함께
분석하지 않는다. 같은 경기명을 입력한 복수 마켓은 조합 생성 단계에서 자동으로
서로 제외한다. 결과 화면은 최고 생존확률, 최고 기대값, 최저 생존확률, 전체 조합
확률 막대, 확률-기대값 산점도와 개별 픽 진단표를 제공한다.

기본적으로 생존확률 상위 5개, 기대값 상위 5개만 보여준다(`top_n`, 기본값 5).
전체를 다 보려면 요청에 `"top_n": null`을 넣으면 된다.

공공 API 활용신청 후 발급된 키는 `.env` 또는 환경변수
`DATA_GO_KR_SERVICE_KEY`에만 저장한다. 키는 Git에 커밋하지 않는다.

백테스트는 적중률뿐 아니라 Brier score와 log loss를 출력한다. ROI는 합성 배당으로
평가하지 않는다. 사용자가 경기 전에 직접 기록한 실제 배당이 있는 행에서만 참고용으로
계산하며, 표본이 충분히 쌓이기 전에는 수익성을 주장하지 않는다.

현재 구현은 외부 패키지 없이 실행된다. `data/samples/example_game.json`은 입력 구조 예시이며 실제 선수 데이터가 아니다.

## 최종 판정

- `BET`: 모델 확률이 배당 손익분기보다 기본 2.5%p 이상 높음
- `FADE`: 사용자의 방향이 2.5%p 이상 불리하여 반대편 검토
- `PASS`: 어느 쪽도 충분한 우위가 없음

2폴 이상은 모든 조합을 5만 회 시뮬레이션하여 적중률, 합산 배당, 기대수익과 포트폴리오 수익 확률을 비교한다.
