# 이번 세션 변경사항 (Claude가 직접 구현)

요청: "캡처 두 장을 올리면 그 안의 경기를 읽고 최적의 조합 5개를 추천해달라"를
`mlb-decision-model`에 실제로 구현.

## 만든 것

1. **`src/mlb_decision_model/ocr_extract.py`** (신규) — 로컬 Tesseract OCR로
   캡처 이미지에서 승패·언더오버 두 마켓의 배당 숫자를 추출. 실제 캡처 2장을
   직접 OCR 돌려서 나온 원본 텍스트를 테스트 픽스처로 고정해뒀다(라벨이 깨지는
   실제 사례 포함). 이미지를 어디로도 전송하지 않으며, `sources.py` 정책 검사
   대상이 아니다(데이터 출처가 아니라 로컬 이미지 처리이므로).
2. **`dashboard.py`** — `/api/ocr-extract` 엔드포인트 추가(저장 없이 미리보기만
   반환). `analyze_payload`에 `top_n`(기본 5) 추가 — "상위 5개 조합 추천"을
   실제로 구현.
3. **`dashboard.html`** — 캡처 업로드 후 "이미지에서 배당 추출" 버튼. 추출된
   승/패/언더/오버 후보를 버튼으로 보여주고, 클릭하면 배당이 채워진 픽 행이
   생성된다(경기명·모델 승률은 여전히 직접 입력).
4. **테스트 7개 추가** (15개 → 22개, 전부 통과 확인).

## 의도적으로 안 만든 것 — 모델 승률 자동화

**이게 제일 중요한 부분입니다.** 오늘 실제 조사(RotoWire, ESPN, Baseball-Reference,
SI 등 웹서치)로 승률을 추정해서 회원님께 조합을 추천해드렸는데, 이 조사 과정을
그대로 코드에 자동화 파이프라인으로 넣지 않았습니다. 이유는 `DATA_SOURCES.md`에
저희가 이미 정해둔 정책과 정면으로 부딪히기 때문입니다:

- 웹서치로 긁어온 RotoWire·ESPN 같은 사이트는 `sources.py`의 허용 목록
  (retrosheet, chadwick_register, nws, data_go_kr_kspo_results, user_input)
  어디에도 속하지 않습니다.
- `require_approved_source()`는 미등록 출처를 기본 차단하도록 만들어져
  있고, 이건 저희가 여러 차례 리뷰를 거쳐 의도적으로 그렇게 설계한 겁니다.
- 이걸 우회해서 "그냥 스크립트로 스크레이핑"을 넣으면, 지금까지 쌓아온
  정책 자체가 무의미해집니다.

그래서 모델 승률은 여전히 사람이 입력해야 합니다. 정직한 해결 경로는 하나뿐입니다:
**Retrosheet ETL을 완성해서 실제로 학습된 `model.json`을 만드는 것.** 이게
README TODO에 계속 남아있는 이유이기도 합니다.

## 실행

```powershell
$env:PYTHONPATH="src"
python -m unittest discover -s tests -v   # 29개 통과해야 함

pip install -e ".[ocr]"                    # 선택사항
python -m mlb_decision_model.dashboard
```

## 이번에 새로 만든 것: Retrosheet ETL

`retrosheet_etl.py` — 추출된 Retrosheet CSV를 시점 안전(point-in-time-safe)한
학습 CSV로 변환. 실제 retrosheet.org 공식 컬럼 문서를 확인해서 짰고, 합성
데이터로 "이 경기 자체의 결과가 이 경기 피처에 새지 않는지"를 직접 테스트로
검증했다(`test_second_start_reflects_only_prior_game_not_current_or_future`).

- 구현: starter_edge, bullpen_edge, closer_edge(불펜 대체), rest_edge
- 미구현(0.0 중립): lineup_edge, bvp_edge, bench_edge, availability_edge,
  defense_edge — plays.csv 기반 작업이 필요해 v2로 남겨둠
- 실제 725MB 번들로는 못 돌려봤다(용량 제약) — 사용자가 직접 실행해서 검증 필요
