# 데이터 출처 정책

이 프로젝트는 **오픈데이터 전용, 미검토 출처 기본 차단** 정책을 사용한다.
공개적으로 접속할 수 있다는 사실만으로 오픈 라이선스라고 판단하지 않는다.

## 허용 출처

| 출처 | 사용 목적 | 자동/대량 처리 | 조건 |
|---|---|---:|---|
| [Retrosheet](https://retrosheet.org/downloads/csvdownloads.html) | 과거 경기·플레이·선수 기록 | 허용 | `NOTICE.md`의 지정 고지문 유지 |
| [Chadwick Register](https://github.com/chadwickbureau/register) | 선수 ID 연결 | 허용 | ODC-By 1.0 귀속 표시 |
| [NWS API](https://www.weather.gov/documentation/services-web-api) | 날씨 | 허용 | User-Agent 및 호출 제한 준수 |
| [공공데이터포털 KSPO 경기결과 API](https://www.data.go.kr/data/15107776/openapi.do) | 프로토 대상 경기 결과 검증 | 허용 | 경기 종료 14일 후 제공, 배당 없음, 인증키·호출량 준수 |
| 사용자 직접 입력 | 배당·당일 라인업 | 자동수집 불가 | 로컬 비공개 보관 |

## 차단 출처

- MLB StatsAPI 및 Baseball Savant 자동수집
- 스포츠북 웹페이지 크롤링
- The Odds API(정식 서비스지만 이 프로젝트가 요구하는 오픈데이터에는 해당하지 않음)
- 출처, 라이선스, 취득 시점이 검증되지 않은 GitHub/Kaggle/커뮤니티 데이터

새 수집기는 네트워크나 파일 다운로드 전에
`mlb_decision_model.sources.require_approved_source()`를 호출해야 한다.
등록되지 않은 출처는 자동으로 거부한다.

## 데이터 계보와 누수 방지

가공 데이터에는 최소한 `source_id`, `source_url`, `retrieved_at`,
`source_version`, `as_of`를 기록한다. 경기 예측 피처는 반드시
`as_of < first_pitch`를 만족해야 한다. 경기 후 확정된 결과나 미래 시점의
선수 상태가 입력에 포함되면 해당 행을 학습에서 제외한다.

배당 원본과 사용자가 제공한 캡처는 `data/private/` 아래에만 저장하며 Git에
커밋하거나 재배포하지 않는다.

## 실시간 배당 처리

KSPO 공공 API 데이터셋 `15107776`은 경기 결과 확정 가능성 때문에 경기 종료
14일 이후의 결과만 제공한다. 응답 필드는 경기일시, 홈/원정팀, 리그, 경기장,
종목, 결과, 상품명이며 배당은 포함하지 않는다. 따라서 이 API를 실시간 배당
출처로 오인하지 않는다. 당일 국내 프로토 배당은 사용자가 제공한 스냅샷과
확인 입력값을 `user_input` 출처로 기록한다.
