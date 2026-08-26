# Claude 3차 검토 결과

실행: `PYTHONPATH=src python -m unittest discover -s tests -v` → **15개 전부 통과**
(dashboard.html까지 디스크에서 확보해서 풀세트로 재구성 후 진행)

---

## 0. 먼저: 지난 리뷰 4건 재확인 — 전부 미수정

새 기능(kspo/snapshot/dashboard) 작업하시느라 지난 P0/P1 지적사항은 손 안 대신 것 같아요.
코드로 다시 확인했습니다.

| 항목 | 위치 | 상태 |
|---|---|---|
| zip-slip 백슬래시 우회 | retrosheet.py:52,74 | **그대로** — 재실행해서 재현 확인 |
| data_sources.json이 실제로 로드 안 됨 | sources.py | **그대로** — `json.load` 호출 어디에도 없음 |
| `_pitcher_score` 내부 고정 가중치 | features.py:62-64 | **그대로** |
| `source_version` 하드코딩 | retrosheet.py:59 | **그대로** `"bundle-through-2025"` |

우선순위 판단하신 거면 상관없는데, **zip-slip은 실제 Retrosheet ZIP을 열기 전에 반드시 고치셔야 하는 항목**이라 다시 짚어둡니다. 나머지는 실행에 지장은 없지만 언젠가는 갚아야 할 빚이에요.

---

## 새로 추가된 부분 검토

### ✅ 잘 만드신 부분 — 실행해서 확인 완료

`dashboard.analyze_payload`를 저희가 예전에 실제로 다뤘던 4개 픽(SEA/SD/CHC/SF-CIN)으로 직접 돌려봤습니다.

```
생존확률 랭킹: SEA 승+SF/CIN 언더(27.5%) > SEA 승+CHC 승(25.3%) > ... > SD 승+CHC 승(20.2%)
```

**저희가 지난 대화에서 손으로/여러 독립 구현으로 검증했던 숫자와 정확히 일치**합니다. 이번이 벌써 세 번째 독립 구현(제 몬테카를로, GPT의 decision.py 1차, 지금 이 dashboard 버전)이 같은 답을 내는 거라 로직 신뢰도가 꽤 높습니다.

같은 경기 마켓 2개(`SEA-PHI 승` + `SEA-PHI 오버`)를 넣었을 때도 실제로 그 조합만 정확히 제외되는 것 확인했습니다 — `event_id` 기반 필터링이 의도대로 작동합니다.

`snapshot.py`의 이미지 검증(정규식 fullmatch, base64 strict 디코딩, 8MB 상한)과 `dashboard.html`의 동적 콘텐츠 이스케이프(`esc()` 일관 적용, innerHTML 사용처마다 확인)도 꼼꼼하게 잘 돼 있습니다.

### P1 — 새로 발견한 것

**`snapshot.validate_picks`가 "최소 2개" 강제라서 `/api/analyze` 단일 경기 판정이 불가능합니다.**

```python
>>> analyze_payload({"picks":[{"event_id":"SEA-PHI","name":"SEA 승","probability":0.55,"odds":1.69}]})
ValueError: at least two picks are required
```

`predict.py` CLI는 "한 경기의 양방향 배당 판정"을 지원하는데, 대시보드는 `analyze_payload`가 `snapshot.py`의 `validate_picks`(스냅샷 저장용으로 만든 함수, 최소 2개 강제가 맞는 함수)를 그대로 재사용하면서 이 제약이 대시보드 전체에 새어 들어갔습니다. 스냅샷 저장(`/api/snapshots`, 조합이 목적)엔 맞는 제약이지만, 단순 분석(`/api/analyze`, 경기 1개만 확인하고 싶을 수도 있음)에는 과한 제약이에요.

**수정 제안**: `validate_picks`에 `minimum: int = 2` 매개변수를 추가하고, `analyze_payload`는 `minimum=1`로 호출하세요.

### P2 — 새로 발견한 것 (경미)

1. **`dashboard.html`이 항상 `/api/snapshots`만 호출합니다** (53행). `/api/analyze`(저장 안 함)는 백엔드에 존재하는데 프런트엔드에서 쓸 방법이 없어서, 그냥 값 좀 바꿔보며 테스트하는 것도 전부 `data/private/odds_snapshots.jsonl`에 영구 기록됩니다. 나중에 진짜 스냅샷과 테스트성 입력이 섞여서 로그가 지저분해질 수 있어요. "저장 없이 미리보기" 버튼을 하나 더 두거나, 최소한 "이 결과를 저장할까요?" 확인 단계를 넣는 걸 추천합니다.
2. **`kspo.py`의 엔드포인트 URL·응답 필드명이 전부 추정치입니다** — `B551014/SRVC_OD_API_TB_SOSFO_MATCH_MGMT/...`, `match_ymd`, `hteam_han_nm` 등. `test_kspo_response_parser`는 이 추정 스키마에 맞춰 만든 가짜 응답으로만 테스트하기 때문에, 테스트가 통과해도 "코드가 스스로 짠 스펙대로 동작한다"만 증명하지 "실제 정부 API와 맞다"는 증명이 안 됩니다. 활용신청 키 받으시면 실제 응답 1건을 꼭 저에게 보여주세요 — 필드명이 다르면 `parse_results`는 에러 없이 그냥 빈 문자열들을 조용히 채워 넣습니다(`item.get(key, "")` 기본값 때문에). 스키마가 틀렸는데 티가 안 나는 게 제일 위험한 실패 모드예요.
3. `kspo.fetch_results`도 retrosheet와 같은 구조적 허점이 있습니다 — `require_approved_source(..., automated=True)`만 호출하고 `bulk=True`는 호출부가 스스로 안 넘기면 절대 안 걸립니다. 즉 이 함수를 날짜 리스트로 반복 호출하는 코드를 나중에 누가 추가해도 정책이 이를 막지 못합니다. (retrosheet.py 리뷰 때 지적한 것과 동일한 패턴이라 간단히만 남깁니다.)
4. 대시보드는 픽 개수 상한이 없어서("경기 추가" 버튼 무제한), 픽을 많이 넣고 4폴을 돌리면 조합 수 × 5만 시뮬레이션이 급격히 늘어 응답이 느려질 수 있습니다. 개인용 로컬 도구라 심각하진 않지만 참고하세요.

---

## 요약

| 심각도 | 항목 | 상태 |
|---|---|---|
| P0 | zip-slip (재확인) | **여전히 미수정** |
| P0 | data_sources.json 이중소스 (재확인) | **여전히 미수정** |
| P1 | `_pitcher_score` 고정 가중치 (재확인) | **여전히 미수정** |
| P1 | `validate_picks` 최소 2개 강제가 단일판정 막음 | 신규 발견 |
| P1 | `source_version` 하드코딩 (재확인) | **여전히 미수정** |
| P2 | 대시보드가 항상 스냅샷을 영구 저장 | 신규 발견 |
| P2 | kspo.py 스키마 전부 미검증 (실제 키로 확인 필요) | 신규 발견, 원래도 알려진 리스크 |
| P2 | kspo.py도 bulk 정책이 구조적으로 강제 안 됨 | 신규 발견 |
| ✅ | 조합 로직/생존확률/EV 분리/동일경기 제외 | 실행 검증 완료, 정상 작동 |
| ✅ | 프런트엔드 XSS 방어 | 확인 완료, 문제 없음 |

새 기능들은 실제로 잘 작동하는 걸 확인했으니, 다음 라운드에서는 **새 기능 추가보다 지난 P0 2건(zip-slip, 이중소스)부터 정리**하시는 걸 권합니다 — 계속 뒤로 밀리고 있어서요. 그다음이 Retrosheet ETL이고요.
