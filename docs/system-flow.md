# 코인 자동거래 시스템 - 전체 프로세스 흐름

## 시스템 아키텍처 개요

```
┌──────────────────────────────────────────────────────────────────────┐
│                      _build_system()                                 │
│  Settings → 컴포넌트 조립                                             │
│                                                                      │
│  ┌──────────┐ ┌──────────┐ ┌────────────┐ ┌──────────────────────┐  │
│  │ Exchange  │ │  Broker  │ │  Strategy  │ │   LLM Advisory       │  │
│  │ Adapter   │ │ Paper/   │ │ Conserva-  │ │ OAuth / gpt-5.2      │  │
│  │ (Upbit)   │ │ Live     │ │ tive       │ │ (주도적 의사결정자)   │  │
│  └──────────┘ └──────────┘ └────────────┘ └──────────────────────┘  │
│  ┌──────────┐ ┌──────────┐ ┌────────────┐ ┌──────────────────────┐  │
│  │  Risk    │ │ Execution│ │  Kill      │ │  Anomaly             │  │
│  │ Manager  │ │ Engine   │ │  Switch    │ │  Monitor             │  │
│  └──────────┘ └──────────┘ └────────────┘ └──────────────────────┘  │
│  ┌──────────┐ ┌──────────┐ ┌────────────┐                          │
│  │  State   │ │ Idempt.  │ │   Slack    │                          │
│  │  Store   │ │ Manager  │ │  Notifier  │                          │
│  └──────────┘ └──────────┘ └────────────┘                          │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 1. 시스템 시작 단계

### 1.1 설정 로드

```
Settings.load_safe()
├── .env 파일 읽기 (pydantic-settings)
├── 환경변수 오버라이드
├── 기본값 적용 (trading_mode=paper, exchange=upbit)
└── 검증: 모드, 심볼, 리스크 파라미터
```

### 1.2 Live 모드 진입 시 2단계 확인

```
LIVE 모드 요청
├── RUN/live_mode_token.txt 파일 존재 확인 (1단계: 파일 기반 토큰)
├── .env에 TRADING_MODE=live 확인 (2단계)
└── 실패 시 → 즉시 종료
```

웹 대시보드에서 `TRADING_MODE=live`가 설정된 상태로 컨테이너가 재시작되면 실거래가 자동으로 시작됩니다.

### 1.3 컴포넌트 조립 (`_build_system`)

```
_build_system(settings)
│
├── 로깅 초기화 (structlog JSON / JSONL 파일 분리)
├── StateStore 생성 (SQLite WAL 모드)
├── KillSwitch 생성 (파일 + SIGTERM/SIGINT 핸들러)
├── RiskManager 생성 (불변 RiskLimits 주입)
├── AnomalyMonitor 생성 (가격/API/잔고 감시)
├── IdempotencyManager 생성 (중복 주문 방지, DB 연동)
│
├── Exchange Adapter 분기
│   ├── LIVE → KeyManager.decrypt_keys() → UpbitAdapter(인증키)
│   └── PAPER → UpbitAdapter(빈 키, 공개 API만)
│
├── Broker 분기
│   ├── LIVE → LiveBroker (배치 ticker, 스테일 캐시 폴백, 순입금 계산)
│   └── PAPER → PaperBroker (시뮬레이션, 수수료 0.05%)
│
├── ExecutionEngine 생성 (Broker + Risk + Idempotency + Store + KillSwitch)
├── ConservativeStrategy 생성
│
├── LLM Advisory 분기 (llm_enabled=true 일 때만)
│   └── oauth 모드 → LLMAdvisory(auth_file, model=gpt-5.2)
│
└── Slack Notifier (webhook_url 설정 시)
```

---

## 2. 메인 트레이딩 루프

```
_main_loop(settings)
│
└── while True:
    ├── 동적 심볼 유니버스 갱신(기본 1시간)
    ├── 활성 심볼 배치 ticker 조회
    ├── for symbol in active_symbols:
    │   └── _run_tick(components, symbol)
    │
    └── sleep(market_data_interval_sec)  # 기본 60초
```

---

## 3. 단일 Tick 처리 (`_run_tick`)

각 심볼에 대해 한 번의 거래 판단 사이클을 수행합니다.

```
_run_tick(components, "BTC/KRW")
│
│  ── 사전 검사 ──────────────────────────────────────
│
├── [킬 스위치 확인]
│   └── 활성화 상태 → 즉시 return (거래 중단)
│
│  ── 시장 데이터 수집 ───────────────────────────────
│
├── [1] 시세 데이터 요청
│   ├── exchange_adapter.get_ticker(symbol)
│   │   (알려진 거래 심볼은 배치 조회, 기타 코인은 개별 조회)
│   ├── 성공 → anomaly_monitor.record_api_success()
│   └── 실패 →
│       ├── 429(Too Many Requests)면 실패 누적 없이 틱 스킵
│       ├── 그 외 오류: anomaly_monitor.record_api_failure()
│       ├── 연속 실패 임계값(5회) 도달 시:
│       │   ├── safety_event 저장
│       │   ├── kill_switch 활성화
│       │   └── Slack 알림 (critical)
│       └── return
│
├── [가격 유효성 검증]
│   └── trade_price ≤ 0 → return
│
├── MarketData 객체 생성
│   └── exchange, symbol, OHLCV, bid/ask, timestamp
│
│  ── 안전 검사 ──────────────────────────────────────
│
├── [2] 가격 이상 감지
│   ├── anomaly_monitor.check_price_anomaly(md)
│   ├── 직전 가격 대비 10%↑ 변동 시:
│   │   ├── safety_event 저장
│   │   └── Slack 알림 (high)
│   └── (거래는 계속 진행)
│
├── [3] 서킷 브레이커
│   ├── 전일 종가 대비 변동률 계산
│   ├── risk_manager.check_circuit_breaker(변동률)
│   ├── 15%↑ 변동 시:
│   │   ├── Slack 알림 (critical)
│   │   └── return (이번 틱 건너뜀)
│   └── (정상 → 계속)
│
│  ── 전략 시그널 생성 (참고 데이터) ─────────────────
│
├── [4] 캔들 데이터 업데이트
│   ├── 초기 또는 만료 시점에만 get_candles(symbol, "minutes/60", count=200)
│   └── strategy.update_candles(symbol, candles)
│      (기본 갱신 주기: CANDLE_REFRESH_INTERVAL_SEC=3600)
│
├── [5] 전략 시그널 생성 (LLM 참고용)
│   └── strategy.on_tick(md) → signals: [Signal, ...]
│       │
│       │  ConservativeStrategy (참고 시그널만 생성):
│       │  ┌─────────────────────────────────────────┐
│       │  │ BUY 조건 (모두 충족 시):                  │
│       │  │  ① Fast EMA > Slow EMA (상승 추세)       │
│       │  │  ② 현재가 > Slow EMA (추세 확인)         │
│       │  │  ③ ATR/가격 비율 < 3% (저변동성)         │
│       │  │  ④ RSI 30~65 (과매수 아님)               │
│       │  │  ⑤ 거래량 > 평균의 50%                   │
│       │  ├─────────────────────────────────────────┤
│       │  │ SELL 조건 (하나라도 충족 시):              │
│       │  │  ① Fast EMA < Slow EMA (추세 반전)       │
│       │  │  ② RSI > 75 (과매수)                     │
│       │  ├─────────────────────────────────────────┤
│       │  │ 나머지 → HOLD                            │
│       │  └─────────────────────────────────────────┘
│       │  ※ 이 시그널은 LLM에 전달되는 참고 데이터.
│       │    LLM이 활성화된 경우 단독으로 주문을 내지 않음.
│
│  ── 상태 수집 ──────────────────────────────────────
│
├── [6] 잔고/포지션 조회
│   ├── broker.fetch_balances() → BalanceSnapshot
│   ├── broker.fetch_positions() → [Position, ...]
│   ├── balance snapshot DB 저장
│   └── state 딕셔너리 구성:
│       { total_balance, position_count, today_pnl, market_price }
│
│  ── LLM 의사결정 (주도적 결정자) ───────────────────
│
├── [7] LLM Advisory
│   │
│   ├── [스킵 조건 확인]
│   │   ├── 가격 변화 < 1% AND 마지막 LLM 호출 < 30분 전
│   │   ├── ※ 예외: 소프트 손절(-5%~-10%) 또는 익절(+10%+) 구간이면 강제 호출
│   │   └── → advice = None (LLM 스킵, 로그 미기록)
│   │
│   ├── [LLM 호출 조건 충족 시]
│   │   ├── 입력 데이터 구성:
│   │   │   ├── 시장 데이터, 기술적 지표 (EMA, RSI, ATR)
│   │   │   ├── 보유 포지션, 잔고
│   │   │   ├── 최근 주문, 최근 캔들
│   │   │   └── 이전 5개 결정
│   │   ├── llm_advisory.get_advice(symbol, data)
│   │   │   └── OAuth 모드:
│   │   │       ├── get_reusable_auth() → 토큰 로드/갱신
│   │   │       └── query_codex() → ChatGPT Codex API (SSE 스트리밍)
│   │   ├── 응답 → JSON 파싱 → LLMAdvice
│   │   │   ├── action: HOLD | BUY_CONSIDER | SELL_CONSIDER
│   │   │   ├── confidence: 최소 0.65 이상이어야 실행
│   │   │   ├── reasoning: 한국어
│   │   │   └── risk_notes: 한국어
│   │   ├── 결과 → log_event("decision", ...) 저장 (프롬프트 포함)
│   │   └── 실패 시 → warning 로그, advice = None (fail-safe)
│   │
│   └── advice 반환 (None 또는 LLMAdvice)
│
│  ── 포지션 보호 ─────────────────────────────────────
│
├── [8] 포지션 보호 검사 (보유 포지션 있을 때)
│   ├── 하드 손절 (-10% 이상 손실):
│   │   └── 시장가(MARKET) 자동 매도, LLM 우회
│   ├── 소프트 손절 (-5%~-10% 손실):
│   │   └── LLM 판단 (SELL_CONSIDER = 매도, HOLD = 유지)
│   ├── 익절 (+10% 이상 수익):
│   │   └── LLM 판단 (명시적 HOLD만 유지, 나머지 매도)
│   ├── 트레일링 스탑:
│   │   └── 최고점 대비 trailing_stop_pct 하락 시 자동 매도
│   └── ※ 보호 매도 발생 시 protection_sold 플래그 → 이후 combined 판단 스킵 (이중 매도 방지)
│
│  ── 액션 결정 ──────────────────────────────────────
│
├── [9] _resolve_action
│   ├── LLM 활성화 + advice 있음 → LLM action 따름
│   ├── LLM 활성화 + advice None (스킵/오류) → HOLD
│   └── LLM 비활성화 → 전략 시그널 따름
│
│  ── 주문 실행 ──────────────────────────────────────
│
└── [10] 액션별 주문 처리
    │
    ├── HOLD → skip (아무 것도 안 함)
    │
    ├── BUY_CONSIDER (confidence >= 0.65) →
    │   ├── 주문 금액 = 총잔고 × max_position_size_pct
    │   ├── OrderIntent 생성
    │   └── engine.execute(intent, state)
    │
    └── SELL_CONSIDER (confidence >= 0.65) →
        ├── 해당 심볼 보유 포지션 순회
        ├── OrderIntent 생성 (전량 매도)
        └── engine.execute(intent, state)
```

---

## 3.1 동적 심볼 유니버스 선발

```
_refresh_dynamic_symbols()
│
├── KRW 마켓 목록 조회 (/market/all)
├── 배치 ticker 조회 (/ticker?markets=...)
├── 배치 orderbook 조회 (/orderbook?markets=...) → 실제 스프레드 계산
├── 1차 필터: 거래대금 최소치
├── 후보 K 계산: min(20, max(12, 3*top_n))
├── 하드 필터:
│   ├── spread_bps <= DYNAMIC_SYMBOL_MAX_SPREAD_BPS
│   ├── abs(change_24h_pct) <= DYNAMIC_SYMBOL_MAX_ABS_CHANGE_24H_PCT
│   └── intraday_range_pct <= DYNAMIC_SYMBOL_MAX_INTRADAY_RANGE_PCT
├── LLM 최종 선발(top_n)
│   ├── candidates: 신규 후보 지표
│   └── active_symbols: 현재 활성 심볼 지표(강제유지 제외)
├── 강제유지 결합:
│   ├── ALWAYS_KEEP_SYMBOLS
│   └── 현재 보유 심볼
└── 최종 심볼 cap: DYNAMIC_SYMBOL_MAX_SYMBOLS(기본 10)
```

---

## 4. 주문 실행 파이프라인 (`ExecutionEngine.execute`)

모든 주문은 반드시 이 파이프라인을 통과합니다.

```
engine.execute(intent, state)
│
├── [Gate 1] 킬 스위치 확인
│   └── 활성 → return None (주문 차단)
│
├── [Gate 2] 멱등성 확인
│   ├── intent_id로 client_order_id 생성
│   ├── 메모리 캐시 확인 (in-memory set)
│   ├── DB 확인 (orders 테이블)
│   └── 중복 → return None (주문 건너뜀)
│
├── [Gate 3] 리스크 검증 (RiskManager.validate)
│   │
│   │  리스크 체크:
│   │
│   ├── ① 포지션 크기: 총자산의 max_position_size_pct 초과 시 거부
│   ├── ② 일일 손실 한도: 초과 시 거부
│   ├── ③ 초당 주문 제한: BUY에만 적용 (SELL은 항상 통과)
│   ├── ④ 일일 주문 제한: BUY에만 적용 (SELL은 항상 통과)
│   ├── ⑤ 선물/파생상품: 비활성화 시 거부
│   └── ⑥ 슬리피지: 50bps 초과 시 거부
│   │
│   ├── → RiskDecisionRecord 생성 (APPROVED/REJECTED)
│   └── → DB 저장 (decisions_log)
│
├── REJECTED → return None
│
├── [실행] broker.place(intent, client_order_id)
│   ├── PAPER: PaperBroker
│   │   ├── 잔고 확인 (KRW 또는 코인)
│   │   ├── 수수료 적용 (0.05%)
│   │   ├── Fill 생성 → 메모리 포지션/잔고 업데이트
│   │   └── Order(status=FILLED) 반환
│   │
│   └── LIVE: LiveBroker
│       ├── UpbitAdapter.place_order() → 실제 API 호출
│       │   (손절은 MARKET 주문, 일반 매매는 LIMIT 주문)
│       └── Order(status=PENDING or FILLED) 반환
│
├── [영속화]
│   ├── store.save_order(order)
│   └── idempotency.mark_processed(client_order_id)
│
└── return order
```

---

## 5. 안전 체계 (Safety Layer)

```
┌────────────────────────────────────────────────────────────────┐
│                    다중 방어 계층                               │
│                                                                │
│  Layer 1: Kill Switch                                          │
│  ├── 파일 기반 (RUN/kill_switch)                                │
│  ├── SIGTERM/SIGINT 시그널 핸들링                               │
│  ├── Web API: POST /api/kill-switch/activate                   │
│  └── API 연속 실패 5회 시 자동 활성화                            │
│                                                                │
│  Layer 2: 포지션 보호                                           │
│  ├── 하드 손절 (-10%): 시장가 자동 매도, LLM 우회               │
│  ├── 소프트 손절 (-5%~-10%): LLM 판단                          │
│  ├── 익절 (+10%+): LLM 판단                                    │
│  └── 트레일링 스탑: 최고점 대비 자동 매도                        │
│                                                                │
│  Layer 3: Circuit Breaker                                      │
│  └── 전일 종가 대비 15% 이상 변동 → 해당 틱 거래 중단            │
│                                                                │
│  Layer 4: Anomaly Monitor                                      │
│  ├── API 연속 실패 감지 → kill switch 자동 활성화               │
│  ├── 가격 급변 감지 (10%↑) → safety_event 기록                 │
│  ├── 잔고 불일치 감지 → safety_event 기록                       │
│  └── 스프레드 이상 감지 (5%↑) → safety_event 기록              │
│                                                                │
│  Layer 5: Risk Manager                                         │
│  ├── 포지션 크기 제한                                           │
│  ├── 일일 손실 한도                                             │
│  ├── 주문 빈도 제한 (BUY에만 적용)                              │
│  ├── 슬리피지 제한 (50bps)                                      │
│  └── 선물/레버리지 차단 (기본)                                   │
│                                                                │
│  Layer 6: Idempotency                                          │
│  └── intent_id 기반 중복 주문 방지 (메모리 + DB)                │
│                                                                │
│  Layer 7: Live 모드 진입 제어                                   │
│  └── 토큰 파일 확인 (RUN/live_mode_token.txt)                   │
│                                                                │
│  Layer 8: API 키 보안                                           │
│  ├── Fernet 암호화 저장 (KeyManager)                            │
│  └── 로그 자동 마스킹 (redact_sensitive_data)                   │
└────────────────────────────────────────────────────────────────┘
```

---

## 6. LLM Advisory 흐름

LLM은 **주도적 의사결정자**입니다. LLM의 action이 실제 매매를 결정합니다.

```
┌─ OAuth 모드 (gpt-5.2) ────────────────────────┐
│                                                │
│  LLMAdvisory.get_advice(symbol, data)          │
│  │                                             │
│  ├── [스킵 조건]                               │
│  │   ├── 가격 변화 < 1%                        │
│  │   └── 마지막 호출 < 30분 전                  │
│  │   → advice = None, 로그 미기록              │
│  │                                             │
│  ├── [인증] get_reusable_auth()                │
│  │   ├── 저장된 토큰 로드 (.auth/openai-oauth) │
│  │   ├── 만료 → refresh_access_token()         │
│  │   └── AuthTokens (access, refresh, account) │
│  │                                             │
│  ├── [프롬프트 구성]                            │
│  │   ├── 시장 데이터, 기술적 지표               │
│  │   │   ├── ema200_1h (1h 200봉 ~8일 MA)     │
│  │   │   └── BTC 일봉 EMA200 트렌드 (200일)   │
│  │   ├── 보유 포지션, 잔고                      │
│  │   ├── 최근 캔들 (마지막 캔들은 합성, 볼륨 0) │
│  │   └── 이전 5개 결정                         │
│  │                                             │
│  └── [API 호출] query_codex()                  │
│      ├── POST chatgpt.com/backend-api/codex/   │
│      │   responses                             │
│      ├── model: gpt-5.2                        │
│      ├── SSE 스트리밍 수신                      │
│      │   ├── response.output_text.delta → 조립 │
│      │   └── [DONE] → 종료                     │
│      └── 조립된 텍스트 반환                     │
│                                                │
└────────────────────────────────────────────────┘

┌─ 응답 처리 ───────────────────────────────────┐
│                                                │
│  _parse_response(text)                         │
│  ├── JSON 파싱 (```json 코드펜스 처리)          │
│  └── LLMAdvice 스키마 검증:                    │
│      ├── action: HOLD | BUY_CONSIDER |         │
│      │          SELL_CONSIDER                   │
│      ├── confidence: 0.0 ~ 1.0                 │
│      │   (0.65 미만이면 실행 안 함)             │
│      ├── reasoning: 한국어, 최대 500자          │
│      ├── risk_notes: 한국어, 최대 300자         │
│      └── 검증 실패 → None (fail-safe)           │
│                                                │
│  결과: log_event("decision", ...) 저장          │
│  (프롬프트 포함, 대시보드 "P" 버튼으로 확인 가능) │
│                                                │
│  BUY_CONSIDER → 매수 실행                      │
│  SELL_CONSIDER → 매도 실행                     │
│  HOLD → 아무 것도 안 함                        │
└────────────────────────────────────────────────┘
```

---

## 7. 데이터 영속화

```
StateStore (SQLite, WAL 모드)
│
├── events            # 전체 이벤트 append-only 감사 로그
├── order_intents     # 주문 의도
├── orders            # 주문 (upsert)
├── fills             # 체결
├── positions_snapshot # 포지션 스냅샷 (시계열)
├── balances_snapshot # 잔고 스냅샷 (시계열)
├── decisions_log     # 리스크 결정 기록
└── safety_events     # 안전 이벤트 기록

로그 파일 (JSONL, 코인별 분리)
├── logs/decisions_btc_krw.jsonl   # BTC AI 결정 (HOLD 압축 적용)
├── logs/decisions_eth_krw.jsonl   # ETH AI 결정
├── logs/decisions_xrp_krw.jsonl   # XRP AI 결정
├── logs/decisions_ada_krw.jsonl   # ADA AI 결정
├── logs/decisions_sol_krw.jsonl   # SOL AI 결정
├── logs/orders.jsonl
├── logs/safety.jsonl
└── logs/general.jsonl

HOLD 압축: 연속 HOLD가 1시간 초과 시 첫 번째와 마지막만 보존
LLM 스킵: 가격 변화 < 1% + 30분 미경과 → 로그 미기록
```

---

## 8. 웹 대시보드

```
FastAPI + Jinja2 (포트 8932)
│
├── 인증 (브루트포스 방어 + CSRF + Secure 쿠키 + 24h 세션 만료)
│   ├── GET  /login                    → 로그인 페이지
│   ├── POST /api/auth/login           → 마스터 코드 인증 (IP당 5회 실패 시 5분 잠금)
│   │   └── 성공 시: 세션 쿠키(httponly) + CSRF 토큰 쿠키 발급
│   └── GET  /                         → 대시보드 (인증 필요, POST 시 CSRF 헤더 검증)
│
├── 시스템 제어
│   ├── GET  /api/status               → 시스템 상태
│   ├── POST /api/trading/start        → 거래 시작 (paper/live)
│   ├── POST /api/trading/stop         → 거래 중지
│   ├── POST /api/kill-switch/activate → Kill Switch 활성화
│   └── POST /api/kill-switch/deactivate → Kill Switch 해제
│
├── 데이터 조회
│   ├── GET  /api/balances             → 잔고
│   ├── GET  /api/positions            → 포지션 (현재가 포함)
│   ├── GET  /api/pnl                  → 손익 (초기 잔고 = 2026년 이후 순입금)
│   ├── GET  /api/orders               → 전체 주문 (10개씩 페이지네이션)
│   ├── GET  /api/orders/open          → 미체결 주문
│   ├── GET  /api/safety-events        → 안전 이벤트
│   ├── GET  /api/decisions            → 전체 AI 결정 (10개씩 페이지네이션)
│   ├── GET  /api/decisions/{symbol}   → 코인별 AI 결정
│   └── GET  /api/symbol-decisions     → AI 심볼 선발 로그
│
├── 수동 거래
│   ├── POST /api/orders/manual-buy    → 수동 매수 {symbol, amount_krw}
│   └── POST /api/orders/manual-sell   → 수동 매도 {symbol}
│
└── 클라이언트 자동 갱신
    ├── 15초: 상태, 잔고, 포지션, 손익, 미체결 주문
    └── 30초: 전체 주문, 안전 이벤트, AI 결정

대시보드 주요 기능:
├── 포지션 테이블: 심볼, 수량, 평균진입가, 현재가, 평가금액,
│                 미실현손익, 미실현손익률, 전량매도 버튼
├── AI 판단 로그: 코인별 필터, reasoning 모달 팝업
├── AI 심볼 판단 로그: 선발 이력, LLM 사유 모달, 프롬프트(P)
└── 프롬프트 확인: "P" 버튼으로 LLM에 전달된 프롬프트 조회
```

---

## 9. 파일 구조

```
coin_trader/
├── main.py                     # 메인 루프, _run_tick, _resolve_action, LLM 연동
├── config/settings.py          # 설정 (pydantic-settings)
├── core/
│   ├── contracts.py            # 인터페이스 정의
│   └── models.py               # 도메인 모델 (Position, Order 등)
├── exchange/
│   ├── base.py                 # httpx + tenacity + 레이트 리밋
│   └── upbit.py                # 업비트 어댑터 (배치 ticker/orderbook, 입출금 조회)
├── broker/
│   ├── paper.py                # 모의투자 브로커
│   └── live.py                 # 실거래 브로커 (배치 ticker, 스테일 캐시 폴백, 순입금)
├── strategy/
│   └── conservative.py         # EMA+RSI+ATR (참고 시그널 전용)
├── risk/
│   ├── limits.py               # 불변 리스크 상수
│   └── manager.py              # 리스크 게이트 (포지션 수 제한 없음, SELL 우회)
├── execution/
│   ├── engine.py               # 주문 실행 파이프라인
│   └── idempotency.py          # 중복 주문 방지
├── safety/
│   ├── kill_switch.py          # 긴급 정지
│   └── monitor.py              # 이상징후 감지
├── llm/
│   ├── advisory.py             # LLM 주도 의사결정 (한국어 출력, 프롬프트 저장)
│   ├── oauth_openai.py         # OAuth PKCE 인증
│   └── codex_client.py         # ChatGPT Codex SSE 클라이언트
├── state/
│   └── store.py                # SQLite WAL 상태 저장소
├── security/
│   └── key_manager.py          # API 키 Fernet 암호화
├── logging/
│   ├── logger.py               # structlog + JSONL 타입별 분리 + HOLD 압축
│   └── redaction.py            # 민감정보 마스킹
├── notify/
│   └── slack.py                # Slack 알림
└── web/
    ├── api.py                  # FastAPI + 인증 미들웨어(브루트포스/CSRF) + 실거래 자동 시작
    └── templates/
        ├── dashboard.html      # 대시보드 (페이지네이션, 모달, 인증)
        └── login.html          # 로그인 페이지
```
