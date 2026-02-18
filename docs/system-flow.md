# 코인 자동거래 시스템 - 전체 프로세스 흐름

## 시스템 아키텍처 개요

```
┌─────────────────────────────────────────────────────────────────────┐
│                          CLI (click)                                │
│   trader run [--mode paper|live] [--once]                           │
│   trader kill [--close-positions]                                   │
│   trader selftest                                                   │
│   trader web [--host] [--port]                                      │
│   trader encrypt-keys --exchange upbit                              │
└──────────────┬──────────────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                      _build_system()                                 │
│  Settings → 컴포넌트 조립 (DI Container)                              │
│                                                                      │
│  ┌──────────┐ ┌──────────┐ ┌────────────┐ ┌──────────────────────┐  │
│  │ Exchange  │ │  Broker  │ │  Strategy  │ │   LLM Advisor        │  │
│  │ Adapter   │ │ Paper/   │ │ Conserva-  │ │ API Key / OAuth      │  │
│  │ (Upbit)   │ │ Live     │ │ tive       │ │ (선택)               │  │
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
├── 'I_UNDERSTAND_LIVE_TRADING' 확인 입력 (2단계: 사용자 확인)
└── 실패 시 → 즉시 종료
```

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
│   ├── LIVE → LiveBroker (실제 주문)
│   └── PAPER → PaperBroker (시뮬레이션, 수수료 0.05%)
│
├── ExecutionEngine 생성 (Broker + Risk + Idempotency + Store + KillSwitch)
├── ConservativeStrategy 생성
│
├── LLM Advisor 분기 (llm_enabled=true 일 때만)
│   ├── api_key 모드 → LLMAdvisor(api_key, provider, model)
│   └── oauth 모드 → LLMAdvisor.create_oauth(auth_file, model)
│
└── Slack Notifier (webhook_url 설정 시)
```

---

## 2. 메인 트레이딩 루프

```
_main_loop(settings)
│
└── while True:
    ├── for symbol in trading_symbols:  # 예: ["BTC/KRW", "ETH/KRW"]
    │   └── _run_tick(components, symbol)
    │
    ├── once=True → break (단일 실행 모드)
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
│   ├── 성공 → anomaly_monitor.record_api_success()
│   └── 실패 →
│       ├── anomaly_monitor.record_api_failure()
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
│  ── 전략 실행 ──────────────────────────────────────
│
├── [4] 캔들 데이터 업데이트
│   ├── exchange_adapter.get_candles(symbol, "minutes/60", count=200)
│   └── strategy.update_candles(symbol, candles)
│
├── [5] 전략 시그널 생성
│   └── strategy.on_tick(md) → signals: [Signal, ...]
│       │
│       │  ConservativeStrategy 판단 로직:
│       │  ┌─────────────────────────────────────────┐
│       │  │ BUY 조건 (모두 충족 시):                  │
│       │  │  ① Fast EMA > Slow EMA (상승 추세)       │
│       │  │  ② 현재가 > Slow EMA (추세 확인)         │
│       │  │  ③ ATR/가격 비율 < 3% (저변동성)         │
│       │  │  ④ RSI 30~65 (과매수 아님)               │
│       │  │  ⑤ 거래량 > 평균의 50%                   │
│       │  │  → confidence ≥ 0.6 시 BUY 시그널        │
│       │  ├─────────────────────────────────────────┤
│       │  │ SELL 조건 (하나라도 충족 시):              │
│       │  │  ① Fast EMA < Slow EMA (추세 반전)       │
│       │  │  ② RSI > 75 (과매수)                     │
│       │  ├─────────────────────────────────────────┤
│       │  │ 나머지 → HOLD                            │
│       │  └─────────────────────────────────────────┘
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
│  ── LLM 자문 (선택, 실행에 영향 없음) ──────────────
│
├── [7] LLM Advisory (llm_advisor 존재 시)
│   ├── market_summary + strategy_signals 구성
│   ├── llm_advisor.get_advice(symbol, summary, signals)
│   │   ├── API Key 모드: OpenAI/Anthropic REST API
│   │   └── OAuth 모드:
│   │       ├── get_reusable_auth() → 토큰 로드/갱신/브라우저 로그인
│   │       └── query_codex() → ChatGPT Codex API (SSE 스트리밍)
│   ├── 응답 → JSON 파싱 → LLMAdvice (action, confidence, reasoning, risk_notes)
│   ├── 결과 → log_event("decision", {type: "llm_advice", ...})
│   └── 실패 시 → warning 로그, 무시 (fail-safe)
│
│  ── 주문 실행 ──────────────────────────────────────
│
└── [8] 시그널별 주문 처리
    │
    ├── HOLD → skip (아무 것도 안 함)
    │
    ├── BUY →
    │   ├── 주문 금액 = 총잔고 × max_position_size_pct(10%)
    │   ├── OrderIntent 생성 (LIMIT 주문)
    │   └── engine.execute(intent, state) → 아래 참조
    │
    └── SELL →
        ├── 해당 심볼 보유 포지션 순회
        ├── OrderIntent 생성 (전량 매도, LIMIT)
        └── engine.execute(intent, state) → 아래 참조
```

---

## 4. 주문 실행 파이프라인 (`ExecutionEngine.execute`)

모든 주문은 반드시 이 파이프라인을 통과합니다. 순서가 바뀌거나 건너뛸 수 없습니다.

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
│   │  8단계 리스크 체크 (모두 통과해야 승인):
│   │
│   ├── ① 시장가 주문 정책: LIMIT만 허용 (기본)
│   ├── ② 포지션 크기: 총자산의 10% 초과 시 거부
│   ├── ③ 최대 포지션 수: 5개 초과 시 거부
│   ├── ④ 일일 손실 한도: 5% 초과 시 거부
│   ├── ⑤ 초당 주문 제한: 1건/초 초과 시 거부
│   ├── ⑥ 일일 주문 제한: 100건/일 초과 시 거부
│   ├── ⑦ 선물/파생상품: 비활성화 시 거부
│   └── ⑧ 슬리피지: 50bps 초과 시 거부
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
│  ├── CLI: trader kill                                          │
│  ├── Web API: POST /api/kill-switch/activate                   │
│  └── API 연속 실패 5회 시 자동 활성화                            │
│                                                                │
│  Layer 2: Circuit Breaker                                      │
│  └── 전일 종가 대비 15% 이상 변동 → 해당 틱 거래 중단            │
│                                                                │
│  Layer 3: Anomaly Monitor                                      │
│  ├── API 연속 실패 감지 → kill switch 자동 활성화               │
│  ├── 가격 급변 감지 (10%↑) → safety_event 기록                 │
│  ├── 잔고 불일치 감지 → safety_event 기록                       │
│  └── 스프레드 이상 감지 (5%↑) → safety_event 기록              │
│                                                                │
│  Layer 4: Risk Manager (우회 불가)                              │
│  ├── 포지션 크기 제한 (10%)                                     │
│  ├── 최대 포지션 수 (5개)                                       │
│  ├── 일일 손실 한도 (5%)                                        │
│  ├── 주문 빈도 제한 (1건/초, 100건/일)                          │
│  ├── 시장가 주문 차단 (기본)                                     │
│  ├── 슬리피지 제한 (50bps)                                      │
│  └── 선물/레버리지 차단 (기본)                                   │
│                                                                │
│  Layer 5: Idempotency                                          │
│  └── intent_id 기반 중복 주문 방지 (메모리 + DB)                │
│                                                                │
│  Layer 6: Live 모드 진입 제어                                   │
│  ├── 토큰 파일 확인                                             │
│  └── 사용자 확인 문구 입력                                      │
│                                                                │
│  Layer 7: API 키 보안                                           │
│  ├── Fernet 암호화 저장 (KeyManager)                            │
│  └── 로그 자동 마스킹 (redact_sensitive_data)                   │
└────────────────────────────────────────────────────────────────┘
```

---

## 6. LLM Advisory 흐름

LLM은 자문만 제공하며 주문 실행에 직접 관여하지 않습니다.

```
┌─ API Key 모드 ────────────────────────────────┐
│                                                │
│  LLMAdvisor._call_openai()                     │
│  ├── POST https://api.openai.com/v1/...        │
│  ├── Authorization: Bearer {api_key}           │
│  └── JSON 응답 → _parse_response()             │
│                                                │
│  LLMAdvisor._call_anthropic()                  │
│  ├── POST https://api.anthropic.com/v1/...     │
│  ├── x-api-key: {api_key}                      │
│  └── JSON 응답 → _parse_response()             │
└────────────────────────────────────────────────┘

┌─ OAuth 모드 ──────────────────────────────────┐
│                                                │
│  LLMAdvisor._call_oauth_codex()                │
│  │                                             │
│  ├── [인증] get_reusable_auth()                │
│  │   ├── 저장된 토큰 로드 (.auth/openai-oauth) │
│  │   ├── 만료 → refresh_access_token()         │
│  │   ├── 없음 → 브라우저 OAuth PKCE 로그인     │
│  │   │   ├── PKCE verifier + challenge 생성    │
│  │   │   ├── localhost:1455 콜백 서버 시작      │
│  │   │   ├── auth.openai.com 인가 URL 열기     │
│  │   │   ├── 사용자 로그인 → 콜백으로 code 수신 │
│  │   │   └── code → token exchange → 저장      │
│  │   └── AuthTokens (access, refresh, account) │
│  │                                             │
│  ├── [모델 검증] normalize_model()              │
│  │   └── 허용: gpt-5.2-codex, gpt-5.1-codex...│
│  │                                             │
│  └── [API 호출] query_codex()                  │
│      ├── POST chatgpt.com/backend-api/codex/   │
│      │   responses                             │
│      ├── Headers:                              │
│      │   ├── Authorization: Bearer {token}     │
│      │   ├── chatgpt-account-id: {id}          │
│      │   ├── OpenAI-Beta: responses=experiment │
│      │   └── originator: opencode              │
│      ├── Body: model, stream, instructions,    │
│      │   reasoning, input                      │
│      ├── SSE 스트리밍 수신                      │
│      │   ├── response.output_text.delta → 조립 │
│      │   └── [DONE] → 종료                     │
│      └── 조립된 텍스트 반환                     │
│                                                │
└────────────────────────────────────────────────┘

┌─ 공통 응답 처리 ──────────────────────────────┐
│                                                │
│  _parse_response(text)                         │
│  ├── JSON 파싱 (```json 코드펜스 처리)          │
│  └── LLMAdvice.from_mapping() 스키마 검증:      │
│      ├── action: HOLD | BUY_CONSIDER |         │
│      │          SELL_CONSIDER                   │
│      ├── confidence: 0.0 ~ 1.0                 │
│      ├── reasoning: 최대 500자                  │
│      ├── risk_notes: 최대 300자                 │
│      └── 검증 실패 → None (fail-safe)           │
│                                                │
│  결과: log_event("decision", ...) 로 기록만     │
│  → 주문 로직에 영향 없음                        │
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

로그 파일 (JSONL 분리)
├── logs/decisions.jsonl
├── logs/orders.jsonl
├── logs/fills.jsonl
├── logs/balances.jsonl
├── logs/safety.jsonl
└── logs/general.jsonl
```

---

## 8. 웹 대시보드

```
trader web --port 8000
│
├── FastAPI + Jinja2 HTML 템플릿
├── GET  /                      → 대시보드 UI (한국어)
│
├── GET  /api/status            → 시스템 상태 (모드, 킬스위치, 가동시간)
├── GET  /api/balances          → 보유 자산
├── GET  /api/positions         → 보유 포지션
├── GET  /api/pnl               → 손익 현황
├── GET  /api/orders            → 전체 주문 내역
├── GET  /api/orders/open       → 미체결 주문
├── GET  /api/safety-events     → 안전 이벤트
├── GET  /api/decisions         → 리스크 결정 로그
│
├── POST /api/trading/start     → 거래 시작 (paper/live)
├── POST /api/trading/stop      → 거래 중지
├── POST /api/kill-switch/activate   → 킬 스위치 활성화
└── POST /api/kill-switch/deactivate → 킬 스위치 해제

클라이언트 자동 갱신:
├── 5초 간격: 상태, 잔고, 포지션, 손익, 미체결 주문
└── 15초 간격: 전체 주문, 안전 이벤트, 리스크 결정
```

---

## 9. 파일 구조

```
coin_trader/
├── main.py                     # CLI + 메인 루프 + _run_tick
├── config/
│   └── settings.py             # 환경 설정 (pydantic-settings)
├── core/
│   ├── contracts.py            # 인터페이스 정의
│   └── models.py               # 도메인 모델 (Pydantic)
├── exchange/
│   ├── base.py                 # 거래소 어댑터 베이스 (httpx + tenacity)
│   └── upbit.py                # 업비트 API 어댑터
├── broker/
│   ├── paper.py                # 모의투자 브로커
│   └── live.py                 # 실거래 브로커
├── strategy/
│   └── conservative.py         # 보수적 추세추종 전략 (EMA+ATR+RSI)
├── risk/
│   ├── limits.py               # 불변 리스크 상수
│   └── manager.py              # 리스크 검증 (8단계)
├── execution/
│   ├── engine.py               # 주문 실행 파이프라인
│   └── idempotency.py          # 중복 주문 방지
├── safety/
│   ├── kill_switch.py          # 긴급 정지
│   └── monitor.py              # 이상 징후 감지
├── llm/
│   ├── advisory.py             # LLM 자문 (API Key + OAuth)
│   ├── oauth_openai.py         # OpenAI OAuth PKCE 인증
│   └── codex_client.py         # ChatGPT Codex API SSE 클라이언트
├── state/
│   └── store.py                # SQLite 상태 저장소
├── security/
│   └── key_manager.py          # API 키 암호화 (Fernet)
├── logging/
│   ├── logger.py               # 구조화 로깅 (structlog)
│   └── redaction.py            # 민감정보 마스킹
├── notify/
│   └── slack.py                # Slack 알림
└── web/
    ├── api.py                  # FastAPI 웹 서버
    └── templates/
        └── dashboard.html      # 대시보드 UI
```
