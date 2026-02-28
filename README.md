# 코인 자동거래 시스템

LLM이 주도하는 암호화폐 자동거래 시스템입니다. 모의투자가 기본이며, 실거래 전환 시 2단계 확인이 필요합니다.

## 주요 특징

- **LLM 주도 의사결정**: LLM(ChatGPT Codex OAuth, 모델 설정 가능)이 매매를 직접 결정. BUY_CONSIDER = 매수, SELL_CONSIDER = 매도. 시장가/지정가 주문 타입도 LLM이 결정
- **동적 심볼 유니버스**: 1시간마다 KRW 마켓 후보를 필터링하고 LLM이 최종 거래 심볼을 선발
- **전략 시그널은 참고 데이터**: EMA, RSI, ATR 시그널은 LLM에 전달되는 참고 정보일 뿐, 단독으로 주문을 내지 않음
- **BTC 일봉 EMA200 필터**: BTC 일봉 기준 200일 EMA로 시장 레짐 판단, 알트 매매 시 LLM에 전달
- **포지션 보호**: 하드 손절(-10% 자동 시장가 매도), 소프트 손절(-5%~-10% LLM 판단), 익절(+10%+ LLM 완전 위임), 트레일링 스탑
- **안전 최우선**: Kill Switch, 이상징후 감지, 서킷브레이커
- **웹 대시보드**: 포트 8932, 마스터 코드 인증(브루트포스 방어, CSRF 보호), 수동 매수/매도, AI 판단 로그, AI 심볼 판단 로그, 프롬프트 확인
- **멀티 거래소**: 업비트(KRW 현물) + 바이낸스(USDT-M 선물) 지원
- **바이낸스 USDT-M 선물 지원**: 같은 이미지로 업비트(현물)와 바이낸스(선물) 병렬 운영. 숏/커버, 레버리지, 청산가 추적

## 거래 심볼

- 기본은 `TRADING_SYMBOLS` 초기값을 사용해 시작합니다.
- 동적 심볼 선택이 활성화되면 `DYNAMIC_SYMBOL_REFRESH_SEC` 주기로 유니버스를 재선정합니다.
- `ALWAYS_KEEP_SYMBOLS`와 현재 보유 심볼은 강제 유지됩니다.

## 아키텍처

```
coin_trader/
├── main.py                     # 엔트리포인트(웹서버), _build_system, _run_tick, _resolve_action
├── config/settings.py          # 설정 (pydantic-settings)
├── core/
│   ├── contracts.py
│   └── models.py               # 도메인 모델 (Position, Order 등)
├── exchange/
│   ├── base.py                 # httpx + tenacity + 레이트 리밋
│   ├── upbit.py                # 업비트 어댑터 (정규화, 배치 ticker/orderbook)
│   └── binance_futures.py      # 바이낸스 USDT-M 선물 어댑터 (HMAC-SHA256, 테스트넷)
├── broker/
│   ├── paper.py                # 모의투자 (현물)
│   ├── live.py                 # 실거래 브로커 (현물)
│   ├── paper_futures.py        # 모의투자 (선물, 마진/레버리지/청산)
│   └── live_futures.py         # 실거래 브로커 (선물)
├── strategy/conservative.py    # EMA+RSI+ATR (참고 시그널 전용, ema200_1h로 구분)
├── risk/
│   ├── limits.py               # 불변 리스크 상수
│   └── manager.py              # 리스크 게이트 (포지션 수 제한 없음, SELL은 레이트/일일 한도 우회)
├── execution/
│   ├── engine.py               # 주문 실행 파이프라인
│   └── idempotency.py
├── safety/
│   ├── kill_switch.py
│   └── monitor.py              # 이상징후 감지
├── llm/
│   ├── advisory.py             # LLM 주도 의사결정 (한국어 출력, 프롬프트 저장)
│   ├── oauth_openai.py         # OAuth PKCE 인증
│   └── codex_client.py         # ChatGPT Codex SSE 클라이언트
├── state/store.py              # SQLite WAL
├── security/key_manager.py     # Fernet 암호화
├── logging/
│   ├── logger.py               # structlog + JSONL 타입별 분리 + HOLD 압축
│   └── redaction.py
├── notify/slack.py
└── web/
    ├── api.py                  # FastAPI + 인증 미들웨어(브루트포스/CSRF) + 실거래 자동 시작
    └── templates/
        ├── dashboard.html      # 대시보드 (페이지네이션, 모달, 인증)
        └── login.html
```

## LLM 의사결정 구조

LLM은 자문이 아니라 **주도적 의사결정자**입니다.

```
틱(120초) → 가격 조회 → strategy.on_tick() → 시그널(참고용)
    → BTC 일봉 EMA200 트렌드 계산 (15분 캐시, 일봉 1시간마다 갱신)
    → 마지막 LLM 호출 대비 가격 변화 확인
    → 변화 >= 1% 또는 30분 경과:
        → LLM 호출 (이전 5개 결정 포함)
        → 결정 로그 저장 (프롬프트 포함)
    → 그 외: LLM 스킵, advice=None → HOLD
        ※ 단, 소프트 손절(-5%~-10%) 또는 익절(+10%+) 구간이면 강제 LLM 호출
    → 포지션 보호 (보호 매도 발생 시 이후 combined 판단 스킵):
        - 하드 손절(-10%): 시장가 매도, LLM 우회
        - 소프트 손절(-5%~-10%): LLM 판단
        - 익절(+10%+): LLM 완전 위임 (SELL_CONSIDER만 매도, 나머지 유지. 트레일링 스탑이 안전망)
        - 트레일링 스탑: 자동
    → _resolve_action:
        - LLM 활성화 + advice 있음: LLM 따름
        - LLM 활성화 + advice None (스킵/오류): HOLD
        - LLM 비활성화: 전략 따름
    → 주문 실행 (SELL은 레이트 리밋, 일일 주문 한도 우회)
```

## 동적 심볼 유니버스 (LLM 선발)

기본값 기준: 1시간마다 재선정, 최종 5개 선발(`DYNAMIC_SYMBOL_TOP_K=5`), 전체 심볼 cap 10(`DYNAMIC_SYMBOL_MAX_SYMBOLS=10`).

```
1) KRW 마켓 전체 조회 + 배치 ticker 수집
2) 거래대금 최소치 필터(DYNAMIC_SYMBOL_MIN_KRW_24H)
3) 후보 K 계산: min(20, max(12, 3*top_n))
4) 하드 필터(스프레드/24h 변동/일중 변동폭)
5) LLM이 최종 top_n 선발
   - candidates: 신규 후보 지표
   - active_symbols: 현재 활성 심볼 지표(코어 제외)
6) 최종 활성 심볼 = 강제유지(ALWAYS_KEEP + 보유) + LLM 선발, max_symbols 이내
```

### LLM 입력 데이터

- 시장 데이터, 기술적 지표, 보유 포지션, 잔고
- 최근 주문, 최근 캔들
- 이전 5개 결정

### LLM 출력

- `action`: HOLD / BUY_CONSIDER / SELL_CONSIDER / SHORT_CONSIDER / COVER_CONSIDER
- `confidence`: 최소 0.65 (65%) 이상이어야 실행
- `reasoning`: 한국어
- `risk_notes`: 한국어
- `buy_pct`: 총자산(KRW 현금 + 코인 평가액) 대비 매수 비율 (0-30%)
- `sell_pct`: 보유 수량 대비 매도 비율 (0-100%)
- `order_type`: "market" (시장가) 또는 "limit" (지정가, 기본값). LLM이 주문 타입 결정
- `target_price`: 지정가 주문 시 희망 가격

SHORT_CONSIDER/COVER_CONSIDER는 선물 모드(`FUTURES_ENABLED=true`)에서만 활성화

### LLM 스킵 조건

가격 변화 < 1% **AND** 마지막 LLM 호출 < 30분 전 → LLM 스킵, action = HOLD (전략 폴백 없음)

**예외**: 보유 포지션이 소프트 손절(-5%~-10%) 또는 익절(+10%+) 구간에 있으면 스킵 조건을 무시하고 LLM을 강제 호출합니다.

## 포지션 보호

| 조건 | 동작 |
|------|------|
| 손실 -10% 이상 (하드 손절) | 시장가 자동 매도, LLM 우회 |
| 손실 -5%~-10% (소프트 손절) | LLM 판단 (SELL_CONSIDER = 매도, HOLD = 유지) |
| 수익 +10% 이상 (익절) | LLM 완전 위임 (SELL_CONSIDER만 매도, 나머지 유지. 트레일링 스탑 안전망) |
| 트레일링 스탑 | 최고점 대비 trailing_stop_pct 하락 시 자동 매도 |
| 숏 손실 -10% 이상 (하드 손절) | 시장가 자동 커버(BUY), LLM 우회 |
| 숏 손실 -5%~-10% (소프트 손절) | LLM 판단 (COVER_CONSIDER = 커버, HOLD = 유지) |
| 숏 트레일링 스탑 | 최저점 대비 반등 trailing_stop_pct 시 자동 커버 |

손절 주문은 LIMIT이 아닌 **MARKET 주문**으로 실행됩니다.

## 실행 방법

### Docker (권장)

```bash
# 이미지 빌드
docker build -t coin-trader .

# 컨테이너 실행
docker run -d --name coin-trader -p 8932:8932 \
  -v "$(pwd)/.env:/app/.env:ro" \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/RUN:/app/RUN" \
  -v "$(pwd)/logs:/app/logs" \
  coin-trader
```

`.env`에 `TRADING_MODE=live`가 설정되어 있으면 컨테이너 재시작 시 실거래가 자동으로 시작됩니다.

실행 후 `http://localhost:8932` 접속 (마스터 코드 로그인 필요).

## 바이낸스 선물 실행 (병렬 운영)

같은 Docker 이미지를 사용하되, 다른 env 파일과 컨테이너 이름으로 병렬 운영합니다.

### .env.binance 설정

```env
# 거래소
EXCHANGE=binance
TRADING_MODE=paper

# 심볼
TRADING_SYMBOLS=["BTC/USDT","ETH/USDT"]
ALWAYS_KEEP_SYMBOLS=BTC/USDT,ETH/USDT

# 바이낸스 설정
BINANCE_TESTNET=true
BINANCE_KEY_FILE=data/binance_keys.enc
BINANCE_MASTER_KEY=
BINANCE_MARGIN_TYPE=isolated
BINANCE_DEFAULT_LEVERAGE=1

# 선물 활성화
FUTURES_ENABLED=true
MAX_LEVERAGE=1

# Quote currency
QUOTE_CURRENCY=USDT
BTC_REFERENCE_SYMBOL=BTC/USDT

# 동적 심볼 (USDT 기준 거래대금)
DYNAMIC_SYMBOL_MIN_TURNOVER_24H=10000000

# LLM/웹/기타는 업비트와 동일하게 설정
LLM_ENABLED=true
LLM_AUTH_MODE=oauth
WEB_MASTER_CODE=your-master-code-here
LOG_LEVEL=INFO
```

### Docker 실행

```bash
# 업비트 (기존, 변경 없음)
docker run -d --name coin-trader \
  --restart unless-stopped -p 8932:8932 \
  --env-file .env \
  -v $(pwd)/data:/app/data -v $(pwd)/logs:/app/logs -v $(pwd)/RUN:/app/RUN \
  coin-trader

# 바이낸스 (신규, 별도 포트/데이터)
docker run -d --name coin-trader-binance \
  --restart unless-stopped -p 8933:8932 \
  --env-file .env.binance \
  -v $(pwd)/data-binance:/app/data -v $(pwd)/logs-binance:/app/logs -v $(pwd)/RUN-binance:/app/RUN \
  coin-trader
```

### 바이낸스 API 키 설정

1. https://www.binance.com/en/my/settings/api-management 에서 API 키 발급
2. USDT-M 선물 거래 권한 활성화 (출금 비활성화 권장)
3. API 키는 환경 변수(`BINANCE_MASTER_KEY`)로 복호화 키를 전달하고, `data/binance_keys.enc`에 Fernet 암호화되어 저장됩니다.

### 테스트넷부터 시작

`BINANCE_TESTNET=true`로 테스트넷에서 먼저 검증한 후, 확인이 되면 `false`로 변경하여 메인넷으로 전환합니다.

접속: `http://localhost:8933` (바이낸스 대시보드)

### 컨테이너 관리

```bash
# 로그 확인
docker logs -f coin-trader

# 중지
docker stop coin-trader

# 재시작
docker start coin-trader

# 이미지 재빌드 (코드 변경 시)
docker build -t coin-trader . && docker rm -f coin-trader && \
docker run -d --name coin-trader -p 8932:8932 \
  -v "$(pwd)/.env:/app/.env:ro" \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/RUN:/app/RUN" \
  -v "$(pwd)/logs:/app/logs" \
  coin-trader
```

## 웹 대시보드

포트 **8932**, `/login`에서 마스터 코드(`WEB_MASTER_CODE`) 입력 후 쿠키 세션 유지.

### 인증 보안

| 보안 기능 | 설명 |
|-----------|------|
| 브루트포스 방어 | IP당 5회 실패 시 5분 잠금 (429 응답) |
| 타이밍 공격 방어 | `hmac.compare_digest`로 상수 시간 비교 |
| CSRF 보호 | Double-submit cookie 패턴 (`ct_csrf` 쿠키 + `x-csrf-token` 헤더) |
| Secure 쿠키 | HTTPS 접속 시 자동으로 `secure=True` 설정 |
| HttpOnly 세션 | 세션 쿠키 XSS 탈취 불가 |
| 서버측 세션 만료 | 24시간 TTL, 주기적 자동 정리 |

| 기능 | 설명 |
|------|------|
| 시스템 상태 | 거래 모드, Kill Switch, 가동 시간 |
| 손익 현황 | 초기 잔고 = 2026년 이후 KRW 순입금 기준 |
| 보유 포지션 | 심볼, 수량, 평균진입가, 현재가, 평가금액, 미실현손익, 미실현손익률, 전량매도 버튼 |
| 수동 매수 | 심볼 선택 + KRW 금액 입력 |
| 수동 매도 | 포지션별 "전량 매도" 버튼 |
| AI 판단 로그 | 코인별 필터, 10개씩 페이지네이션, reasoning 모달, "P" 버튼으로 프롬프트 확인 |
| AI 심볼 판단 로그 | 10개씩 페이지네이션, LLM 사유 클릭 모달, "P" 버튼으로 유니버스 프롬프트 확인 |
| 전체 주문내역 | 10개씩 페이지네이션 |

자동 갱신: 상태/잔고/포지션/손익/미체결 주문 15초, 주문/안전/결정 30초.

## API 엔드포인트

| 메서드 | 경로 | 설명 |
|--------|------|------|
| GET | `/` | 대시보드 (인증 필요) |
| GET | `/login` | 로그인 페이지 |
| POST | `/api/auth/login` | 마스터 코드 인증 |
| GET | `/api/status` | 시스템 상태 |
| POST | `/api/trading/start` | 거래 시작 `{mode: "paper"|"live"}` |
| POST | `/api/trading/stop` | 거래 중지 |
| POST | `/api/kill-switch/activate` | Kill Switch 활성화 |
| POST | `/api/kill-switch/deactivate` | Kill Switch 해제 |
| GET | `/api/balances` | 잔고 조회 |
| GET | `/api/positions` | 포지션 조회 (현재가 포함) |
| GET | `/api/orders` | 전체 주문 내역 |
| GET | `/api/orders/open` | 미체결 주문 |
| POST | `/api/orders/manual-buy` | 수동 매수 `{symbol, amount_krw}` |
| POST | `/api/orders/manual-sell` | 수동 매도 `{symbol}` |
| GET | `/api/safety-events` | 안전 이벤트 로그 |
| GET | `/api/decisions` | 전체 AI 결정 로그 |
| GET | `/api/decisions/{symbol}` | 코인별 AI 결정 로그 |
| GET | `/api/symbol-decisions` | AI 심볼 선발 로그 |
| GET | `/api/pnl` | 손익 요약 |

## 설정 (.env)

주요 설정 항목:

```env
# 운영 모드 (paper/live)
TRADING_MODE=paper

# 거래소
EXCHANGE=upbit

# 거래 심볼 (JSON 배열 형식)
TRADING_SYMBOLS=["BTC/KRW","ETH/KRW","XRP/KRW","ADA/KRW","SOL/KRW"]

# 시장 데이터 갱신 주기 (초) - 틱 간격
MARKET_DATA_INTERVAL_SEC=120

# 캔들 갱신 주기 (초)
CANDLE_REFRESH_INTERVAL_SEC=3600

# 동적 심볼 유니버스
DYNAMIC_SYMBOL_SELECTION_ENABLED=true
DYNAMIC_SYMBOL_REFRESH_SEC=3600
DYNAMIC_SYMBOL_TOP_K=5
DYNAMIC_SYMBOL_MAX_SYMBOLS=10
DYNAMIC_SYMBOL_MIN_KRW_24H=1000000000
DYNAMIC_SYMBOL_BATCH_SIZE=80
DYNAMIC_SYMBOL_MAX_SPREAD_BPS=80
DYNAMIC_SYMBOL_MAX_ABS_CHANGE_24H_PCT=20
DYNAMIC_SYMBOL_MAX_INTRADAY_RANGE_PCT=30
ALWAYS_KEEP_SYMBOLS=BTC/KRW,ETH/KRW

# 웹 대시보드 마스터 코드
WEB_MASTER_CODE=your-master-code-here

# LLM 설정 (OAuth 방식)
LLM_ENABLED=true
LLM_AUTH_MODE=oauth
LLM_OAUTH_MODEL=gpt-5.1-codex-mini

# 로그 레벨
LOG_LEVEL=INFO

# 바이낸스 전용 설정 (.env.binance)
EXCHANGE=binance
QUOTE_CURRENCY=USDT
BTC_REFERENCE_SYMBOL=BTC/USDT
BINANCE_TESTNET=true
BINANCE_MARGIN_TYPE=isolated
BINANCE_DEFAULT_LEVERAGE=1
FUTURES_ENABLED=true
DYNAMIC_SYMBOL_MIN_TURNOVER_24H=10000000
```

전체 설정은 `.env.example` 파일을 참고하세요.

## 안전장치

| 안전장치 | 설명 |
|---------|------|
| **모의투자 기본** | 설정 오류 시에도 모의투자로 폴백 |
| **2단계 실거래 게이트** | .env 설정 + 토큰 파일 |
| **Kill Switch** | 파일+메모리+시그널(SIGTERM/SIGINT) 기반 즉시 정지 |
| **하드 손절** | -10% 도달 시 시장가 자동 매도, LLM 우회 |
| **소프트 손절** | -5%~-10% 구간에서 LLM이 판단 |
| **트레일링 스탑** | 최고점 대비 설정 비율 하락 시 자동 매도 |
| **서킷브레이커** | 급격한 가격 변동 시 거래 자동 중단 |
| **일일 손실한도** | 초과 시 거래 중단 |
| **주문 빈도 제한** | BUY에만 적용, SELL은 항상 통과 |
| **이상징후 감지** | API 연속 실패, 잔고 불일치, 가격 이상 감지 |
| **감사 로그** | 모든 이벤트 SQLite에 영구 기록 |
| **민감정보 마스킹** | API 키, JWT 토큰 등 로그에서 자동 마스킹 |
| **웹 인증 보안** | 브루트포스 잠금, CSRF 토큰, Secure 쿠키, 서버측 세션 만료 |
| **선물 리스크 체크** | 레버리지 마진 확인, 노셔널 한도, 청산가 근접도, 펀딩비 체크 |

## 결정 로그 파일

코인별로 분리된 JSONL 파일에 저장됩니다:

```
logs/decisions_btc_krw.jsonl
logs/decisions_eth_krw.jsonl
logs/decisions_xrp_krw.jsonl
...
logs/symbol_decisions.jsonl
```

연속 HOLD가 1시간 초과 시 첫 번째와 마지막만 남기고 압축됩니다. LLM 스킵 결정(가격 변화 < 1% + 30분 미경과)은 로그에 기록되지 않습니다.

## 새 서버로 이전

```bash
# 1. 코드 클론
git clone https://github.com/lovelyunsh/llm-coin-trader.git
cd llm-coin-trader

# 2. 필수 파일 수동 복사 (기존 서버에서)
#    .env
#    data/upbit_keys.enc
#    data/.auth/openai-oauth.json
#    RUN/live_mode_token.txt

# 3. Docker 빌드 및 실행
docker build -t coin-trader .
docker run -d --name coin-trader -p 8932:8932 \
  -v "$(pwd)/.env:/app/.env:ro" \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/RUN:/app/RUN" \
  -v "$(pwd)/logs:/app/logs" \
  coin-trader
```

## 라이선스

MIT
