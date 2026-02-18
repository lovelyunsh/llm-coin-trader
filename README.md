# 코인 자동거래 시스템

LLM이 주도하는 암호화폐 자동거래 시스템입니다. 모의투자가 기본이며, 실거래 전환 시 2단계 확인이 필요합니다.

## 주요 특징

- **LLM 주도 의사결정**: LLM(gpt-5.2, ChatGPT Codex OAuth)이 매매를 직접 결정. BUY_CONSIDER = 매수, SELL_CONSIDER = 매도
- **전략 시그널은 참고 데이터**: EMA, RSI, ATR 시그널은 LLM에 전달되는 참고 정보일 뿐, 단독으로 주문을 내지 않음
- **포지션 보호**: 하드 손절(-10% 자동 시장가 매도), 소프트 손절(-5%~-10% LLM 판단), 익절(+10%+ LLM 판단), 트레일링 스탑
- **안전 최우선**: Kill Switch, 이상징후 감지, 서킷브레이커
- **웹 대시보드**: 포트 8932, 마스터 코드 인증, 수동 매수/매도, AI 판단 로그, 프롬프트 확인
- **업비트 지원**: 업비트 거래소 API 연동 (JWT 인증)

## 거래 심볼

```
BTC/KRW, ETH/KRW, XRP/KRW, ADA/KRW, SOL/KRW
```

## 아키텍처

```
coin_trader/
├── main.py                     # 메인 루프, _run_tick, _resolve_action, LLM 연동
├── config/settings.py          # 설정 (pydantic-settings)
├── core/
│   ├── contracts.py
│   └── models.py               # 도메인 모델 (Position, Order 등)
├── exchange/
│   ├── base.py                 # httpx + tenacity + 레이트 리밋
│   └── upbit.py                # 업비트 어댑터 (배치 ticker, 입출금 조회)
├── broker/
│   ├── paper.py                # 모의투자
│   └── live.py                 # 실거래 브로커 (배치 ticker, 스테일 캐시 폴백, 순입금)
├── strategy/conservative.py    # EMA+RSI+ATR (참고 시그널 전용)
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
    ├── api.py                  # FastAPI + 인증 미들웨어 + 실거래 자동 시작
    └── templates/
        ├── dashboard.html      # 대시보드 (페이지네이션, 모달, 인증)
        └── login.html
```

## LLM 의사결정 구조

LLM은 자문이 아니라 **주도적 의사결정자**입니다.

```
틱(30초) → 가격 조회 → strategy.on_tick() → 시그널(참고용)
    → 마지막 LLM 호출 대비 가격 변화 확인
    → 변화 >= 1% 또는 30분 경과:
        → LLM 호출 (이전 5개 결정 포함)
        → 결정 로그 저장 (프롬프트 포함)
    → 그 외: LLM 스킵, advice=None → HOLD
    → 포지션 보호:
        - 하드 손절(-10%): 시장가 매도, LLM 우회
        - 소프트 손절(-5%~-10%): LLM 판단
        - 익절(+10%+): LLM 판단 (HOLD만 유지, 나머지 매도)
        - 트레일링 스탑: 자동
    → _resolve_action:
        - LLM 활성화 + advice 있음: LLM 따름
        - LLM 활성화 + advice None (스킵/오류): HOLD
        - LLM 비활성화: 전략 따름
    → 주문 실행 (SELL은 레이트 리밋, 일일 주문 한도 우회)
```

### LLM 입력 데이터

- 시장 데이터, 기술적 지표, 보유 포지션, 잔고
- 최근 주문, 최근 캔들
- 이전 5개 결정

### LLM 출력

- `action`: HOLD / BUY_CONSIDER / SELL_CONSIDER
- `confidence`: 최소 0.65 (65%) 이상이어야 실행
- `reasoning`: 한국어
- `risk_notes`: 한국어

### LLM 스킵 조건

가격 변화 < 1% **AND** 마지막 LLM 호출 < 30분 전 → LLM 스킵, action = HOLD (전략 폴백 없음)

## 포지션 보호

| 조건 | 동작 |
|------|------|
| 손실 -10% 이상 (하드 손절) | 시장가 자동 매도, LLM 우회 |
| 손실 -5%~-10% (소프트 손절) | LLM 판단 (SELL_CONSIDER = 매도, HOLD = 유지) |
| 수익 +10% 이상 (익절) | LLM 판단 (명시적 HOLD만 유지, 나머지 매도) |
| 트레일링 스탑 | 최고점 대비 trailing_stop_pct 하락 시 자동 매도 |

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

| 기능 | 설명 |
|------|------|
| 시스템 상태 | 거래 모드, Kill Switch, 가동 시간 |
| 손익 현황 | 초기 잔고 = 2026년 이후 KRW 순입금 기준 |
| 보유 포지션 | 심볼, 수량, 평균진입가, 현재가, 평가금액, 미실현손익, 미실현손익률, 전량매도 버튼 |
| 수동 매수 | 심볼 선택 + KRW 금액 입력 |
| 수동 매도 | 포지션별 "전량 매도" 버튼 |
| AI 판단 로그 | 코인별 필터, 10개씩 페이지네이션, reasoning 모달, "P" 버튼으로 프롬프트 확인 |
| 전체 주문내역 | 10개씩 페이지네이션 |

자동 갱신: 상태/잔고/포지션/손익/미체결 주문 10초, 주문/안전/결정 30초.

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
MARKET_DATA_INTERVAL_SEC=30

# 웹 대시보드 마스터 코드
WEB_MASTER_CODE=your-master-code-here

# LLM 설정 (OAuth 방식)
LLM_ENABLED=true
LLM_AUTH_MODE=oauth
LLM_OAUTH_MODEL=gpt-5.2

# 로그 레벨
LOG_LEVEL=INFO
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

## 결정 로그 파일

코인별로 분리된 JSONL 파일에 저장됩니다:

```
logs/decisions_btc_krw.jsonl
logs/decisions_eth_krw.jsonl
logs/decisions_xrp_krw.jsonl
...
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
