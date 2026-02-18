# 코인 자동거래 시스템

안전성 최우선의 암호화폐 자동거래 시스템입니다. 모의투자가 기본이며, 실거래 전환 시 2단계 확인이 필요합니다.

## 주요 특징

- **안전 최우선**: 모의투자 기본, Kill Switch, 서킷브레이커, 이상징후 감지
- **리스크 관리**: 8단계 안전 게이트 (포지션 크기, 손절/익절, 일일 손실한도, 주문 빈도 제한 등)
- **감사 로그**: 모든 주문/결정/안전 이벤트를 SQLite에 영구 기록
- **웹 대시보드**: 브라우저에서 거래 시작/중지, Kill Switch, 수익률 확인 등 모든 기능 제어
- **업비트 지원**: 업비트 거래소 API 연동 (JWT 인증)
- **LLM 자문**: OpenAI/Anthropic API 연동 (선택사항, 자문 전용 — 직접 주문 불가)

## 설치

### 요구사항

- Python 3.11 이상
- pip 또는 Poetry

### 설치 방법

```bash
# 저장소 클론
git clone <repository-url>
cd coin-trader

# 의존성 설치 (pip)
pip install -e .

# 또는 Poetry 사용
poetry install
```

### 환경 설정

```bash
# 환경 설정 파일 복사
cp .env.example .env

# .env 파일을 편집하여 설정 변경
# 기본값은 모의투자(paper) 모드이므로 별도 설정 없이 바로 실행 가능
```

## 실행 방법

### 1. 웹 대시보드 (권장)

브라우저에서 모든 기능을 제어할 수 있습니다.

```bash
python -m coin_trader.main web
# 또는
trader web

# 기본: http://0.0.0.0:8000
# 포트 변경: trader web --port 9000
# 호스트 변경: trader web --host 127.0.0.1 --port 8000
```

브라우저에서 `http://localhost:8000` 접속 후:
- **모의투자 시작/중지**: 버튼 클릭으로 간편 제어
- **실거래 시작**: 라이브 모드 ARMED 상태에서만 가능
- **Kill Switch**: 긴급 정지 버튼
- **수익률 확인**: 실시간 P&L, 포지션, 잔고 확인
- **주문 내역**: 체결/미체결 주문 조회
- **안전 이벤트**: 서킷브레이커, 이상징후 등 안전 이벤트 로그

### 2. CLI 명령어

```bash
# 모의투자 실행 (1회 틱)
trader run --mode paper --once

# 모의투자 연속 실행
trader run --mode paper

# 실거래 실행 (2단계 확인 필요)
trader run --mode live

# Kill Switch 활성화
trader kill

# Kill Switch + 미체결 주문 취소
trader kill --close-positions

# 시스템 자가진단
trader selftest

# API 키 암호화 저장
trader encrypt-keys --exchange upbit --master-key <마스터키>
```

### 3. 실거래 전환 (주의!)

실거래는 2단계 확인이 필요합니다:

```bash
# 1단계: .env에서 모드 변경
TRADING_MODE=live

# 2단계: 토큰 파일 생성
mkdir -p RUN
echo "ARMED" > RUN/live_mode_token.txt

# 두 조건 모두 충족해야 실거래 가능
# CLI에서는 추가로 "I_UNDERSTAND_LIVE_TRADING" 입력 필요
```

## 아키텍처

```
coin_trader/
├── main.py                 # CLI 진입점 (run, kill, selftest, web, encrypt-keys)
├── core/
│   ├── models.py           # 도메인 모델 (Pydantic, Decimal 기반)
│   └── contracts.py        # 인터페이스 정의 (ABC)
├── config/
│   └── settings.py         # 설정 (pydantic-settings, fail-closed)
├── security/
│   └── key_manager.py      # API 키 Fernet 암호화/복호화
├── logging/
│   ├── redaction.py        # 민감정보 마스킹
│   └── logger.py           # structlog JSON 로깅 + JSONL 이벤트
├── state/
│   └── store.py            # SQLite WAL 모드, 감사 로그
├── exchange/
│   ├── base.py             # httpx + tenacity 재시도 + 레이트 리밋
│   └── upbit.py            # 업비트 어댑터 (PyJWT 인증)
├── risk/
│   ├── limits.py           # 불변 리스크 한도 (frozen dataclass)
│   └── manager.py          # 8단계 안전 게이트
├── safety/
│   ├── kill_switch.py      # 파일+메모리 기반 Kill Switch
│   └── monitor.py          # 이상징후 감지
├── broker/
│   ├── paper.py            # 모의투자 브로커 (수수료 0.05%, 슬리피지 0.1%)
│   └── live.py             # 실거래 브로커
├── execution/
│   ├── engine.py           # 실행 파이프라인 (kill→중복→리스크→주문→저장)
│   └── idempotency.py      # 중복 주문 방지
├── strategy/
│   └── conservative.py     # EMA 크로스오버 + ATR + RSI
├── llm/
│   └── advisory.py         # LLM 자문 (주문 불가, 자문만)
├── notify/
│   └── slack.py            # Slack 알림
└── web/
    ├── api.py              # FastAPI REST API + 웹 대시보드 서버
    └── templates/
        └── dashboard.html  # 웹 대시보드 (한글 UI)
```

## 안전장치

| 안전장치 | 설명 |
|---------|------|
| **모의투자 기본** | 설정 오류 시에도 모의투자로 폴백 |
| **2단계 실거래 게이트** | .env 설정 + 토큰 파일 + CLI 확인 |
| **Kill Switch** | 파일+메모리+시그널(SIGTERM/SIGINT) 기반 즉시 정지 |
| **서킷브레이커** | 급격한 가격 변동 시 거래 자동 중단 (기본 15%) |
| **포지션 한도** | 최대 포지션 수 5개, 1회 최대 10% |
| **일일 손실한도** | 5% 초과 시 거래 중단 |
| **주문 빈도 제한** | 초당 1건, 일일 100건 |
| **이상징후 감지** | API 연속 실패, 잔고 불일치, 가격 이상 감지 |
| **감사 로그** | 모든 이벤트 SQLite에 영구 기록 |
| **민감정보 마스킹** | API 키, JWT 토큰 등 로그에서 자동 마스킹 |
| **불변 리스크 한도** | frozen dataclass로 런타임 변경 불가 |
| **선물/레버리지 차단** | 기본 비활성화, 현물 전용 |

## API 엔드포인트

웹 대시보드 서버 실행 시 사용 가능한 REST API:

| 메서드 | 경로 | 설명 |
|--------|------|------|
| GET | `/` | 웹 대시보드 페이지 |
| GET | `/api/status` | 시스템 상태 |
| POST | `/api/trading/start` | 거래 시작 (`{mode: "paper"\|"live"}`) |
| POST | `/api/trading/stop` | 거래 중지 |
| POST | `/api/kill-switch/activate` | Kill Switch 활성화 |
| POST | `/api/kill-switch/deactivate` | Kill Switch 해제 |
| GET | `/api/balances` | 잔고 조회 |
| GET | `/api/positions` | 포지션 조회 |
| GET | `/api/orders` | 주문 내역 |
| GET | `/api/orders/open` | 미체결 주문 |
| GET | `/api/safety-events` | 안전 이벤트 로그 |
| GET | `/api/decisions` | 리스크 결정 로그 |
| GET | `/api/pnl` | 수익률 요약 |

## 설정 (.env)

주요 설정 항목:

```env
# 운영 모드 (paper/live)
TRADING_MODE=paper

# 거래소
EXCHANGE=upbit

# 거래 심볼 (JSON 배열 형식)
TRADING_SYMBOLS=["BTC/KRW","ETH/KRW"]

# 시장 데이터 갱신 주기 (초)
MARKET_DATA_INTERVAL_SEC=60

# 로그 레벨
LOG_LEVEL=INFO
```

전체 설정은 `.env.example` 파일을 참고하세요.

## Docker 실행

### 이미지 빌드

```bash
docker build -t coin-trader .
```

### 컨테이너 실행

```bash
docker run -d --name coin-trader -p 8932:8932 \
  -v "$(pwd)/.env:/app/.env:ro" \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/RUN:/app/RUN" \
  -v "$(pwd)/logs:/app/logs" \
  coin-trader
```

- `.env` — 설정 파일 (읽기 전용 마운트)
- `data/` — DB, 암호화된 API 키, OAuth 토큰 등 영구 데이터
- `RUN/` — 실거래 토큰, Kill Switch 파일
- `logs/` — JSONL 감사 로그

### 컨테이너 관리

```bash
# 로그 확인
docker logs -f coin-trader

# 중지
docker stop coin-trader

# 재시작
docker start coin-trader

# 삭제 (컨테이너만, 데이터는 볼륨에 보존)
docker rm coin-trader

# 이미지 재빌드 (코드 변경 시)
docker build -t coin-trader . && docker rm -f coin-trader && \
docker run -d --name coin-trader -p 8932:8932 \
  -v "$(pwd)/.env:/app/.env:ro" \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/RUN:/app/RUN" \
  -v "$(pwd)/logs:/app/logs" \
  coin-trader
```

실행 후 `http://localhost:8932` 에서 웹 대시보드에 접속할 수 있습니다.

## 라이선스

MIT
