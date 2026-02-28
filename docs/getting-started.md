# 코인 자동거래 시스템 - A to Z 시작 가이드

## 목차

1. [사전 요구사항](#1-사전-요구사항)
2. [프로젝트 설치](#2-프로젝트-설치)
3. [환경 설정 (.env)](#3-환경-설정)
4. [LLM 설정 (OAuth)](#4-llm-설정)
5. [실거래 전환하기](#5-실거래-전환하기)
6. [Docker로 실행하기](#6-docker로-실행하기)
7. [웹 대시보드 사용법](#7-웹-대시보드-사용법)
8. [새 서버로 이전하기](#8-새-서버로-이전하기)
9. [긴급 정지 (Kill Switch)](#9-긴급-정지)
10. [로그 확인 방법](#10-로그-확인-방법)
11. [Slack 알림 연동](#11-slack-알림-연동)
12. [FAQ / 트러블슈팅](#12-faq--트러블슈팅)

---

## 1. 사전 요구사항

| 항목 | 요구 |
|------|------|
| Docker | 최신 버전 |
| OS | macOS / Linux |
| 거래소 계정 | 업비트 (API 키 발급 필요, 실거래 시) |
| 바이낸스 계정 | API 키 발급 필요 (선물 거래 시) |
| ChatGPT 계정 | LLM OAuth 인증용 |

모의투자(Paper)는 거래소 계정 없이 바로 시작할 수 있습니다.

---

## 2. 프로젝트 설치

```bash
# 저장소 클론
git clone https://github.com/lovelyunsh/llm-coin-trader.git
cd llm-coin-trader

# Docker 이미지 빌드
docker build -t coin-trader .
```

---

## 3. 환경 설정

### 3.1 .env 파일 생성

```bash
cp .env.example .env
```

### 3.2 필수 설정 항목

`.env` 파일을 열어 아래 항목을 설정합니다:

```env
# 운영 모드 (paper/live)
TRADING_MODE=paper

# 거래소
EXCHANGE=upbit

# 거래 심볼 (JSON 배열 형식)
TRADING_SYMBOLS=["BTC/KRW","ETH/KRW","XRP/KRW","ADA/KRW","SOL/KRW"]

# 틱 간격 (초)
MARKET_DATA_INTERVAL_SEC=60

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

# 웹 대시보드 마스터 코드 (로그인에 사용)
WEB_MASTER_CODE=your-master-code-here

# LLM 설정
LLM_ENABLED=true
LLM_AUTH_MODE=oauth
LLM_OAUTH_MODEL=gpt-5.2

# 업비트 마스터 키 (실거래 시 필요)
UPBIT_MASTER_KEY=your-master-key-here

# 로그 레벨
LOG_LEVEL=INFO
```

### 3.3 디렉토리 구조

```
coin-trader/
├── data/              # DB, 암호화 키, OAuth 토큰
├── logs/              # JSONL 로그 파일
├── RUN/               # Kill Switch, 실거래 토큰
└── .env               # 환경 설정
```

---

## 4. LLM 설정

이 시스템은 LLM이 매매를 **직접 결정**합니다. LLM 없이는 HOLD만 반복됩니다.

### OAuth 방식 (권장, gpt-5.2 사용)

로컬 머신에서 최초 1회 로그인이 필요합니다:

```bash
# 로컬에서 Python으로 OAuth 토큰 생성
# (Docker 환경에서는 브라우저 열기가 불가하므로 로컬에서 먼저 진행)
LLM_OAUTH_OPEN_BROWSER=true python -c "
from coin_trader.llm.oauth_openai import get_reusable_auth
import asyncio
asyncio.run(get_reusable_auth())
"
```

로그인 후 `data/.auth/openai-oauth.json`이 생성됩니다. 이 파일을 서버의 `data/.auth/` 디렉토리에 복사하면 이후 자동으로 토큰이 갱신됩니다.

서버(헤드리스) 환경에서는:

```env
LLM_OAUTH_OPEN_BROWSER=false
```

토큰 파일이 없으면 LLM이 비활성화되고 모든 틱이 HOLD로 처리됩니다.

### LLM 동작 방식

- 가격 변화 >= 1% 또는 마지막 LLM 호출 후 30분 경과 시 LLM 호출
- 그 외에는 LLM 스킵, action = HOLD (전략 폴백 없음)
- LLM 최소 신뢰도: 0.65 (65%)
- LLM 출력(reasoning, risk_notes)은 한국어

---

## 5. 실거래 전환하기

> **경고**: 실거래는 실제 자산이 거래됩니다. 반드시 모의투자로 충분히 테스트한 후 전환하세요.

### 5.1 업비트 API 키 발급

1. https://upbit.com/mypage/open_api_management 접속
2. API 키 발급 (출금 권한은 비활성화 권장)
3. Access Key와 Secret Key 복사

### 5.2 API 키 암호화 저장

API 키는 환경 변수(`UPBIT_MASTER_KEY`)로 복호화 키를 전달하고, `data/upbit_keys.enc`에 Fernet 암호화되어 저장됩니다. 원본 키는 어디에도 평문으로 저장되지 않습니다.

### 5.3 .env 설정

```env
TRADING_MODE=live
UPBIT_MASTER_KEY=your-master-key-here
```

### 5.4 실거래 토큰 파일 생성

```bash
mkdir -p RUN
echo "ARMED" > RUN/live_mode_token.txt
```

`.env`의 `TRADING_MODE=live`와 이 토큰 파일이 모두 있어야 실거래가 시작됩니다.

### 5.5 실거래 체크리스트

- [ ] 모의투자에서 충분히 테스트했는가?
- [ ] API 키 권한에 출금이 비활성화되어 있는가?
- [ ] `WEB_MASTER_CODE`가 설정되어 있는가?
- [ ] `data/.auth/openai-oauth.json`이 존재하는가?
- [ ] `data/upbit_keys.enc`가 존재하는가?
- [ ] `RUN/live_mode_token.txt`가 존재하는가?

---

## 5.5 바이낸스 선물 설정

### .env.binance 파일 생성

업비트와 별도의 환경 파일을 생성합니다:

```env
EXCHANGE=binance
TRADING_MODE=paper
TRADING_SYMBOLS=["BTC/USDT","ETH/USDT"]
BINANCE_TESTNET=true
FUTURES_ENABLED=true
QUOTE_CURRENCY=USDT
BTC_REFERENCE_SYMBOL=BTC/USDT
ALWAYS_KEEP_SYMBOLS=BTC/USDT,ETH/USDT
DYNAMIC_SYMBOL_MIN_TURNOVER_24H=10000000
WEB_MASTER_CODE=your-master-code-here
LLM_ENABLED=true
LLM_AUTH_MODE=oauth
LOG_LEVEL=INFO
```

### 바이낸스 API 키 발급

1. https://www.binance.com/en/my/settings/api-management 접속
2. API 키 생성 (USDT-M 선물 권한 활성화, 출금 비활성화 권장)
3. IP 화이트리스트 설정 권장

### API 키 암호화

API 키는 환경 변수(`BINANCE_MASTER_KEY`)로 복호화 키를 전달하고, `data/binance_keys.enc`에 Fernet 암호화되어 저장됩니다.

### 바이낸스 컨테이너 실행

```bash
docker run -d --name coin-trader-binance \
  --restart unless-stopped -p 8933:8932 \
  --env-file .env.binance \
  -v $(pwd)/data-binance:/app/data \
  -v $(pwd)/logs-binance:/app/logs \
  -v $(pwd)/RUN-binance:/app/RUN \
  coin-trader
```

테스트넷(`BINANCE_TESTNET=true`)에서 충분히 검증한 후 메인넷으로 전환하세요.

접속: `http://localhost:8933`

---

## 6. Docker로 실행하기

### 컨테이너 시작

```bash
docker run -d --name coin-trader -p 8932:8932 \
  -v "$(pwd)/.env:/app/.env:ro" \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/RUN:/app/RUN" \
  -v "$(pwd)/logs:/app/logs" \
  coin-trader
```

볼륨 설명:
- `.env` — 설정 파일 (읽기 전용)
- `data/` — DB, 암호화 키, OAuth 토큰 등 영구 데이터
- `RUN/` — 실거래 토큰, Kill Switch 파일
- `logs/` — JSONL 감사 로그

`.env`에 `TRADING_MODE=live`가 설정되어 있으면 컨테이너 재시작 시 실거래가 자동으로 시작됩니다.

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

---

## 7. 웹 대시보드 사용법

브라우저에서 `http://localhost:8932` 접속 후 `/login` 페이지에서 `WEB_MASTER_CODE`를 입력합니다.

### 인증 보안

- **브루트포스 방어**: IP당 5회 실패 시 5분 잠금 (HTTP 429)
- **CSRF 보호**: 모든 POST 요청에 Double-submit cookie 패턴 적용
- **Secure 쿠키**: HTTPS 접속 시 자동 활성화 (HTTP/localhost에서도 정상 작동)
- **서버측 세션 만료**: 24시간 후 자동 만료 및 정리

| 탭/섹션 | 내용 |
|---------|------|
| 시스템 상태 | 거래 모드, Kill Switch, 가동 시간 |
| 손익 현황 | 초기 잔고 = 2026년 이후 KRW 순입금 기준 |
| 보유 포지션 | 심볼, 수량, 평균진입가, 현재가, 평가금액, 미실현손익, 미실현손익률, 전량매도 버튼 |
| 수동 매수 | 심볼 선택 + KRW 금액 입력 |
| AI 판단 로그 | 코인별 필터, 10개씩 페이지네이션, reasoning 모달 팝업 |
| AI 심볼 판단 로그 | 10개씩 페이지네이션, LLM 사유 클릭 모달, "P" 버튼으로 심볼 선발 프롬프트 확인 |
| 프롬프트 확인 | AI 판단 로그/AI 심볼 판단 로그의 "P" 버튼 클릭 |
| 전체 주문내역 | 10개씩 페이지네이션 |

자동 갱신 주기:
- 15초: 상태, 잔고, 포지션, 손익, 미체결 주문
- 30초: 전체 주문, 안전 이벤트, AI 결정

---

## 8. 새 서버로 이전하기

```bash
# 1. 코드 클론
git clone https://github.com/lovelyunsh/llm-coin-trader.git
cd llm-coin-trader

# 2. Docker 이미지 빌드
docker build -t coin-trader .
```

기존 서버에서 아래 파일들을 수동으로 복사합니다:

```
.env
data/upbit_keys.enc
data/.auth/openai-oauth.json
RUN/live_mode_token.txt
```

```bash
# 컨테이너 실행
docker run -d --name coin-trader -p 8932:8932 \
  -v "$(pwd)/.env:/app/.env:ro" \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/RUN:/app/RUN" \
  -v "$(pwd)/logs:/app/logs" \
  coin-trader
```

---

## 9. 긴급 정지

문제 발생 시 즉시 모든 거래를 중단합니다.

### 방법 1: 웹 대시보드

`http://localhost:8932` → **Kill Switch 활성화** 버튼 클릭

### 방법 2: 파일 생성

```bash
echo "emergency stop" > RUN/kill_switch
```

### 방법 3: Docker 컨테이너 중지

```bash
docker stop coin-trader
```

### Kill Switch 해제

```bash
# 파일 삭제
rm -f RUN/kill_switch
```

또는 웹 대시보드 → **Kill Switch 해제** 버튼

---

## 10. 로그 확인 방법

### Docker 로그

```bash
docker logs -f coin-trader
```

### 결정 로그 파일 (코인별)

```bash
# BTC 결정 로그
tail -f logs/decisions_btc_krw.jsonl

# ETH 결정 로그
tail -f logs/decisions_eth_krw.jsonl
```

연속 HOLD가 1시간 초과 시 첫 번째와 마지막만 남기고 자동 압축됩니다. LLM 스킵(가격 변화 < 1% + 30분 미경과)은 로그에 기록되지 않습니다.

### 기타 로그 파일

```
logs/
├── decisions_btc_krw.jsonl   # BTC AI 결정
├── decisions_eth_krw.jsonl   # ETH AI 결정
├── decisions_xrp_krw.jsonl   # XRP AI 결정
├── decisions_ada_krw.jsonl   # ADA AI 결정
├── decisions_sol_krw.jsonl   # SOL AI 결정
├── symbol_decisions.jsonl    # AI 심볼 유니버스 선발 로그
├── orders.jsonl              # 주문 생성/체결
├── safety.jsonl              # 안전 이벤트
└── general.jsonl             # 기타 이벤트
```

---

## 11. Slack 알림 연동

긴급 상황(가격 급변, API 실패, 서킷 브레이커 작동)을 Slack으로 받을 수 있습니다.

### Slack Incoming Webhook 생성

1. https://api.slack.com/apps 에서 새 앱 생성
2. Incoming Webhooks 활성화
3. 채널 선택 후 Webhook URL 복사

### .env 설정

```env
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
```

---

## 12. FAQ / 트러블슈팅

### Q: 거래가 전혀 일어나지 않습니다

LLM이 HOLD를 반환하거나 스킵되고 있을 가능성이 높습니다. 웹 대시보드의 "AI 판단 로그" 탭에서 최근 결정을 확인하세요.

LLM 스킵 조건: 가격 변화 < 1% AND 마지막 LLM 호출 < 30분 전. 이 경우 action = HOLD이며 로그에 기록되지 않습니다.

### Q: `TRADING_SYMBOLS` 파싱 오류

```
pydantic_settings.exceptions.SettingsError: error parsing value for field "trading_symbols"
```

`.env` 파일에서 JSON 배열 형식을 확인하세요:

```env
# 올바른 형식
TRADING_SYMBOLS=["BTC/KRW","ETH/KRW","XRP/KRW","ADA/KRW","SOL/KRW"]
```

### Q: 업비트 API 호출이 실패합니다

- 모의투자 모드에서는 공개 API만 사용하므로 API 키가 불필요합니다
- 연속 5회 실패 시 Kill Switch가 자동 활성화됩니다
- 네트워크 문제인 경우 Kill Switch를 해제하고 재시작하세요

### Q: Live 모드가 시작되지 않습니다

아래를 모두 확인하세요:
1. `.env`에 `TRADING_MODE=live`
2. `RUN/live_mode_token.txt` 파일이 존재하고 내용에 `ARMED` 포함
3. `UPBIT_MASTER_KEY`가 올바른 값으로 설정
4. `data/upbit_keys.enc`가 존재

### Q: OAuth 토큰이 만료되었습니다

`data/.auth/openai-oauth.json`을 삭제하고 로컬에서 다시 로그인한 후 파일을 서버로 복사하세요.

### Q: 바이낸스 선물 모드가 작동하지 않습니다

아래를 확인하세요:
1. `.env.binance`에 `EXCHANGE=binance`, `FUTURES_ENABLED=true`
2. `QUOTE_CURRENCY=USDT`
3. 테스트넷 모드: `BINANCE_TESTNET=true` (API 키 없이 paper 모드 가능)
4. 실거래 시: `data/binance_keys.enc` 존재, `RUN-binance/live_mode_token.txt` 존재

### Q: 데이터를 초기화하고 싶습니다

```bash
# 로그 초기화
rm -f logs/*.jsonl

# Kill Switch 초기화
rm -f RUN/kill_switch

# 전체 초기화 (주의: API 키, OAuth 토큰도 삭제됨)
rm -rf data/ logs/ RUN/
```

---

## 빠른 시작 요약

```bash
# 1. 클론 및 빌드
git clone https://github.com/lovelyunsh/llm-coin-trader.git
cd llm-coin-trader
docker build -t coin-trader .

# 2. 설정
cp .env.example .env
# .env 편집: WEB_MASTER_CODE, LLM_ENABLED=true, LLM_AUTH_MODE=oauth 등

# 3. OAuth 토큰 생성 (로컬에서 1회)
# data/.auth/openai-oauth.json 생성 후 서버로 복사

# 4. 컨테이너 실행
docker run -d --name coin-trader -p 8932:8932 \
  -v "$(pwd)/.env:/app/.env:ro" \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/RUN:/app/RUN" \
  -v "$(pwd)/logs:/app/logs" \
  coin-trader

# 5. 브라우저에서 확인
open http://localhost:8932
```
