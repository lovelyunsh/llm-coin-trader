# 코인 자동거래 시스템 - A to Z 시작 가이드

## 목차

1. [사전 요구사항](#1-사전-요구사항)
2. [프로젝트 설치](#2-프로젝트-설치)
3. [환경 설정 (.env)](#3-환경-설정)
4. [시스템 점검 (selftest)](#4-시스템-점검)
5. [모의투자로 시작하기](#5-모의투자로-시작하기)
6. [웹 대시보드 띄우기](#6-웹-대시보드-띄우기)
7. [LLM 자문 기능 활성화](#7-llm-자문-기능-활성화)
8. [Slack 알림 연동](#8-slack-알림-연동)
9. [실거래 전환하기](#9-실거래-전환하기)
10. [긴급 정지 (Kill Switch)](#10-긴급-정지)
11. [로그 확인 방법](#11-로그-확인-방법)
12. [리스크 파라미터 조정](#12-리스크-파라미터-조정)
13. [FAQ / 트러블슈팅](#13-faq--트러블슈팅)

---

## 1. 사전 요구사항

| 항목 | 요구 |
|------|------|
| Python | 3.11 이상 |
| OS | macOS / Linux (Windows WSL 가능) |
| 패키지 관리 | Poetry |
| 거래소 계정 | 업비트 (API 키 발급 필요, 실거래 시) |

모의투자(Paper)는 거래소 계정 없이 바로 시작할 수 있습니다.

---

## 2. 프로젝트 설치

```bash
# 1) 저장소 클론
git clone <repository-url> coin-trader
cd coin-trader

# 2) 의존성 설치
poetry install

# 3) 가상환경 활성화
poetry shell

# 4) 설치 확인
trader --help
```

정상 출력:
```
Usage: trader [OPTIONS] COMMAND [ARGS]...

  Coin Trader - Safety-first automated crypto trading system.

Commands:
  encrypt-keys  Encrypt API keys for secure storage.
  kill          Activate kill switch - halt all trading.
  run           Start the trading bot.
  selftest      Run basic self-test to verify system integrity.
  web           Start the web dashboard server.
```

---

## 3. 환경 설정

### 3.1 .env 파일 생성

```bash
cp .env.example .env
```

### 3.2 모의투자 최소 설정

`.env` 파일을 열어서 확인합니다. 모의투자는 기본값 그대로 바로 실행 가능합니다:

```env
# 모의투자 모드 (기본값, 수정 불필요)
TRADING_MODE=paper

# 거래소 (기본값)
ENABLED_EXCHANGES=upbit

# 거래할 코인 (원하는 심볼로 수정)
TRADING_SYMBOLS=BTC/KRW,ETH/KRW

# 시세 조회 주기 (초)
MARKET_DATA_INTERVAL_SEC=60
```

이것만으로 모의투자를 시작할 수 있습니다. API 키는 필요하지 않습니다.

### 3.3 디렉토리 구조 (자동 생성됨)

```
coin-trader/
├── data/              # DB, 암호화 키 (자동 생성)
├── logs/              # 로그 파일 (자동 생성)
├── RUN/               # 킬 스위치, 토큰 파일 (자동 생성)
└── .env               # 환경 설정 (직접 생성)
```

---

## 4. 시스템 점검

실행 전 반드시 셀프테스트를 수행합니다:

```bash
trader selftest
```

정상 출력:
```
Running self-test...
  [OK] Config loads (paper mode)
  [OK] State store initializes
  [OK] Risk limits immutable
  [OK] Kill switch works
  [OK] Sensitive data redaction
  [OK] Domain models validate

OK - All self-tests passed.
```

하나라도 실패하면 `.env` 설정을 다시 확인하세요.

---

## 5. 모의투자로 시작하기

### 5.1 단일 실행 (테스트)

먼저 한 번만 실행해서 정상 동작을 확인합니다:

```bash
trader run --mode paper --once
```

이 명령은:
1. 업비트에서 BTC/KRW, ETH/KRW 시세를 가져옵니다
2. 기술적 분석(EMA, RSI, ATR)으로 매매 시그널을 생성합니다
3. 가상 잔고 100만원으로 모의 매매를 수행합니다
4. 한 사이클 실행 후 종료합니다

### 5.2 연속 실행

```bash
trader run --mode paper
```

60초(기본) 간격으로 계속 거래를 반복합니다. `Ctrl+C`로 종료합니다.

### 5.3 실행 중 확인할 것

```
Starting trader in paper mode...
```

이 메시지가 나오면 정상입니다. 터미널에 JSON 형태의 로그가 출력됩니다:

```json
{"event": "paper_mode_active", "exchange": "upbit", "level": "info", "ts": "2025-..."}
{"event": "trader_started", "mode": "paper", "symbols": ["BTC/KRW", "ETH/KRW"], ...}
```

---

## 6. 웹 대시보드 띄우기

별도 터미널에서 실행합니다:

```bash
trader web --port 8000
```

브라우저에서 `http://localhost:8000` 접속:

| 화면 | 내용 |
|------|------|
| 시스템 상태 | 거래 모드, 킬 스위치, 가동 시간, 심볼 |
| 거래 제어 | 모의투자 시작/중지, 실거래 시작/중지 버튼 |
| 손익 현황 | 총 자산, 총 손익, 손익률, 초기 잔고 |
| 보유 자산 | 통화별 잔고 |
| 보유 포지션 | 심볼, 수량, 평균단가, 현재가, 미실현 손익 |
| 미체결 주문 | 주문 상태 실시간 추적 |
| 안전 이벤트 | 가격 급변, API 실패 등 |
| 리스크 결정 | 주문 승인/거부 이력 |

대시보드는 5초/15초 간격으로 자동 갱신됩니다.

> 웹 대시보드에서도 거래 시작/중지, 킬 스위치 조작이 가능합니다.

---

## 7. LLM 자문 기능 활성화

LLM은 매매 판단에 대한 참고 자문만 제공합니다. 주문 실행에 직접 영향을 주지 않습니다.

### 방법 A: API Key 방식 (OpenAI / Anthropic)

`.env`:
```env
LLM_ENABLED=true
LLM_AUTH_MODE=api_key
LLM_PROVIDER=openai
LLM_API_KEY=sk-your-api-key-here
LLM_MODEL=gpt-4o-mini
```

### 방법 B: OAuth 방식 (ChatGPT 계정 로그인)

API 키 없이 ChatGPT 계정으로 로그인하여 사용합니다.

`.env`:
```env
LLM_ENABLED=true
LLM_AUTH_MODE=oauth
LLM_OAUTH_MODEL=gpt-5.2-codex
LLM_OAUTH_OPEN_BROWSER=true
```

최초 실행 시:
1. 브라우저가 자동으로 열립니다
2. OpenAI 계정으로 로그인합니다
3. 토큰이 `data/.auth/openai-oauth.json`에 저장됩니다
4. 이후 자동으로 토큰을 갱신하므로 재로그인 불필요

> 서버 환경(헤드리스)에서는 `LLM_OAUTH_OPEN_BROWSER=false`로 설정하면 토큰이 없을 때 LLM이 비활성화됩니다.

### LLM 결과 확인

```bash
# 로그에서 LLM 자문 기록 확인
cat logs/decisions.jsonl | grep llm_advice
```

출력 예시:
```json
{
  "type": "llm_advice",
  "symbol": "BTC/KRW",
  "action": "HOLD",
  "confidence": "0.8",
  "reasoning": "RSI 중립 구간이며 뚜렷한 추세가 없어 관망 권장",
  "risk_notes": "변동성 확대 가능성 주의"
}
```

---

## 8. Slack 알림 연동

긴급 상황(가격 급변, API 실패, 서킷 브레이커 작동)을 Slack으로 받을 수 있습니다.

### 8.1 Slack Incoming Webhook 생성

1. https://api.slack.com/apps 에서 새 앱 생성
2. Incoming Webhooks 활성화
3. 채널 선택 후 Webhook URL 복사

### 8.2 .env 설정

```env
SLACK_WEBHOOK_URL=<your-slack-webhook-url>
```

### 8.3 알림 종류

| 심각도 | 상황 | 이모지 |
|--------|------|--------|
| critical | API 연속 실패 → 킬 스위치 자동 활성화 | :sos: |
| critical | 서킷 브레이커 발동 | :sos: |
| high | 가격 급변 (10%↑) | :rotating_light: |
| medium | 스프레드 이상 (5%↑) | :warning: |

---

## 9. 실거래 전환하기

> **경고**: 실거래는 실제 자산이 거래됩니다. 반드시 모의투자로 충분히 테스트한 후 전환하세요.

### 9.1 업비트 API 키 발급

1. https://upbit.com/mypage/open_api_management 접속
2. API 키 발급 (출금 권한은 비활성화 권장)
3. Access Key와 Secret Key 복사

### 9.2 API 키 암호화 저장

```bash
trader encrypt-keys --exchange upbit --master-key
```

프롬프트 순서:
```
Master key: (마스터 키 입력 - 기억해야 합니다)
Confirm master key: (재입력)
API Key: (업비트 Access Key)
API Secret: (업비트 Secret Key)
Keys encrypted and saved to data/upbit_keys.enc
```

> API 키는 암호화되어 `data/upbit_keys.enc`에 저장됩니다. 원본 키는 어디에도 평문으로 저장되지 않습니다.

### 9.3 .env 설정

```env
TRADING_MODE=live
UPBIT_MASTER_KEY=your-master-key-here
```

### 9.4 Live 모드 잠금 해제 (2단계 확인)

```bash
# 1단계: 토큰 파일 생성
mkdir -p RUN
echo "ARMED" > RUN/live_mode_token.txt
```

### 9.5 실거래 시작

```bash
# 먼저 단일 실행으로 테스트
trader run --mode live --once
```

연속 실행 시 2단계 확인이 요구됩니다:
```bash
trader run --mode live
```

```
WARNING: You are about to start LIVE trading.
Type 'I_UNDERSTAND_LIVE_TRADING' to confirm: I_UNDERSTAND_LIVE_TRADING
Starting trader in live mode...
```

### 9.6 실거래 체크리스트

실거래 시작 전 확인할 사항:

- [ ] 모의투자에서 충분히 테스트했는가? (최소 1주일 권장)
- [ ] API 키 권한에 출금이 비활성화되어 있는가?
- [ ] 리스크 파라미터를 검토했는가? (`MAX_POSITION_SIZE_PCT` 등)
- [ ] Slack 알림이 설정되어 있는가?
- [ ] `trader selftest`가 모두 통과하는가?
- [ ] 킬 스위치 동작을 테스트했는가?

---

## 10. 긴급 정지

문제 발생 시 즉시 모든 거래를 중단합니다.

### 방법 1: CLI

```bash
# 즉시 정지 (새 주문 차단)
trader kill

# 즉시 정지 + 미체결 주문 취소
trader kill --close-positions
```

### 방법 2: 웹 대시보드

`http://localhost:8000` → **킬 스위치 활성화** 버튼 클릭

### 방법 3: 파일 생성

```bash
echo "emergency stop" > RUN/kill_switch
```

### 방법 4: 프로세스 종료

```bash
# SIGTERM/SIGINT → 자동으로 킬 스위치 활성화
kill <PID>
# 또는 Ctrl+C
```

### 킬 스위치 해제

```bash
# 파일 삭제로 해제
rm RUN/kill_switch
```

또는 웹 대시보드 → **킬 스위치 해제** 버튼

---

## 11. 로그 확인 방법

### 실시간 로그

```bash
# 전체 콘솔 로그 (trader run 실행 중)
# → 터미널에 JSON 출력

# 주문 내역만 보기
tail -f logs/orders.jsonl

# 안전 이벤트만 보기
tail -f logs/safety.jsonl

# 리스크 결정 (승인/거부) 보기
tail -f logs/decisions.jsonl
```

### 로그 파일 구조

```
logs/
├── decisions.jsonl   # 리스크 결정 + LLM 자문
├── orders.jsonl      # 주문 생성/체결
├── fills.jsonl       # 체결 상세
├── balances.jsonl    # 잔고 스냅샷
├── safety.jsonl      # 가격 급변, API 실패 등
└── general.jsonl     # 기타 이벤트
```

### DB 직접 조회

```bash
sqlite3 data/coin_trader.db

-- 최근 주문 10건
SELECT json_extract(data, '$.symbol'), json_extract(data, '$.side'), json_extract(data, '$.status')
FROM orders ORDER BY updated_at DESC LIMIT 10;

-- 리스크 거부 사유
SELECT reason, timestamp FROM decisions_log WHERE decision = 'rejected' ORDER BY timestamp DESC LIMIT 10;

-- 안전 이벤트
SELECT event_type, severity, description FROM safety_events ORDER BY timestamp DESC LIMIT 10;
```

---

## 12. 리스크 파라미터 조정

`.env`에서 설정합니다. 모든 값은 보수적 기본값이 적용되어 있습니다.

### 주요 파라미터

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `MAX_POSITION_SIZE_PCT` | 10 | 1회 주문 최대 비중 (총 자산의 %) |
| `MAX_POSITIONS` | 5 | 동시 보유 가능 코인 종류 |
| `DAILY_MAX_DRAWDOWN_PCT` | 5 | 일일 최대 허용 손실 (%) |
| `STOP_LOSS_PCT` | -3 | 손절 기준 (%) |
| `TAKE_PROFIT_PCT` | 10 | 익절 기준 (%) |
| `TRAILING_STOP_PCT` | 5 | 트레일링 스탑 (최고점 대비 %) |
| `MAX_ORDERS_PER_SECOND` | 1 | 초당 최대 주문 수 |
| `MAX_ORDERS_PER_DAY` | 100 | 일일 최대 주문 수 |
| `CIRCUIT_BREAKER_THRESHOLD_PCT` | 15 | 서킷 브레이커 발동 기준 (%) |

### 보수적 설정 예시 (초보자 권장)

```env
MAX_POSITION_SIZE_PCT=5
MAX_POSITIONS=3
DAILY_MAX_DRAWDOWN_PCT=3
MARKET_DATA_INTERVAL_SEC=120
```

### 적극적 설정 예시 (숙련자)

```env
MAX_POSITION_SIZE_PCT=15
MAX_POSITIONS=8
DAILY_MAX_DRAWDOWN_PCT=8
MARKET_DATA_INTERVAL_SEC=30
TRADING_SYMBOLS=BTC/KRW,ETH/KRW,XRP/KRW,SOL/KRW
```

---

## 13. FAQ / 트러블슈팅

### Q: `trader run` 실행 시 아무 거래도 안 됩니다

정상입니다. ConservativeStrategy는 5가지 조건이 모두 충족되어야 매수합니다:
- 상승 추세 (Fast EMA > Slow EMA)
- 가격이 Slow EMA 위
- 저변동성 (ATR/가격 < 3%)
- RSI 30~65
- 거래량이 평균 이상

시장 상황에 따라 며칠간 HOLD만 나올 수 있습니다. 이는 "안전 우선" 설계입니다.

### Q: `Settings` 로드 실패

```
pydantic_settings.exceptions.SettingsError: error parsing value for field "trading_symbols"
```

`.env` 파일에서 `TRADING_SYMBOLS` 값 형식을 확인하세요:
```env
# 올바른 형식
TRADING_SYMBOLS=["BTC/KRW","ETH/KRW"]

# 또는 .env를 사용하지 않으면 기본값 적용
# (해당 줄을 삭제하거나 주석 처리)
```

### Q: 업비트 API 호출이 실패합니다

- 모의투자 모드에서는 공개 API만 사용하므로 API 키가 불필요합니다
- API 키 없이 호출 가능한 엔드포인트: 시세 조회, 캔들 데이터
- 연속 5회 실패 시 킬 스위치가 자동 활성화됩니다
- 네트워크 문제인 경우 킬 스위치를 해제하고 재시작하세요

### Q: Live 모드가 시작되지 않습니다

```
ERROR: LIVE MODE NOT ARMED
```

아래 3가지를 모두 확인하세요:
1. `.env`에 `TRADING_MODE=live`
2. `RUN/live_mode_token.txt` 파일이 존재하고 내용에 `ARMED` 포함
3. `UPBIT_MASTER_KEY`가 올바른 값으로 설정

### Q: OAuth 로그인 브라우저가 열리지 않습니다

서버 환경에서는:
```env
LLM_OAUTH_OPEN_BROWSER=false
```

로컬에서 먼저 한 번 로그인하여 토큰을 생성한 후, `data/.auth/openai-oauth.json` 파일을 서버로 복사하세요.

### Q: 킬 스위치가 해제되지 않습니다

```bash
# 수동으로 파일 삭제
rm -f RUN/kill_switch

# 재시작
trader run --mode paper
```

### Q: 데이터를 초기화하고 싶습니다

```bash
# DB 초기화
rm -f data/coin_trader.db

# 로그 초기화
rm -f logs/*.jsonl

# 킬 스위치 초기화
rm -f RUN/kill_switch

# 전체 초기화
rm -rf data/ logs/ RUN/
```

---

## 빠른 시작 요약

```bash
# 1. 설치
poetry install && poetry shell

# 2. 설정
cp .env.example .env

# 3. 점검
trader selftest

# 4. 모의투자 시작
trader run --mode paper

# 5. (별도 터미널) 대시보드
trader web --port 8000

# 6. 브라우저에서 확인
open http://localhost:8000
```
