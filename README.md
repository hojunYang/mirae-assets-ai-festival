## API 입력값 / 호출값 작성

API URL = http://49.50.132.254/agent

### API 입력 헤더
- Authorization: Bearer {CLOVA Studio API Key}
- X-NCP-CLOVASTUDIO-REQUEST-ID: {Request ID}

#### 필수 Query 파라미터
- **question**: 주식 관련 질문 내용
  - 형식: URL 인코딩된 문자열
  - 예시: `question=삼성전자%20주가` (삼성전자 주가)

### API 출력 헤더
- Content-Type: application/json
- **응답 형태**: LLM이 생성한 자연어 응답을 텍스트로 직접 반환(response.text)

### API 호출 예시

**Curl 예시:**
```bash
curl -X GET "http://49.50.132.254/agent?question=질문내용" \
  -H "Authorization: Bearer your-api-key" \
  -H "X-NCP-CLOVASTUDIO-REQUEST-ID: your-request-id"
```

**Python 예시:**
```python
import requests

# API 설정
base_url = "http://49.50.132.254"
headers = {
    "Authorization": "Bearer your-api-key",
    "X-NCP-CLOVASTUDIO-REQUEST-ID": "your-request-id"
}

params = {"question": {질문내용}"}
response = requests.get("http://49.50.132.254/agent", params=params, headers=headers)
print(response.text) # LLM 응답 텍스트 출력
```

## 시스템 설계 및 아키텍처

### 성능 최적화 전략
#### 1. 데이터 사전 적재 (Pre-loading)
- **적재 범위**: 2023-01-01부터 현재까지의 모든 주식 데이터
- **데이터 소스**: KRX 크롤링 데이터에 yfinance API를 통해 **적재 항목**를 덮어 씌워 작업함
- **적재 항목**:
  - 시가 (Open)
  - 종가 (Close) 
  - 고가 (High)
  - 저가 (Low)
  - 거래량 (Volume)
  - 상승률 (Change Rate)

#### 2. 보조지표 사전 계산 (Pre-computed Indicators)
실시간 계산 부하를 줄이기 위해 다음 보조지표들을 배치 처리로 미리 계산하여 저장:

- **볼린저 밴드** (Bollinger Bands)
  - 상단선, 중간선, 하단선
  - 20일 이동평균 기준, 편차 값 2
  
- **골든 크로스** (Golden Cross)
  - 단기 이동평균선과 장기 이동평균선 교차점
  - 5일, 20일 이동평균 조합

- **데드 크로스** (Golden Cross)
  - 단기 이동평균선과 장기 이동평균선 교차점
  - 5일, 20일 이동평균 조합

- **신고가/신저가** (New High/Low)
  - 52주 신고가/신저가 여부
  - 일별 갱신

- **RSI** (Relative Strength Index)
  - 14일 기준 RSI 값
  - 과매수/과매도 구간 사전 판별

#### 3. 고성능 데이터베이스 엔진
- **DuckDB 채택**: Pandas 대비 약 3-5배 빠른 쿼리 성능

#### 4. 배치 처리 시스템
- **스케줄링**: 21시 기준 일 단위 데이터 갱신 및 지표 계산
- **서버정지**: 배치 작업이 도는 동안 서버를 사용할 수 없음(약 5분 소요)

### 편의기능

#### 1. 멀티턴 대화 지원 (Multi-turn Conversation)
- **세션 식별**: API Key + Request ID 조합으로 고유한 대화 세션 생성
- **대화 히스토리**: 최근 6~15개 대화 내용 자동 저장
- **컨텍스트 유지**: 이전 질문과 답변을 기억하여 연속적인 대화 가능

#### 2. 실시간 시그널 종목 검색 일부 지원
- **KRX 실시간 크롤링**: 사용자의 요청에 따라 실시간으로 한국거래소 데이터를 크롤링
- **실시간 랭킹**: 당일 상승률, 거래량 상위 종목, 전일대비 거래량 상승 등 단순 시그널 검색 지원


## 핵심 특화기능: AI 기반 투자전략 백테스팅 시스템

### 1. 문제 정의 및 배경 분석

#### 현황 진단
개인투자자들은 다양한 투자 지표와 전략을 활용하고 있으나, 다음과 같은 문제점이 관찰됨.

- **지표 맹신 현상**: 검증되지 않은 기술적 지표에 대한 과도한 의존
- **정성적 판단 위주**: 감정이나 직감에 기반한 투자 의사결정
- **정량적 분석 부재**: 체계적인 성과 검증 도구의 접근성 한계

#### 근본 원인 분석
일반 투자자의 정량적 분석 장벽
- **기술적 복잡성**: 프로그래밍 및 데이터 분석 역량 부족
- **도구 접근성**: 전문 백테스팅 도구의 높은 진입장벽
- **시간적 제약**: 과거 데이터 수집 및 분석에 소요되는 시간

### 2. 시스템 플로우
```mermaid
사용자 전략 입력 → LLM SQL 생성 → 조건 필터링 → 백테스팅 엔진 → 성과 분석 → 결과 리포트
```

### 3. 구현 세부사항

#### 3.1 전략 정의 및 SQL 변환
```python
# 사용자 입력 예시
"RSI 30 이하에서 매수, 익절률 10%, 손절률 5%"

# LLM이 생성하는 SQL
SELECT date, ticker, 1 as buy_signal 
FROM data
WHERE rsi <= 30
```

#### 3.2 백테스팅 엔진 핵심 로직
```python
def backtest_strategy(signal_df, profit_rate=0.10, stoploss_rate=0.05):
    """
    매일 종가 기준 매수/매도 시뮬레이션
    - 익절가 도달: take_profit_price = buy_price * (1 + profit_rate)
    - 손절가 도달: stop_loss_price = buy_price * (1 - stoploss_rate)
    (익절가 도달을 먼저 처리함)
    """
    for date in trading_dates:
        # 당일 고가가 익절가 달성 시 익절 처리
        # 당일 저가가 손절가 달성 시 손절 처리
```

#### 3.3 성과 측정 지표
- **CAGR** (연복리수익률): `((최종값/초기값)^(1/기간) - 1) × 100`
- **승률**: `수익거래 건수 / 총거래 건수 × 100`
- **평균 거래당 손익**: `총손익 / 총거래 건수`
- **최대 보유 기간**: 포지션별 보유일수 분석

### 4. 기대 효과 및 가치 제안
- **투자 교육 효과**: 정량적 사고방식 습득 지원
- **전략 검증 정확도**: 실제 시장 데이터 기반 신뢰성 확보
- **분석 시간 단축**

### 5. 실제 백테스팅 예시
**종목이 많아질수록 처리 시간이 길어져, 적절한 전략이 필요함**
전략: "RSI 20 이하 매수, 익절 15%, 손절 8%"