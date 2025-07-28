import pandas as pd
import duckdb
import os
from time import sleep
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DATE_FORMAT = "%Y-%m-%d"

# # # CSV 파일을 읽어와서 데이터프레임으로 로드
# df = pd.read_csv("data/korea_price_2025-07-18.csv")
# df1 = pd.read_csv("data/korea_price_updated_change.csv")


# # df와 df1의 change와 change_rate 컬럼을 비교
# # 먼저 두 데이터프레임을 병합 (ticker와 date를 기준으로)
# merged_df = pd.merge(df, df1, on=['ticker', 'date'], suffixes=('_df', '_df1'))

# # change 값이 다른 행의 개수 계산
# change_diff_count = (merged_df['change_df'] != merged_df['change_df1']).sum()

# # change_rate 값이 다른 행의 개수 계산
# change_rate_diff_count = (merged_df['change_rate_df'] != merged_df['change_rate_df1']).sum()

# # change 또는 change_rate 중 하나라도 다른 행의 개수 계산
# either_diff_count = ((merged_df['change_df'] != merged_df['change_df1']) | 
#                     (merged_df['change_rate_df'] != merged_df['change_rate_df1'])).sum()

# print(f"change 값이 다른 행의 개수: {change_diff_count}")
# print(f"change_rate 값이 다른 행의 개수: {change_rate_diff_count}")
# print(f"change 또는 change_rate 중 하나라도 다른 행의 개수: {either_diff_count}")
# print(f"전체 행의 개수: {len(merged_df)}")


# 티커별로 그룹화하여 각 그룹에 대해 변화율 계산
# df = pd.read_csv("data/korea_price_updated.csv").astype({"date": "str", "ticker": "str"})
# df_result = df.groupby('ticker').apply(lambda group: group.assign(
#     change=((group['close'] - group['close'].shift(1)).round(0)).fillna(0).astype(int),
#     change_rate=((group['close'] - group['close'].shift(1)) / group['close'].shift(1) * 100).round(2).fillna(0.0).astype(float)
# )).reset_index(drop=True).sort_values(['date', 'ticker'])

# # 결과를 새로운 CSV 파일로 저장
# df_result.to_csv("data/korea_price_updated_change.csv", index=False)



# import FinanceDataReader as fdr
# import yfinance as yf

# # 1) 코스피 종목 코드 리스트 가져오기 (예: '005930', '000660', ...)
# kospi_df = fdr.StockListing('KOSPI')
# kosdaq_df = fdr.StockListing('KOSDAQ')

# symbols  = kospi_df['Code'].tolist()
# symbols1  = kosdaq_df['Code'].tolist()

# # 2) yfinance용으로 '.KS' 접미사 추가
# tickers  = [f"{sym}.KS" for sym in symbols]
# tickers1  = [f"{sym}.KQ" for sym in symbols1]

# # 3) 기간 지정
# start_date = "2022-09-19"
# end_date   = "2025-07-19"

# # 4) 한번에 모두 다운로드
# #    group_by='ticker' 로 묶으면 MultiIndex(DataFrame) 형태로 반환됩니다.
# data = yf.download(
#     tickers + tickers1,
#     start=start_date,
#     end=end_date,
#     group_by='ticker',
#     threads=True          # 멀티스레드 다운로드 (속도 향상)
# )

# tidy = data.stack(level=0).rename_axis(['Date', 'Ticker']).reset_index()

# tidy = tidy[['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']]
# tidy = tidy.sort_values(['Date', 'Ticker']).reset_index(drop=True)

# # 결과 확인
# tidy.to_csv("data/korea_price_yfinance.csv", index=False)

# import pandas as pd
# import datetime

# df = pd.read_csv("data/korea_price_2025-07-15.csv").astype({"date": "str", "ticker": "str"})
# df1 = pd.read_csv("data/korea_index_2025-07-15.csv").astype({"date": "str"})

# df['date'] = pd.to_datetime(df['date']).dt.strftime("%Y-%m-%d")
# df1['date'] = pd.to_datetime(df1['date']).dt.strftime("%Y-%m-%d")

# df.to_csv("data/korea_price_2025-07-16.csv", index=False)
# df1.to_csv("data/korea_index_2025-07-16.csv", index=False)

# df = pd.read_csv("data/korea_price_2025-07-17.csv").astype({"date": "str", "ticker": "str"})
# df1 = pd.read_csv("data/korea_price_yfinance.csv").astype({"Date": "str"})

# df1 = df1.rename(columns={"Date": "date", "Ticker": "ticker", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})

# df = pd.merge(df, df1, on=['date', 'ticker'], how='left', suffixes=('', '_new'))

# col = ['open', 'high', 'low', 'close', 'volume']

# for c in col:
#     df[c] = df[f'{c}_new'].fillna(df[c]).astype(int)
#     df.drop(f'{c}_new', axis=1, inplace=True)

# df.to_csv("data/korea_price_updated.csv", index=False)

# df_index = pd.read_csv(f"{DATA_DIR}/korea_index_2025-07-22.csv").astype({"date": "str"})
# conn = duckdb.connect("data.duckdb")
# conn.execute("DROP TABLE IF EXISTS market_index")
# conn.register("index", df_index)
# conn.execute("CREATE TABLE market_index AS SELECT * FROM index")
# conn.close()

# df = pd.read_csv("data/korea_price_2025-07-22_for_duckdb.csv").astype({"date": "str", "ticker": "str"})
# conn = duckdb.connect("data.duckdb")
# conn.execute("DROP TABLE IF EXISTS data")
# conn.register("temp", df)
# conn.execute("CREATE TABLE data AS SELECT * FROM temp")
# conn.close()

import requests
import json
#각 참가팀의 API End-point.
URL = 'http://localhost:8000/agent'
#실제 평가시에는 미래에셋증권 평가용 API KEY 사용.
API_KEY = 'nv-0dbb7dd0f6234f0eb130279f6de64401U0yo'
REQUEST_ID = '9b2af2d213cc4f9d8d2ba4d5fb396416'
headers = {
    'Authorization': f'Bearer {API_KEY}',
    'X-NCP-CLOVASTUDIO-REQUEST-ID': f'{REQUEST_ID}'
}

# testing.py에서 JSON 파일 읽는 부분을 수정
# import json

# # 기존: json.loads('./tuning/simple_queries.json')
# # 수정:
# with open('./tuning/signal_queries.json', 'r', encoding='utf-8') as f:
#     read_json = json.load(f)

# for query in read_json[0:]:
#     params = {'question': "RSI가 80이상인 종목의 백테스팅 결과를 알려줘 익절은 7%고 손절은 5%야"}# query['input_data']['message']}
#     print(params)
#     response = requests.get(URL, headers=headers, params=params)
#     # 응답에서 실제 텍스트 내용 추출
#     try:
#         response_data = json.loads(response.text)
#         actual_output = response_data.get('text', '')
#     except:
#         actual_output = response.text
    
#     # expected_output과 비교하여 평가
#     expected = query['expected_output']
#     evaluation_type = query['evaluation_type']
    
#     # expected_output의 요소들이 actual_output에 포함되어 있는지 확인
#     if expected in actual_output:
#         match_percentage = 100.0
#     else:
#         # 간단한 부분 매칭으로 포함도 계산
#         expected_words = expected.replace(',', ' ').replace('(', ' ').replace(')', ' ').split()
#         matched_words = sum(1 for word in expected_words if word in actual_output)
#         match_percentage = (matched_words / len(expected_words)) * 100 if expected_words else 0
    
#     # 90% 이상일 때 패스 판정
#     if match_percentage >= 90:
#         result = "PASS"
#     else:
#         result = "FAIL"
    
#     print(f"질문: {query['input_data']['message']}")
#     print(f"기대값: {expected}")
#     print(f"실제값: {actual_output}")
#     print(f"매칭률: {match_percentage:.1f}%")
#     print(f"평가결과: {result}")
#     print("-" * 50)
#     sleep(5)

