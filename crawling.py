import requests
from datetime import date, timedelta
import pandas as pd
from time import sleep
from typing import Dict, List, Any
import os
import pandas_market_calendars as mcal
from fake_useragent import UserAgent
import glob

# 프로젝트 루트 경로 계산
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

DATE_FORMAT = "%Y-%m-%d"

REQ_URL = "http://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd"
STOCK_HEADERS = {
    "Host": "data.krx.co.kr",
    "Connection": "keep-alive",
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Accept-Language": "ko-KR,ko;q=0.9",
    "Accept-Encoding": "gzip, deflate",
    "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
    "Origin": "http://data.krx.co.kr",
    "Referer": "http://data.krx.co.kr/contents/MDC/MDI/mdiLoader/index.cmd?menuId=MDC0201020101",
    "User-Agent": UserAgent().random,
    "X-Requested-With": "XMLHttpRequest",
}
INDEX_HEADERS = {
    "Host": "data.krx.co.kr",
    "Connection": "keep-alive",
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Accept-Language": "ko-KR,ko;q=0.9",
    "Accept-Encoding": "gzip, deflate",
    "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
    "Origin": "http://data.krx.co.kr",
    "Referer": "http://data.krx.co.kr/contents/MDC/MDI/mdiLoader/index.cmd?menuId=MDC0201010101",
    "User-Agent": UserAgent().random,
    "X-Requested-With": "XMLHttpRequest",
}


class DataProcessor:
    def __init__(self):
        pass

    def get_trading_days(self, start_date: date, end_date: date) -> list[date]:
        # 한국 거래소 캘린더 가져오기
        exchange_calendar = mcal.get_calendar("XKRX")

        # 특정 기간의 영업일 조회
        trading_days = exchange_calendar.valid_days(
            start_date=start_date, end_date=end_date
        )
        return trading_days.strftime(DATE_FORMAT).tolist()

    def export_to_csv(self, dataframe: pd.DataFrame, filename: str):
        output_directory = os.path.join(PROJECT_ROOT, "data")
        os.makedirs(output_directory, exist_ok=True)

        # 절대 경로로 파일 저장
        output_path = os.path.join(output_directory, f"{filename}.csv")
        dataframe.to_csv(output_path, index=False)
        print(f"Saved data to {output_path}")

    def transform_price_data(
        self, raw_data: Dict, trading_date: str, market_type: str
    ) -> Dict[str, Any]:

        def _parse_numeric_value(value: str) -> int:
            return 0 if value == "-" else int(value.replace(",", ""))

        def _parse_float_value(value: str) -> float:
            return 0 if value == "-" else float(value.replace(",", ""))

        return {
            "date": trading_date,
            "ticker": raw_data["ISU_SRT_CD"],
            "ticker_name": raw_data["ISU_ABBRV"].replace(" ", ""),
            "ticker_market": f"{'KS' if market_type == 'kospi' else 'KQ'}",
            "open": _parse_numeric_value(raw_data["TDD_OPNPRC"]),
            "high": _parse_numeric_value(raw_data["TDD_HGPRC"]),
            "low": _parse_numeric_value(raw_data["TDD_LWPRC"]),
            "close": _parse_numeric_value(raw_data["TDD_CLSPRC"]),
            "change": _parse_numeric_value(raw_data["CMPPREVDD_PRC"]),
            "change_rate": _parse_float_value(raw_data["FLUC_RT"]),
            "volume": _parse_numeric_value(raw_data["ACC_TRDVOL"]),
            "trade_value": _parse_numeric_value(raw_data["ACC_TRDVAL"]),
            "cap": _parse_numeric_value(raw_data["MKTCAP"]),
            "shares_out": _parse_numeric_value(raw_data["LIST_SHRS"]),
        }

    def transform_financial_data(
        self, raw_data: Dict, trading_date: str, market_type: str
    ) -> Dict[str, Any]:

        def _parse_numeric_value(value: str) -> float:
            return 0 if value == "-" else float(value.replace(",", ""))

        return {
            "date": trading_date,
            "ticker": raw_data["ISU_SRT_CD"],
            "eps": _parse_numeric_value(raw_data["EPS"]),
            "per": _parse_numeric_value(raw_data["PER"]),
            "fwd_eps": _parse_numeric_value(raw_data["FWD_EPS"]),
            "fwd_per": _parse_numeric_value(raw_data["FWD_PER"]),
            "pbr": _parse_numeric_value(raw_data["PBR"]),
        }

    def transform_index_data(
        self, raw_data: Dict, trading_date: str
    ) -> Dict[str, Any]:
        def _parse_numeric_value(value: str) -> int:
            return 0 if value == "-" else int(value.replace(",", ""))

        def _parse_float_value(value: str) -> float:
            return 0 if value == "-" else float(value.replace(",", ""))

        return {
            "date": trading_date,
            "index_name": raw_data['IDX_NM'],
            "close": _parse_float_value(raw_data["CLSPRC_IDX"]),
            "change": _parse_float_value(raw_data["CMPPREVDD_IDX"]),
            "change_rate": _parse_float_value(raw_data["FLUC_RT"]),
            "open": _parse_float_value(raw_data["OPNPRC_IDX"]),
            "high": _parse_float_value(raw_data["HGPRC_IDX"]),
            "low": _parse_float_value(raw_data["LWPRC_IDX"]),
            "volume": _parse_numeric_value(raw_data["ACC_TRDVOL"]),
            "trade_value": _parse_numeric_value(raw_data["ACC_TRDVAL"]),
        }


class MarketDataCollector:
    def __init__(self):
        self.data_processor = DataProcessor()
        self.session = requests.Session()
        self._set_user_agent()

    def _set_user_agent(self):
        self.session.headers.update(STOCK_HEADERS)
        self.session.headers.update(INDEX_HEADERS)

    def _make_request_with_retry(
        self, payload: Dict, max_retries: int = 3
    ) -> List[Dict]:
        """재시도 로직이 포함된 요청 메서드"""
        for attempt in range(max_retries):
            try:
                response = self.session.post(REQ_URL, data=payload, timeout=10)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                print(f"Request failed (attempt {attempt + 1}/{max_retries}): {e}")
                self._set_user_agent()
                if attempt < max_retries - 1:
                    sleep_time = (attempt + 1) * 10  # 10초, 20초, 30초...
                    print(f"Waiting {sleep_time} seconds before retry...")
                    sleep(sleep_time)
                else:
                    raise e

    def fetch_daily_market_price_data(
        self, market_type: str, trading_date: str
    ) -> List[Dict]:
        market_id = "STK" if market_type == "kospi" else "KSQ"

        request_payload = {
            "bld": "dbms/MDC/STAT/standard/MDCSTAT01501",
            "mktId": market_id,
            "trdDd": trading_date.replace('-', ''),
            "share": "1",
            "money": "1",
            "csvxls_isNo": "false",
        }

        response_data = self._make_request_with_retry(request_payload)
        return response_data.get("OutBlock_1", [])

    def fetch_daily_market_financial_data(
        self, market_type: str, trading_date: str
    ) -> List[Dict]:
        request_payload = {
            "bld": "dbms/MDC/STAT/standard/MDCSTAT03501",
            "searchType": "1",
            "mktId": "STK" if market_type == "kospi" else "KSQ",
            "trdDd": trading_date.replace('-', ''),
            "csvxls_isNo": "false",
        }

        response_data = self._make_request_with_retry(request_payload)
        return response_data.get("output", [])

    def fetch_daily_index_data(self, index_name: str, trading_date: str) -> List[Dict]:
        request_payload = {
            "bld": "dbms/MDC/STAT/standard/MDCSTAT00101",
            "idxIndMidclssCd": "02" if index_name == "kospi" else "03",
            "trdDd": trading_date.replace('-', ''),
            "share": "1",
            "money": "1",
            "csvxls_isNo": "false",
        }

        response_data = self._make_request_with_retry(request_payload)
        return response_data.get("output", [])

    def collect_real_time_data(self) -> List[Dict]:
        price_records = []

        for market_type in ["kospi", "kosdaq"]:
            daily_price_data = self.fetch_daily_market_price_data(
                market_type, date.today().strftime('%Y%m%d')
            )
            price_records.extend(
                [
                    self.data_processor.transform_price_data(
                        record, date.today().strftime(DATE_FORMAT), market_type
                    )
                    for record in daily_price_data
                ]
            )

        dataframe = pd.DataFrame(price_records)
        return dataframe

    def collect_historical_index_data(self, order_date: date, start_date: date, end_date: date):
        trading_days = self.data_processor.get_trading_days(
            start_date, end_date
        )
        print(f"Total trading days to process: {len(trading_days)}")

        index_records = []
        for index, current_date in enumerate(trading_days):
            print(f"Processing {current_date} ({index + 1}/{len(trading_days)})")

            for index_name in ["kospi", "kosdaq"]:
                try:
                    daily_index_data = self.fetch_daily_index_data(
                        index_name, current_date
                    )
                    index_records.extend(
                        [
                            self.data_processor.transform_index_data(
                                record, current_date
                            )
                            for record in daily_index_data
                        ]
                    )

                    print(f"  {index_name}: {len(daily_index_data)} index records")

                except Exception as e:
                    print(f"Error processing {index_name} data for {current_date}: {e}")
                    continue

                # 각 시장 처리 후 잠시 대기
                sleep(1)

            if index > 0 and index % 100 == 0:
                print("Taking a longer break...")
                sleep(10)

                index_dataframe = pd.DataFrame(index_records)
                temp_df = index_dataframe.copy()
                self.data_processor.export_to_csv(
                    temp_df, f"korea_index_{order_date.strftime(DATE_FORMAT)}_temp"
                )
                self._set_user_agent()

        index_dataframe = pd.DataFrame(index_records)
        index_dataframe = index_dataframe[(index_dataframe['index_name'] == '코스피') | (index_dataframe['index_name'] == '코스닥')]
        index_dataframe = index_dataframe.sort_values(
            by=["date", "index_name"]
        ).reset_index(drop=True)
        self.data_processor.export_to_csv(
            index_dataframe, f"korea_index_{order_date.strftime(DATE_FORMAT)}"
        )

    def collect_historical_data(self, start_date: date, end_date: date):
        trading_days = self.data_processor.get_trading_days(start_date, end_date)
        print(f"Total trading days to process: {len(trading_days)}")

        price_records = []
        financial_records = []

        for index, current_date in enumerate(trading_days):
            print(f"Processing {current_date} ({index + 1}/{len(trading_days)})")

            for market_type in ["kospi", "kosdaq"]:
                try:
                    daily_price_data = self.fetch_daily_market_price_data(
                        market_type, current_date
                    )
                    daily_financial_data = self.fetch_daily_market_financial_data(
                        market_type, current_date
                    )

                    # 가격 데이터 변환
                    price_records.extend(
                        [
                            self.data_processor.transform_price_data(
                                record, current_date, market_type
                            )
                            for record in daily_price_data
                        ]
                    )

                    # 재무 데이터 변환
                    financial_records.extend(
                        [
                            self.data_processor.transform_financial_data(
                                record, current_date, market_type
                            )
                            for record in daily_financial_data
                        ]
                    )

                    print(
                        f"  {market_type}: {len(daily_price_data)} price records, {len(daily_financial_data)} financial records"
                    )

                except Exception as e:
                    print(
                        f"Error processing {market_type} data for {current_date}: {e}"
                    )
                    continue

                # 각 시장 처리 후 잠시 대기
                sleep(1)

            # 100번째마다 더 긴 휴식
            if index > 0 and index % 100 == 0:
                print("Taking a longer break...")
                sleep(10)

                # DataFrame으로 변환
                price_dataframe = pd.DataFrame(price_records)
                financial_dataframe = pd.DataFrame(financial_records)

                print(f"Price records: {len(price_dataframe)}")
                print(f"Financial records: {len(financial_dataframe)}")

                # date와 ticker 기준으로 merge
                temp_df = pd.merge(
                    price_dataframe,
                    financial_dataframe,
                    on=["date", "ticker"],
                    how="left",  # 가격 데이터를 기준으로 left join
                )
                temp_df = temp_df.sort_values(by=["date", "ticker"]).reset_index(
                    drop=True
                )
                self.data_processor.export_to_csv(
                    temp_df, f"korea_price_{date.today().strftime(DATE_FORMAT)}_temp"
                )
                self._set_user_agent()

        # DataFrame으로 변환
        price_dataframe = pd.DataFrame(price_records)
        financial_dataframe = pd.DataFrame(financial_records)

        print(f"Price records: {len(price_dataframe)}")
        print(f"Financial records: {len(financial_dataframe)}")

        # date와 ticker 기준으로 merge
        merged_dataframe = pd.merge(
            price_dataframe,
            financial_dataframe,
            on=["date", "ticker"],
            how="left",  # 가격 데이터를 기준으로 left join
        )
        # 정렬
        final_dataframe = merged_dataframe.sort_values(
            by=["date", "ticker"]
        ).reset_index(drop=True)
        print(f"Final merged records: {len(final_dataframe)}")

        self.data_processor.export_to_csv(
            final_dataframe, f"korea_price_{date.today().strftime(DATE_FORMAT)}"
        )

    def collect_add_time_data(self, order_date: date, start_date: date, end_date: date):
        trading_days = self.data_processor.get_trading_days(start_date, end_date)
        print(f"Total trading days to process: {len(trading_days)}")

        price_records = []
        financial_records = []

        for index, current_date in enumerate(trading_days):
            print(f"Processing {current_date} ({index + 1}/{len(trading_days)})")

            for market_type in ["kospi", "kosdaq"]:
                try:
                    daily_price_data = self.fetch_daily_market_price_data(
                        market_type, current_date
                    )
                    daily_financial_data = self.fetch_daily_market_financial_data(
                        market_type, current_date
                    )

                    # 가격 데이터 변환
                    price_records.extend(
                        [
                            self.data_processor.transform_price_data(
                                record, current_date, market_type
                            )
                            for record in daily_price_data
                        ]
                    )

                    # 재무 데이터 변환
                    financial_records.extend(
                        [
                            self.data_processor.transform_financial_data(
                                record, current_date, market_type
                            )
                            for record in daily_financial_data
                        ]
                    )

                    print(
                        f"  {market_type}: {len(daily_price_data)} price records, {len(daily_financial_data)} financial records"
                    )

                except Exception as e:
                    print(
                        f"Error processing {market_type} data for {current_date}: {e}"
                    )
                    continue

                # 각 시장 처리 후 잠시 대기
                sleep(2)

            # 100번째마다 더 긴 휴식
            if index > 0 and index % 10 == 0:  # 100에서 10으로 줄임
                print("Taking a longer break...")
                sleep(10)

        # DataFrame으로 변환
        price_dataframe = pd.DataFrame(price_records)
        financial_dataframe = pd.DataFrame(financial_records)

        print(f"Price records: {len(price_dataframe)}")
        print(f"Financial records: {len(financial_dataframe)}")

        # date와 ticker 기준으로 merge
        merged_dataframe = pd.merge(
            price_dataframe,
            financial_dataframe,
            on=["date", "ticker"],
            how="left",  # 가격 데이터를 기준으로 left join
        )

        # data 디렉토리에서 가장 최신 korea_price 파일 찾기
        price_pattern = os.path.join(DATA_DIR, "korea_price_????-??-??.csv")
        price_files = glob.glob(price_pattern)
        
        print(f"발견된 price 파일들: {price_files}")
        
        if not price_files:
            print("❌ korea_price 데이터 파일을 찾을 수 없습니다.")
            print(f"다음 디렉토리를 확인해주세요: {DATA_DIR}")
            raise FileNotFoundError("주식 데이터 파일이 존재하지 않습니다.")
        
        # 파일명에서 날짜 추출하여 정렬 (가장 최신 파일 선택)
        def extract_date_from_filename(filepath):
            filename = os.path.basename(filepath)
            # korea_price_2025-07-18.csv -> 2025-07-18
            date_part = filename.replace("korea_price_", "").replace(".csv", "")
            return date_part
        
        # 날짜별로 정렬하여 가장 최신 파일 선택
        latest_file = max(price_files, key=extract_date_from_filename)
        latest_date = extract_date_from_filename(latest_file)
        
        print(f"가장 최신 파일 선택: {latest_file} (날짜: {latest_date})")

        df = pd.read_csv(latest_file, dtype={"date": "str", "ticker": "str"})

        # 정렬
        final_dataframe = pd.concat([df, merged_dataframe])
        final_dataframe = final_dataframe.sort_values(
            by=["date", "ticker"]
        ).reset_index(drop=True)
        print(f"Final merged records: {len(final_dataframe)}")

        self.data_processor.export_to_csv(
            final_dataframe, f"korea_price_{order_date.strftime(DATE_FORMAT)}"
        )
    def collect_add_index_data(self, order_date: date, start_date: date, end_date: date):
        trading_days = self.data_processor.get_trading_days(
            start_date, end_date
        )
        print(f"Total trading days to process: {len(trading_days)}")

        index_records = []
        for index, current_date in enumerate(trading_days):
            print(f"Processing {current_date} ({index + 1}/{len(trading_days)})")

            for index_name in ["kospi", "kosdaq"]:
                try:
                    daily_index_data = self.fetch_daily_index_data(
                        index_name, current_date
                    )
                    index_records.extend(
                        [
                            self.data_processor.transform_index_data(
                                record, current_date
                            )
                            for record in daily_index_data
                        ]
                    )

                    print(f"  {index_name}: {len(daily_index_data)} index records")

                except Exception as e:
                    print(f"Error processing {index_name} data for {current_date}: {e}")
                    continue

                # 각 시장 처리 후 잠시 대기
                sleep(1)

            if index > 0 and index % 100 == 0:
                print("Taking a longer break...")
                sleep(10)

                index_dataframe = pd.DataFrame(index_records)
                temp_df = index_dataframe.copy()
                self.data_processor.export_to_csv(
                    temp_df, f"korea_index_{order_date.strftime(DATE_FORMAT)}_temp"
                )
                self._set_user_agent()

        index_dataframe = pd.DataFrame(index_records)
        index_dataframe = index_dataframe[(index_dataframe['index_name'] == '코스피') | (index_dataframe['index_name'] == '코스닥')]
        index_dataframe = index_dataframe.sort_values(
            by=["date", "index_name"]
        ).reset_index(drop=True)

        # data 디렉토리에서 가장 최신 korea_price 파일 찾기
        price_pattern = os.path.join(DATA_DIR, "korea_index_????-??-??.csv")
        price_files = glob.glob(price_pattern)
        
        print(f"발견된 index 파일들: {price_files}")
        
        if not price_files:
            print("❌ korea_index 데이터 파일을 찾을 수 없습니다.")
            print(f"다음 디렉토리를 확인해주세요: {DATA_DIR}")
            raise FileNotFoundError("주식 데이터 파일이 존재하지 않습니다.")
        
        # 파일명에서 날짜 추출하여 정렬 (가장 최신 파일 선택)
        def extract_date_from_filename(filepath):
            filename = os.path.basename(filepath)
            # korea_index_2025-07-18.csv -> 2025-07-18
            date_part = filename.replace("korea_index_", "").replace(".csv", "")
            return date_part
        
        # 날짜별로 정렬하여 가장 최신 파일 선택
        latest_file = max(price_files, key=extract_date_from_filename)
        latest_date = extract_date_from_filename(latest_file)
        
        print(f"가장 최신 파일 선택: {latest_file} (날짜: {latest_date})")

        df = pd.read_csv(latest_file, dtype={"date": "str"})
        
        final_dataframe = pd.concat([df, index_dataframe])
        final_dataframe = final_dataframe.sort_values(
            by=["date", "index_name"]
        ).reset_index(drop=True)

        self.data_processor.export_to_csv(
            final_dataframe, f"korea_index_{order_date.strftime(DATE_FORMAT)}"
        )

if __name__ == "__main__":
    collector = MarketDataCollector()
    collector.collect_add_time_data(date(2025, 7, 22) ,date(2025, 7, 21), date(2025, 7, 22))
    collector.collect_add_index_data(date(2025, 7, 22) ,date(2025, 7, 21), date(2025, 7, 22))
