import pandas as pd
import os
import duckdb
import crawling
from datetime import date, timedelta, datetime
import yfinance as yf
import pandas_market_calendars as mcal

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DATE_FORMAT = "%Y-%m-%d"

class StockDataProcessor:
    def __init__(self, order_date: date):
        self.collector = crawling.MarketDataCollector()

        self.order_date = order_date
        self.three_days_before = (self.order_date - timedelta(days=3)).strftime(DATE_FORMAT)

    def _get_yfinance_data(self, ticker: str, ticker_market: str, start_date: str, end_date: str):
        df = yf.download(f"{ticker}.{ticker_market}", start=start_date, end=end_date)
        return df
    
    def _is_trading_days(self, date: str) -> bool:
        # 한국 거래소 캘린더 가져오기
        exchange_calendar = mcal.get_calendar("XKRX")

        # 더 이상 사용되지 않는 시간 항목을 안전하게 제거
        try:
            if 'break_start' in exchange_calendar.regular_market_times:
                exchange_calendar.remove_time('break_start')
        except (AttributeError, KeyError):
            pass
        
        try:
            if 'break_end' in exchange_calendar.regular_market_times:
                exchange_calendar.remove_time('break_end')
        except (AttributeError, KeyError):
            pass

        date_obj = datetime.strptime(date, DATE_FORMAT)
        # 특정 기간의 영업일 조회
        trading_days = exchange_calendar.valid_days(
            start_date=date_obj, end_date=date_obj
        )
        return True if not trading_days.empty else False
    
    def _apply_all_flags(self, subdf):
        subdf = self._set_previous_day_data(subdf)
        subdf = self._set_high_flag(subdf)
        subdf = self._set_low_flag(subdf)
        subdf = self._set_ma_cross_flag(subdf)
        subdf = self._set_rsi(subdf)
        subdf = self._set_bollinger_bands_flag(subdf)
        return subdf
        
    def process_data(self, start_date: date, end_date: date):

        if not self._is_trading_days(self.order_date.strftime(DATE_FORMAT)):
            print(f"오늘은 영업일이 아닙니다. {self.order_date.strftime(DATE_FORMAT)}")
            return
        
        self.collector.collect_add_time_data(self.order_date, start_date, end_date)
        df = pd.read_csv(f"{DATA_DIR}/korea_price_{self.order_date.strftime(DATE_FORMAT)}.csv").astype({"date": "str", "ticker": "str"})

        df = (
            df
            .groupby("ticker", group_keys=False)
            .apply(self._apply_all_flags)
        ).reset_index(drop=True).sort_values(by=["date", "ticker"])

        df.to_csv(f"{DATA_DIR}/korea_price_{self.order_date.strftime(DATE_FORMAT)}_for_duckdb.csv", index=False)

        conn = duckdb.connect("data.duckdb")
        conn.execute("DROP TABLE IF EXISTS data")
        conn.register("temp", df)
        conn.execute("CREATE TABLE data AS SELECT * FROM temp")
        conn.close()

        self.collector.collect_add_index_data(self.order_date, start_date, end_date)
        df_index = pd.read_csv(f"{DATA_DIR}/korea_index_{self.order_date.strftime(DATE_FORMAT)}.csv").astype({"date": "str"})
        conn = duckdb.connect("data.duckdb")
        conn.execute("DROP TABLE IF EXISTS market_index")
        conn.register("index", df_index)
        conn.execute("CREATE TABLE market_index AS SELECT * FROM index")
        conn.close()

        try:
            os.remove(f"{DATA_DIR}/korea_price_{self.three_days_before}.csv")
            os.remove(f"{DATA_DIR}/korea_price_{self.three_days_before}_for_duckdb.csv")
            os.remove(f"{DATA_DIR}/korea_index_{self.three_days_before}.csv")
        except FileNotFoundError:
            pass

        return df

    def _set_high_flag(
        self, df: pd.DataFrame, window: int = 20, shift: int = 1
    ) -> pd.DataFrame:
        df["high_flag"] = df["high"] == df["high"].rolling(window=window).max()
        
        # 52주(252 거래일) 동안 high_flag가 없었던 경우만 new_high_flag를 True로 설정
        df["high_flag_52w"] = df["high_flag"].rolling(window=252, min_periods=1).sum()
        df["new_high_flag_52w"] = df["high_flag"] & (df["high_flag_52w"].shift(1, fill_value=0) == 0)
        
        return df

    def _set_low_flag(
        self, df: pd.DataFrame, window: int = 20, shift: int = 1
    ) -> pd.DataFrame:
        df["low_flag"] = df["low"] == df["low"].rolling(window=window).min()

        # 52주(252 거래일) 동안 low_flag가 없었던 경우만 new_low_flag를 True로 설정
        df["low_flag_52w"] = df["low_flag"].rolling(window=252, min_periods=1).sum()
        df["new_low_flag_52w"] = df["low_flag"] & (df["low_flag_52w"].shift(1, fill_value=0) == 0)
        
        return df

    def _set_ma_cross_flag(
        self, df: pd.DataFrame, window1: int = 5, window2: int = 20
    ) -> pd.DataFrame:
        df["first_ma"] = df["close"].rolling(window=window1).mean().round(2)
        df["second_ma"] = df["close"].rolling(window=window2).mean().round(2)
        df["gold_cross_flag"] = (
            (df["first_ma"] > df["second_ma"]) & 
            (df["first_ma"].shift(1, fill_value=0) < df["second_ma"].shift(1, fill_value=0))
        )
        df["dead_cross_flag"] = (
            (df["first_ma"] < df["second_ma"]) & 
            (df["first_ma"].shift(1, fill_value=0) > df["second_ma"].shift(1, fill_value=0))
        )
        return df

    def _set_rsi(self, df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        delta = df['close'].diff()

        # 상승분(up), 하락분(down) 분리
        up   = delta.clip(lower=0)
        down = -delta.clip(upper=0)

        # Wilder’s Smoothing (EMA 방식)
        ema_up   = up.ewm(alpha=1/window, adjust=False).mean()
        ema_down = down.ewm(alpha=1/window, adjust=False).mean()

        rs = ema_up / ema_down
        df['rsi'] = 100 - (100 / (1 + rs)).round(2)
        return df

    def _set_bollinger_bands_flag(self, df: pd.DataFrame, window: int = 20, std_dev: int = 2) -> pd.DataFrame:
        # 이동평균선 계산
        df["bb_middle"] = df["close"].rolling(window=window).mean().round(2)
        
        # 표준편차 계산
        rolling_std = df["close"].rolling(window=window).std()
        
        # 상단 밴드와 하단 밴드 계산
        df["bb_upper"] = (df["bb_middle"] + (rolling_std * std_dev)).round(2)
        df["bb_lower"] = (df["bb_middle"] - (rolling_std * std_dev)).round(2)
        
        # 볼린저 밴드 관련 플래그들
        df["bb_squeeze"] = ((df["bb_upper"] - df["bb_lower"]) < (df["bb_upper"] - df["bb_lower"]).rolling(window=window).mean()).round(2)
        df["bb_upper_flag"] = df["close"] >= df["bb_upper"]
        df["bb_lower_flag"] = df["close"] <= df["bb_lower"]

        return df

    def _set_previous_day_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df["prev_open"] = df["open"].shift(1, fill_value=0)
        df["prev_high"] = df["high"].shift(1, fill_value=0)
        df["prev_low"] = df["low"].shift(1, fill_value=0)
        df["prev_close"] = df["close"].shift(1, fill_value=0)
        df["prev_change"] = df["change"].shift(1, fill_value=0)
        df["prev_change_rate"] = df["change_rate"].shift(1, fill_value=0)
        df["prev_volume"] = df["volume"].shift(1, fill_value=0)
        df["prev_trade_value"] = df["trade_value"].shift(1, fill_value=0)
        return df
    

if __name__ == "__main__":
    processor = StockDataProcessor(date(2025,7,28))
    processor.process_data(date(2025,7,25), date(2025,7,28))