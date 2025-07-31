import os
import glob
from typing import Optional, Dict, List
from datetime import date, timedelta, datetime
from crawling import MarketDataCollector
import pandas as pd, ast
import yfinance as yf
import logging
import duckdb
import pandas_market_calendars as mcal
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
print(DATA_DIR)

DATE_FORMAT = "%Y-%m-%d"
# date.today().strftime(DATE_FORMAT)


class Tools:
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.logger.debug("Tools 클래스 초기화 시작")

        # data 디렉토리에서 가장 최신 korea_price 파일 찾기
        price_pattern = os.path.join(DATA_DIR, "korea_price_????-??-??.csv")
        price_files = glob.glob(price_pattern)
        
        self.logger.debug(f"발견된 price 파일들: {price_files}")
        
        if not price_files:
            self.logger.error("❌ korea_price 데이터 파일을 찾을 수 없습니다.")
            self.logger.error(f"다음 디렉토리를 확인해주세요: {DATA_DIR}")
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
        
        self.logger.info(f"가장 최신 파일 선택: {latest_file} (날짜: {latest_date})")
        
        try:
            self.data = pd.read_csv(latest_file, dtype={"date": "str", "ticker": "str"})
            self.logger.info(f"데이터 로딩 완료: {len(self.data)}개 레코드")
        except Exception as e:
            self.logger.error(f"❌ 파일 로딩 실패: {str(e)}")
            raise

        self.data = self.data.groupby("ticker", group_keys=False).apply(self._set_previous_day_data).reset_index(drop=True)
        self.conn = duckdb.connect('data.duckdb')

        # name_to_ticker_map 생성
        self.name_to_ticker_map = (
            self.data[["ticker", "ticker_name"]].copy()
            .drop_duplicates()
            .set_index("ticker_name")["ticker"]
            .to_dict()
        )
       # ticker 컬럼의 중복 없는 값들로, 키와 값을 동일하게 매핑
        tickers = self.data["ticker"].copy().unique()
        self.ticker_map = { t: t for t in tickers }

        self.logger.debug(f"티커 맵 생성 완료: {len(self.name_to_ticker_map)}, {len(self.ticker_map)}개 종목")

        # Function Calling을 위한 도구 정의
        self.function_definitions = [
            {
                "type": "function",
                "function": {
                    "name": "get_real_time_stock_price_by_yfinance",
                    "description": "yfinance를 사용하여 실시간 주가 정보를 조회합니다.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "symbol": {
                                "type": "string",
                                "description": "종목명 또는 티커 코드",
                            }
                        },
                        "required": ["symbol"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_stock_filter_by_dataframe",
                    "description": "실시간 혹은 과거 주가 데이터를 DataFrame으로 수집하여 여러 조건으로 필터링",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "date": {
                                "type": "string",
                                "description": "조회할 날짜 (YYYY-MM-DD 형식)",
                            },
                            "operations": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "method_str": {
                                            "type": "string",
                                            "description": "DataFrame 메서드명 (nlargest, nsmallest, query, sort_values, sample, head, tail 등)",
                                        },
                                        "args_str": {
                                            "type": "string",
                                            "description": "인수들을 문자열로 입력. (코스피 = \\\"KQ\\\", 코스닥 = \\\"KQ\\\" 로 요청)",
                                        },
                                        "kwargs_str": {
                                            "type": "string",
                                            "description": '키워드 인수들을 문자열로 입력. 예: "ascending=False" (정렬용). 빈 문자열("")이면 키워드 인수 없음',
                                        },
                                    },
                                },
                                "required": ["method_str", "args_str", "kwargs_str"],
                            },
                        },
                        "required": ["date", "operations"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_historical_ticker_data",
                    "description": "특정 종목의 과거 데이터(18시 이전 하루 전, 18시 이후 당일)를 조회합니다. 원하는 컬럼만 선택적으로 조회할 수 있습니다.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "symbol": {
                                "type": "string",
                                "description": "종목명",
                            },
                            "date": {
                                "type": "string",
                                "description": "조회할 날짜 (YYYY-MM-DD 형식)",
                            },
                            "select_columns": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "조회할 컬럼 리스트 (예: ['date', 'ticker', 'ticker_name', 'ticker_market', 'open', 'high', 'low', 'close', 'volume', 'trade_value', 'cap', 'shares_out', 'eps', 'per', 'fwd_eps', 'fwd_per', 'pbr'])",
                            },
                        },
                        "required": ["symbol", "date", "select_columns"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_conditional_search_data",
                    "description": "조건에 맞는 데이터를 조회합니다.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "dates": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "description": "조회할 날짜 (YYYY-MM-DD 형식)",
                                },
                            },
                            "query": {
                                "type": "string",
                                "description": "조회할 SQL 문. SQL 문을 자유롭게 추론하여, 요청에 맞는 데이터를 조회합니다.",
                            },
                        },
                        "required": ["dates", "query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_backtest_data",
                    "description": "백테스트 데이터를 조회합니다.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "종목의 buy signal을 추출하는 쿼리, 'date'와 'ticker' 컬럼과 'buy_signal' 컬럼이 있어야 함",
                            },
                            "profit_rate": {
                                "type": "number",
                                "description": "익절률",
                            },
                            "stoploss_rate": {
                                "type": "number",
                                "description": "손절률",
                            },
                        },
                        "required": ["query", "profit_rate", "stoploss_rate"],
                    },
                },
            },
        ]
        self.logger.debug(f"함수 정의 완료: {len(self.function_definitions)}개 함수")

        self.available_functions = {
            "get_real_time_stock_price_by_yfinance": self.get_real_time_stock_price_by_yfinance,
            "get_stock_filter_by_dataframe": self.get_stock_filter_by_dataframe,
            "get_historical_ticker_data": self.get_historical_ticker_data,
            "get_conditional_search_data": self.get_conditional_search_data,
            "get_backtest_data": self.get_backtest_data,
        }

    def get_ticker_by_stock_name(self, symbol: str) -> Dict:
        self.logger.info(f"종목명으로 티커 검색: {symbol}")
        symbol = symbol.replace(" ", "")

        ticker_match = self.ticker_map.get(symbol)
        if ticker_match:
            return {
                "success": False,
                "ticker": ticker_match,
                "message": f"티커로 요청할 수 없습니다. 종목명으로 요청해주세요.",
            }

        # 1. 정확히 일치하는 종목명 먼저 검색
        exact_match = self.name_to_ticker_map.get(symbol)


        if exact_match:
            return {
                "success": True,
                "ticker": exact_match,
                "message": f"{symbol}의 티커: {exact_match}",
            }

        self.logger.debug(
            f"정확히 일치하는 종목명 검색 실패, 부분 일치 검색 시작: {symbol}"
        )
        # 2. 부분 일치 검색
        partial_matches = {
            name: ticker
            for name, ticker in self.name_to_ticker_map.items()
            if symbol in name
        }

        if len(partial_matches) == 1:
            # 부분 일치가 1개면 그것을 반환
            return {
                "success": True,
                "ticker": list(partial_matches.values())[0],
                "message": f"{symbol}의 티커: {list(partial_matches.values())[0]}",
            }
        elif len(partial_matches) > 1:
            # 여러 개 일치하면 선택지 제공
            return {
                "success": False,
                "message": f"여러 종목 발견: {list(partial_matches.keys())}",
            }
        else:
            return {"success": False, "message": "종목을 찾을 수 없습니다"}

    def get_real_time_stock_price_by_yfinance(self, symbol: str) -> Dict:
        self.logger.info(f"실시간 주가 조회: {symbol}")
        try:
            ticker_info = self.get_ticker_by_stock_name(symbol)
            if ticker_info["success"]:
                ticker = ticker_info["ticker"]
            else:
                return ticker_info

            stock = yf.Ticker(f"{ticker}.KS")
            info = stock.info
            previous_close = info.get("regularMarketPreviousClose")
            day_open = info.get("open")
            day_low = info.get("dayLow")
            day_high = info.get("dayHigh")
            volume = info.get("volume")
            market_cap = info.get("marketCap")
            current_price = info.get("regularMarketPrice") or info.get(
                "currentPrice", "N/A"
            )
            current_ratio = info.get("regularMarketChangePercent")
            long_name = info.get("longName")

            result = {
                "success": True,
                "ticker": ticker,
                "info": {
                    "long_name": long_name,
                    "previous_close": previous_close,
                    "day_open": day_open,
                    "day_low": day_low,
                    "day_high": day_high,
                    "volume": volume,
                    "market_cap": market_cap,
                    "current_price": current_price,
                    "current_ratio": current_ratio,
                },
            }
            self.logger.info(f"조회 성공: {current_price}")
            return result
        except Exception as e:
            self.logger.error(f"   ❌ 조회 실패: {str(e)}")
            return {
                "success": False,
                "ticker": symbol,
                "error": str(e),
                "message": f"{symbol} 주가 조회 중 오류가 발생했습니다: {str(e)}",
            }

    def get_stock_filter_by_dataframe(self, date: str, operations: List[Dict]) -> Dict:
        def _dynamic_call(df, method_str, args_str="", kwargs_str=""):
            kwargs = eval(f"dict({kwargs_str})") if kwargs_str else {}
            method = getattr(df, method_str)

            if method_str == "head" or method_str == "tail" or method_str == "sample":
                if df.empty or int(args_str) > len(df):
                    return df
                return method(int(args_str), **kwargs)
            elif method_str == "count":
                return len(df)
            elif args_str.startswith("'"):
                args = ast.literal_eval(f"({args_str},)")
                return method(*args, **kwargs)
            else:
                args = eval(f"({args_str},)")
                return method(*args, **kwargs)


        if not self._is_trading_days(date):
            return {
                "success": False,
                "message": f"{date}는 거래일이 아닙니다. 공휴일이거나 휴장일입니다.",
            }

        try:
            if self.data["date"].max() < date:
                self.logger.info(f"실시간 주가 조회: {date} {len(operations)}개 작업")
                collector = MarketDataCollector()
                df = collector.collect_real_time_data()
                df = pd.concat([self.data, df])
                df = df.groupby("ticker").apply(self._set_previous_day_data)
                df = df[df["date"] == date]
            else:
                df = self.data[self.data["date"] == date]
                self.logger.debug(f"과거 주가 데이터 수집 완료: {df.shape}")

        except Exception as e:
            self.logger.error(f"   ❌ 조회 실패: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "message": f"실시간 주가 데이터 수집 중 오류가 발생했습니다: {str(e)}",
            }

        try:
            filtered_df = df
            for i, op in enumerate(operations):
                method_str = op["method_str"]
                args_str = op["args_str"]
                kwargs_str = op["kwargs_str"]

                filtered_df = _dynamic_call(
                    filtered_df, method_str, args_str, kwargs_str
                )
                self.logger.debug(f"조회 성공: {filtered_df}")

            self.logger.debug(f"조회 성공: {filtered_df}")

            if type(filtered_df) is not pd.DataFrame:
                return {
                    "success": True,
                    "message": f"({method_str}) 건 조회 완료",
                    "data": filtered_df,
                }
            if filtered_df.empty:
                return {
                    "success": True,
                    "message": "요청하신 데이터는 없습니다.",
                }
            if len(filtered_df) >= 30:
                return {
                    "success": True,
                    "message": f"응답 데이터가 {len(filtered_df)}건입니다. 종목명만 응답합니다. ",
                    "data": filtered_df[["ticker_name"]]
                    .head(1000)["ticker_name"]
                    .tolist(),
                }
            return {
                "success": True,
                "message": f"({method_str}) 데이터 조회 완료, 총 {len(filtered_df)}건",
                "data": filtered_df[
                    [
                        "ticker",
                        "ticker_name",
                        "open",
                        "high",
                        "low",
                        "close",
                        "change_rate",
                        "volume",
                        "trade_value",
                        "cap",
                    ]
                ].to_dict(orient='records'),
            }
        except Exception as e:
            self.logger.error(f"   ❌ 조회 실패: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "message": f"조회 중 오류가 발생했습니다: {str(e)}",
            }

    def get_historical_ticker_data(
        self, symbol: str, date: str = None, select_columns: List[str] = None
    ) -> Dict:
        """
        데이터 파일에서 특정 종목의 과거 데이터를 조회합니다.

        Args:
            symbol: 종목명 또는 티커 코드
            date: 조회할 날짜 (YYYY-MM-DD 형식, 생략시 최신 날짜)
            select_columns: 조회할 컬럼 리스트 (생략시 전체 컬럼)

        Returns:
            Dict: 조회 결과
        """
        self.logger.info(
            f"과거 데이터 조회: {symbol}, 날짜: {date}, 컬럼: {select_columns}"
        )

        try:
            # 특정 날짜 데이터 조회
            if not self._is_trading_days(date):
                return {
                    "success": False,
                    "message": f"{date}는 거래일이 아닙니다. 공휴일이거나 휴장일입니다.",
                }
            if symbol == "":
                return {
                    "success": False,
                    "message": "종목명이 없습니다. 종목명을 입력해주세요.",
                }
            ticker_info = self.get_ticker_by_stock_name(symbol)
            if ticker_info["success"]:
                ticker = ticker_info["ticker"]
            else:
                return ticker_info

            stock_data = self.data[
                (self.data["ticker"] == ticker) & (self.data["date"] == date)
            ]

            # 3. 데이터 존재 확인
            if stock_data.empty:
                return {
                    "success": False,
                    "ticker": ticker,
                    "message": f"{date} {ticker} 데이터가 없습니다.",
                }

            # 4. 컬럼 선택
            available_columns = list(stock_data.columns)

            if select_columns:
                # 유효한 컬럼만 필터링
                valid_columns = [
                    col for col in select_columns if col in available_columns
                ]
                invalid_columns = [
                    col for col in select_columns if col not in available_columns
                ]

                if invalid_columns:
                    self.logger.warning(f"⚠️ 잘못된 컬럼명: {invalid_columns}")

                if not valid_columns:
                    return {
                        "success": False,
                        "ticker": ticker,
                        "message": f"유효한 컬럼이 없습니다. 사용 가능한 컬럼: {available_columns}",
                    }

                selected_data = stock_data[valid_columns]
                ticker_name = stock_data["ticker_name"].values[0]
            else:
                selected_data = stock_data
                ticker_name = stock_data["ticker_name"].values[0]
                valid_columns = available_columns

            # 5. 결과 반환
            result_data = selected_data.iloc[0].to_dict()

            result = {
                "success": True,
                "ticker": ticker,
                "date": date,
                "ticker_name": ticker_name,
                "selected_columns": valid_columns,
                "data": result_data,
                "message": f"({ticker}) 데이터 조회 완료",
            }

            self.logger.info(f"조회 성공: {len(valid_columns)}개 컬럼")
            return result

        except Exception as e:
            self.logger.error(f"   ❌ 조회 실패: {str(e)}")
            return {
                "success": False,
                "ticker": ticker,
                "error": str(e),
                "message": f"{ticker} 히스토리컬 데이터 조회 중 오류가 발생했습니다.",
            }

    def get_conditional_search_data(
        self, dates: List[str], query: str
    ) -> Dict:
        self.logger.info(
            f"조건에 맞는 데이터 조회: {dates}, {query}"
        )
        if len(dates) == 1:
            if not self._is_trading_days(dates[0]):
                return {
                    "success": True,
                    "message": f"{dates[0]}는 거래일이 아닙니다. 공휴일이거나 휴장일입니다.",
                }
        
        try:
            df = self.conn.execute(query).df()

            if len(df) == 0:
                return {
                    "success": True,
                    "message": "요청하신 데이터는 없습니다.",
                }
            if len(df) > 30:
                return {
                    "success": True,
                    "message": f"응답 데이터가 {len(df)}건입니다. 종목명만 응답합니다. ",
                    "data": df[["ticker_name"]]
                    .head(1000)["ticker_name"]
                    .tolist(),
                }
            return {
                "success": True,
                "data": df.to_dict(orient="records"),
                "message": f"조회 성공: {len(df)}건",
            }
        except Exception as e:
            self.logger.error(f"   ❌ 조회 실패: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "message": f"조회 중 오류가 발생했습니다: {str(e)}",
            }
        
    def get_backtest_data(self, query: str, profit_rate: float, stoploss_rate: float) -> Dict:
        self.logger.info(f"백테스트 데이터 조회: {query}, {profit_rate}, {stoploss_rate}")
        try:
            if profit_rate >= 1:
                profit_rate = round(profit_rate / 100, 2)
            if stoploss_rate >= 1:
                stoploss_rate = round(stoploss_rate / 100, 2)
            
            df = self.data.copy()
            signal_df = self.conn.execute(query).df()
            signal_df = signal_df[['date', 'ticker', 'buy_signal']]

            df = df.merge(signal_df, on=["date", "ticker"], how="left")
            trades = self._backtest(df, profit_rate, stoploss_rate)
            result = self._calculate_performance(trades)

            return {
                "success": True,
                "result": result,
                "message": f"백테스트 조회 성공: {result['message']}",
            }
        except Exception as e:
            self.logger.error(f"   ❌ 조회 실패: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "message": f"백테스트 조회 중 오류가 발생했습니다: {str(e)}",
            }

    def _backtest(self, df: pd.DataFrame, profit_rate: float, stoploss_rate: float) -> pd.DataFrame:
        # 포지션 초기화
        positions = pd.DataFrame(columns=['ticker', 'shares', 'buy_price', 'take_profit_price', 'stop_loss_price'])
        positions = positions.astype({
            'ticker': 'str',
            'buy_price': 'int64',
            'take_profit_price': 'int64',
            'stop_loss_price': 'int64'
        })
        trades = []

        trading_dates = df['date'].unique()
        for date in trading_dates:
            if date < '2023-01-01':
                continue
            group = df[df['date'] == date]
            if not positions.empty:
                # 현재 price 계산
                current_prices_high = group.set_index('ticker')['high']
                current_prices_low = group.set_index('ticker')['low']
                current_prices_close = group.set_index('ticker')['close']

                positions['current_price_high'] = positions['ticker'].map(current_prices_high)
                positions['current_price_low'] = positions['ticker'].map(current_prices_low)
                positions['current_price'] = positions['ticker'].map(current_prices_close)

                # 상장폐지 처리
                nan_positions = positions[positions['current_price'].isna()]
                if not nan_positions.empty:
                    for _, position in nan_positions.iterrows():
                        profit = -position['buy_price']
                        trades.append({
                            'action': 'SELL',
                            'date': date,
                            'ticker': position['ticker'],
                            'price': 0,
                            'buy_price': position['buy_price'],
                            'take_profit_price': position['take_profit_price'],
                            'stop_loss_price': position['stop_loss_price'],
                            'pnl': profit,
                            'sale_type': 'DELISTING'
                        })
                    positions = positions[~positions['current_price'].isna()]
                # 익절/손절 조건 확인 익절부터 처리
                positions['take_profit_triggered'] = positions['current_price_high'] >= positions['take_profit_price']
                positions['stop_loss_triggered'] = (~positions['take_profit_triggered']) & (positions['current_price_low'] <= positions['stop_loss_price'])
                
                sell_positions = positions[positions['take_profit_triggered'] | positions['stop_loss_triggered']]

                # 익절/손절 조건 확인
                if not sell_positions.empty:
                    for _, position in sell_positions.iterrows():
                        if position['stop_loss_triggered']:
                            sell_price = position['stop_loss_price']
                            sale_type = "STOP_LOSS"
                        else:
                            sell_price = position['take_profit_price']
                            sale_type = "TAKE_PROFIT"

                        profit = sell_price - position['buy_price']

                        trades.append({
                            'action': 'SELL',
                            'date': date,
                            'ticker': position['ticker'],
                            'price': sell_price,
                            'buy_price': position['buy_price'],
                            'take_profit_price': position['take_profit_price'],
                            'stop_loss_price': position['stop_loss_price'],
                            'pnl': profit,
                            'sale_type': sale_type,
                        })

                    positions = positions[~(positions['stop_loss_triggered'] | positions['take_profit_triggered'])]
            # 새로운 매수 신호 확인 및 실행
            buy_signals = group[group['buy_signal'] == 1]
            if not buy_signals.empty:
                for _, signal in buy_signals.iterrows():
                    if not positions.empty and signal['ticker'] in positions['ticker'].values:
                        continue
                    
                    # 오늘 검출된 주식을 오늘 종가에 산다고 계산
                    price = signal['close']
                    if price <= 0 or pd.isna(price):
                        continue


                    new_position = pd.DataFrame({
                        'ticker': [signal['ticker']],
                        'buy_price': [price],
                        'take_profit_price': [int(price * (1 + profit_rate))],
                        'stop_loss_price': [int(price * (1 - stoploss_rate))]
                    })

                    trades.append({
                        'action': 'BUY',
                        'date': date,
                        'ticker': signal['ticker'],
                        'price': price,
                        'buy_price': price,
                        'take_profit_price': int(price * (1 + profit_rate)),
                        'stop_loss_price': int(price * (1 - stoploss_rate)),
                        'pnl': 0,
                        'sale_type': 'BUY',
                    })

                    positions = pd.concat([positions, new_position], ignore_index=True)

        return trades
    
    def _calculate_performance(self, trades: List[Dict]) -> Dict:
        """
        백테스팅 결과에서 CAGR과 승률을 계산합니다.
        
        Args:
            trades: 거래 내역 리스트
            
        Returns:
            Dict: CAGR, 승률, 총 거래수, 수익 거래수, 총 손익 등의 성과 지표
        """
        if not trades:
            return {
                "cagr": 0.0,
                "win_rate": 0.0,
                "total_trades": 0,
                "winning_trades": 0,
                "total_pnl": 0,
                "total_return_rate": 0.0,
                "message": "거래 내역이 없습니다."
            }
        
        # 매도 거래만 필터링 (실제 손익이 발생한 거래)
        sell_trades = [trade for trade in trades if trade['action'] == 'SELL']
        
        if not sell_trades:
            return {
                "cagr": 0.0,
                "win_rate": 0.0,
                "total_trades": 0,
                "winning_trades": 0,
                "total_pnl": 0,
                "total_return_rate": 0.0,
                "message": "매도 거래가 없습니다."
            }
        
        # 1. 승률 계산
        total_trades = len(sell_trades)
        winning_trades = len([trade for trade in sell_trades if trade['pnl'] > 0])
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0.0
        
        # 2. 총 손익 계산
        total_pnl = sum(trade['pnl'] for trade in sell_trades)
        
        # 3. 투자 기간 계산 (첫 거래일부터 마지막 거래일까지)
        all_dates = [datetime.strptime(trade['date'], DATE_FORMAT) for trade in trades]
        start_date = min(all_dates)
        end_date = max(all_dates)
        
        # 투자 기간 (년 단위)
        investment_period_days = (end_date - start_date).days
        investment_period_years = investment_period_days / 365.25 if investment_period_days > 0 else 1
        
        # 4. 총 투자금액 계산 (각 거래의 매수가격 합계)
        total_investment = sum(trade['buy_price'] for trade in sell_trades)
        
        # 5. 총 수익률 계산
        total_return_rate = (total_pnl / total_investment * 100) if total_investment > 0 else 0.0
        
        # 6. CAGR 계산
        # CAGR = (최종값/초기값)^(1/년수) - 1
        # 최종값 = 초기값 + 총손익, 초기값 = 총투자금액
        if total_investment > 0 and investment_period_years > 0:
            final_value = total_investment + total_pnl
            if final_value > 0:
                cagr = ((final_value / total_investment) ** (1 / investment_period_years) - 1) * 100
            else:
                cagr = -100.0  # 전액 손실
        else:
            cagr = 0.0
        
        # 7. 평균 거래당 손익
        avg_pnl_per_trade = total_pnl / total_trades if total_trades > 0 else 0
        
        return {
            "cagr": round(cagr, 2),
            "win_rate": round(win_rate, 2),
            "total_pnl": int(total_pnl),
            "total_investment": int(total_investment),
            "return_rate": round(total_return_rate, 2),
            "avg_pnl_per_trade": round(avg_pnl_per_trade, 2),
            "investment_period_years": round(investment_period_years, 2),
            "start_date": start_date.strftime(DATE_FORMAT),
            "end_date": end_date.strftime(DATE_FORMAT),
            "message": f"백테스팅 성과 분석 완료: 총 {total_trades}회 거래, 승률 {win_rate:.1f}%, CAGR {cagr:.2f}%"
        }

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