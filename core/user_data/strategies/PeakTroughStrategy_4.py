# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these imports ---
import numpy as np
import pandas as pd
import pandas_ta as pta
from datetime import datetime, timedelta
from pandas import DataFrame
from typing import Dict, Optional, Union, Tuple
from freqtrade.strategy import (
    IStrategy,
    Trade,
    Order,
    stoploss_from_absolute,
    DecimalParameter, IntParameter
)
from technical.pivots_points import pivots_points
import talib.abstract as ta


class PeakTroughStrategy_4(IStrategy):
    INTERFACE_VERSION = 3
    startup_candle_count: int = 200  # đảm bảo đủ nến khởi tạo :contentReference[oaicite:6]{index=6}

    # --- Hyperopt-able parameters ---
    buy_pivot_up       = DecimalParameter(0.02, 0.15, default=0.05, space='buy')     # breakout threshold :contentReference[oaicite:7]{index=7}
    rsi_period         = IntParameter(5, 30,     default=14,   space='buy')         # RSI lookback :contentReference[oaicite:8]{index=8}
    rsi_buy            = IntParameter(20, 50,    default=30,   space='buy')         # RSI entry threshold :contentReference[oaicite:9]{index=9}
    volume_ma_period   = IntParameter(10, 50,    default=20,   space='buy')         # MA volume window :contentReference[oaicite:10]{index=10}
    volume_factor      = DecimalParameter(0.5, 2.0, default=1.0,  space='buy')       # Volume filter multiplier :contentReference[oaicite:11]{index=11}

    take_profit_pct    = DecimalParameter(0.02, 0.3, default=0.10, space='sell')     # TP % hyperopt :contentReference[oaicite:12]{index=12}
    stoploss_pct       = DecimalParameter(0.01, 0.2, default=0.05, space='sell')     # SL % hyperopt :contentReference[oaicite:13]{index=13}
    rsi_sell           = IntParameter(50, 90,    default=70,   space='sell')         # RSI exit threshold :contentReference[oaicite:14]{index=14}

    max_hold_candles   = IntParameter(10, 100,   default=48,   space='sell')         # tối đa nến nắm giữ (30m mỗi nến) :contentReference[oaicite:15]{index=15}

    # --- Cấu hình cố định ---
    timeframe = '30m'
    minimal_roi = {"0": 10}
    stoploss = -0.99
    trailing_stop = False

    def informative_pairs(self) -> list:
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Pivot Points (r1 = peak, s1 = trough) :contentReference[oaicite:16]{index=16}
        piv = pivots_points(dataframe, timeperiod=30, levels=1)
        dataframe['peak']   = np.where(piv['r1'].notna(), piv['r1'], np.nan)
        dataframe['trough'] = np.where(piv['s1'].notna(), piv['s1'], np.nan)

        # RSI & Volume MA
        # dataframe[f"rsi{self.rsi_period.value}"] = ta.RSI(dataframe['close'], timeperiod=self.rsi_period.value)
        dataframe['vol_ma'] = ta.SMA(dataframe['volume'], timeperiod=int(self.volume_ma_period.value))

        # Trend filter (EMA50/200 + ADX)
        dataframe['ema50'] = ta.EMA(dataframe['close'], timeperiod=50)
        dataframe['ema200']= ta.EMA(dataframe['close'], timeperiod=200)
        dataframe['adx']   = ta.ADX(dataframe)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['enter_long'] = 0

        for idx in range(1, len(dataframe)):
            # Tính RSI tức thời mỗi lần check
            rsi_series = ta.RSI(dataframe['close'], timeperiod=int(self.rsi_period.value))
            # rsi = int(rsi_series.iat[idx])
            rsi = rsi_series[idx]

            prev_peaks = dataframe['peak'].iloc[:idx].dropna()
            if prev_peaks.empty:
                continue
            last_peak = prev_peaks.iloc[-1]

            price = dataframe['close'].iat[idx]
            vol = dataframe['volume'].iat[idx]
            vol_ma = dataframe['vol_ma'].iat[idx]
            adx = dataframe['adx'].iat[idx]
            ema50 = dataframe['ema50'].iat[idx]
            ema200= dataframe['ema200'].iat[idx]

            cond1 = price > last_peak * (1 + int(self.buy_pivot_up.value))
            cond2 = rsi < int(self.rsi_buy.value)
            cond3 = vol > vol_ma * int(self.volume_factor.value)
            cond4 = (price > ema50) and (ema50 > ema200) and (adx > 25)

            if cond1 and cond2 and cond3 and cond4:
                dataframe.at[dataframe.index[idx], 'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['exit_long'] = 0
        # Giả sử entry_price = last_peak*(1+buy_pivot_up)
        entry_index = None

        for idx in range(1, len(dataframe)):
            price = dataframe['close'].iat[idx]
            # rsi   = dataframe[f"rsi{self.rsi_period.value}"].iat[idx]
            rsi_series = ta.RSI(dataframe['close'], timeperiod=int(self.rsi_period.value))
            # rsi = int(rsi_series.iat[idx])
            rsi = rsi_series[idx]

            # Nếu candle này có entry, nhớ index
            if dataframe['enter_long'].iat[idx] == 1:
                entry_index = idx
                last_peak = dataframe['peak'].iloc[:idx].dropna().iloc[-1]
                entry_price = last_peak * (1 + int(self.buy_pivot_up.value))

            if entry_index is None:
                continue

            # 1) Take-profit
            if price >= entry_price * (1 + int(self.take_profit_pct.value)):
                dataframe.at[dataframe.index[idx], 'exit_long'] = 1
                entry_index = None
                continue

            # 2) Stop-loss
            if price <= entry_price * (1 - int(self.stoploss_pct.value)):
                dataframe.at[dataframe.index[idx], 'exit_long'] = 1
                entry_index = None
                continue

            # 3) RSI exit
            if rsi > int(self.rsi_sell.value):
                dataframe.at[dataframe.index[idx], 'exit_long'] = 1
                entry_index = None
                continue

            # 4) Max hold time
            if idx - entry_index >= int(self.max_hold_candles.value):
                dataframe.at[dataframe.index[idx], 'exit_long'] = 1
                entry_index = None

        return dataframe
