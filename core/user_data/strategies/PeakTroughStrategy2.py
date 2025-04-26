# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these imports ---
import numpy as np
import pandas as pd
import pandas_ta as pta
from datetime import datetime
from pandas import DataFrame
from typing import Dict, Optional, Union, Tuple

from freqtrade.strategy import (
    IStrategy,
    Trade,
    Order,
    stoploss_from_absolute,
)
from technical.pivots_points import pivots_points
import talib.abstract as ta

class PeakTroughStrategy2(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = "30m"
    can_short = False
    process_only_new_candles = True
    startup_candle_count = 40

    minimal_roi = {"0": 0.2}
    stoploss = -0.10

    take_profit = 2     # lợi nhuận 200%

    trading_mode = 'futures'
    margin_mode = 'isolated'

    # Enable trailing stop: start after +2% and trail at 1%
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02
    trailing_only_offset_is_reached = True

    def informative_pairs(self) -> list:
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Pivot Points
        piv = pivots_points(dataframe, timeperiod=15, levels=1)
        dataframe['peak'] = piv['r1']
        dataframe['trough'] = piv['s1']

        # Donchian Channel (20 periods)
        don = pta.donchian(high=dataframe['high'], low=dataframe['low'], length=20)
        dataframe['dc_high'] = dataframe['high'].rolling(window=20).max()
        dataframe['dc_low']  = dataframe['low'].rolling(window=20).min()
        dataframe['dc_middle']  = (dataframe['dc_high'] + dataframe['dc_low']) / 2

        # ATR for stoploss
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['enter_long'] = 0
        for idx in range(1, len(dataframe)):
            last_peak = dataframe['peak'].ffill().iloc[idx-1]
            cond_peak = dataframe['close'].iloc[idx] > last_peak
            cond_dc   = dataframe['close'].iloc[idx] > dataframe['dc_high'].iloc[idx]
            cond_trough = not np.isnan(dataframe['trough'].ffill().iloc[idx-1])
            if (cond_peak or cond_dc) and cond_trough:
                dataframe.at[dataframe.index[idx], 'enter_long'] = 1
        return dataframe

    def custom_entry_price(
        self,
        pair: str,
        trade: Trade | None,
        current_time: datetime,
        proposed_rate: float,
        entry_tag: str | None,
        side: str,
        **kwargs
    ) -> float:
        if entry_tag != 'enter_long':
            return proposed_rate
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_peak = df['peak'].ffill().iloc[-2]
        return max(proposed_rate, last_peak * 1.05)

    def leverage(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs
    ) -> float:
        # Fixed high leverage to boost returns, capped by exchange
        return min(max_leverage, 20)

    def custom_stoploss(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs
    ) -> float:
        # Tight stoploss: 1*ATR below entry
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        atr = df['atr'].iloc[-1]
        stop_price = trade.open_rate - atr
        return (stop_price - trade.open_rate) / trade.open_rate

    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs
    ) -> str | None:
        # Take profit at +20%
        if current_rate >= trade.open_rate * self.take_profit:
            return 'exit_tp'
        # Cancel condition: two lower peaks & troughs
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        piv = pivots_points(df, timeperiod=15, levels=1)
        peaks = piv['r1'].ffill().dropna().tail(2).values
        troughs = piv['s1'].ffill().dropna().tail(2).values
        if len(peaks)==2 and len(troughs)==2 and peaks[1]<peaks[0] and troughs[1]<troughs[0]:
            return 'exit_cancel'
        return None

    def adjust_exit_price(
        self,
        trade: Trade,
        order: Order,
        pair: str,
        current_time: datetime,
        proposed_rate: float,
        current_order_rate: float,
        entry_tag: str | None,
        side: str,
        **kwargs
    ) -> float | None:
        if order.exit_reason == 'exit_tp':
            return trade.open_rate * self.take_profit
        return None

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['exit_long'] = 0
        dataframe['exit_tag'] = ''
        for idx in range(1, len(dataframe)):
            if dataframe['close'].iloc[idx] >= dataframe['open'].iloc[idx] * self.take_profit:
                dataframe.at[dataframe.index[idx], 'exit_long'] = 1
                dataframe.at[dataframe.index[idx], 'exit_tag'] = 'exit_tp'
            piv = dataframe['peak'].ffill().dropna().iloc[:idx].tail(2).values
            troughs = dataframe['trough'].ffill().dropna().iloc[:idx].tail(2).values
            if len(piv)==2 and len(troughs)==2 and piv[1]<piv[0] and troughs[1]<troughs[0]:
                dataframe.at[dataframe.index[idx], 'exit_long'] = 1
                dataframe.at[dataframe.index[idx], 'exit_tag'] = 'exit_cancel'
        return dataframe
