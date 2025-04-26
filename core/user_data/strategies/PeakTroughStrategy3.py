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
    DecimalParameter
)
from technical.pivots_points import pivots_points
import talib.abstract as ta

class PeakTroughStrategy3(IStrategy):
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

    sl_buffer = DecimalParameter(0.01, 0.10, default=0.03, space='stoploss')  # 1%–10%
    reward_multiplier = DecimalParameter(2, 3, default=2, space='exit')       # 2x–3x

    def informative_pairs(self) -> list:
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if 'date' not in dataframe.columns:
            dataframe['date'] = dataframe.index  # chỉ cần khi index là datetime

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
        """
        Tính leverage = (100 / SL_percent) - 5, capped bởi exchange max_leverage
        VD: SL = 5% => leverage = 20 - 5 = 15x
        """        
        # Gọi lại custom_stoploss để tính sl_pct (giả định entry tại current_rate)
        sl_pct = abs(self.custom_stoploss(
            pair=pair,
            trade=None,  # Không có trade thật, giả định entry = current_rate
            current_time=current_time,
            current_rate=current_rate,
            current_profit=0.0
        )) * 100  # chuyển sang %

        # Tránh chia 0
        if sl_pct == 0:
            return 1.0
        
        # Công thức leverage
        leverage_value = max(1.0, (100 / sl_pct) - 5.0)
        return min(max_leverage, leverage_value)

    def custom_stoploss(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs
    ) -> float:
        """
        SL = last trough * (1 - sl_buffer)
        Trả về giá trị stoploss (tỉ lệ so với entry): (SL - entry) / entry
        """
        # Lấy DataFrame đã phân tích
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        if trade is not None and hasattr(trade, "open_rate") and hasattr(trade, "open_date"):
            entry_rate = trade.open_rate
            entry_time = trade.open_date.replace(tzinfo=None)
        else:
            entry_rate = current_rate
            entry_time = current_time or datetime.utcnow()

        # Chuyển về timezone-aware UTC
        entry_time = pd.to_datetime(entry_time)
        if entry_time.tzinfo is None:
            entry_time = entry_time.tz_localize("UTC")
        else:
            entry_time = entry_time.tz_convert("UTC")

        # Xử lý index an toàn
        if isinstance(df.index, pd.DatetimeIndex):
            df_entry = df[df.index <= entry_time]
        elif 'date' in df.columns:
            df_entry = df[df['date'] <= pd.to_datetime(entry_time)]
        else:
            df_entry = df.copy()
        
        # Lấy đáy (trough) cuối cùng trước entry
        trough_series = df_entry['trough'].dropna()
        if not trough_series.empty:
            last_trough = float(trough_series.iloc[-1])
        else:
            # fallback: dùng low candle entry
            last_trough = float(df_entry['low'].iloc[-1])
        
        sl_price = last_trough * (1 - self.sl_buffer.value)  
        # Trả về tỉ lệ SL so với giá entry (trade.open_rate)
        # return (sl_price - trade.open_rate) / trade.open_rate
        return (sl_price - entry_rate) / entry_rate

    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs
    ) -> str | None:
        """
        Exit khi:
         - Take-profit dynamic: Reward = SL * reward_multiplier
         - Hoặc cancel-condition 2 lower peaks & troughs
        """
        # Chuyển về timezone-aware UTC
        current_time = pd.to_datetime(current_time)
        if current_time.tzinfo is None:
            current_time = current_time.tz_localize("UTC")
        else:
            current_time = current_time.tz_convert("UTC")

        # Tính lại SL_pct
        sl_pct = abs(self.custom_stoploss(pair, trade, current_time, current_rate, current_profit))  
        # Giá TP dynamic
        tp_price = trade.open_rate * (1 + sl_pct * self.reward_multiplier.value)
        if current_rate >= tp_price:
            return 'exit_tp'
        
        # Cancel-condition (giữ nguyên)
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        piv = pivots_points(df, timeperiod=15, levels=1)
        peaks   = piv['r1'].ffill().dropna().tail(2).values
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
        """
        - Nếu exit_tp: trả về price = entry * (1 + SL_pct * reward_multiplier)
        - Nếu exit_cancel: hủy order exit để bot tự market-exit
        """
        # Chuyển về timezone-aware UTC
        current_time = pd.to_datetime(current_time)
        if current_time.tzinfo is None:
            current_time = current_time.tz_localize("UTC")
        else:
            current_time = current_time.tz_convert("UTC")

        if order.exit_reason == 'exit_tp':
            # Tính SL_pct và TP_price tương tự custom_exit
            sl_pct = abs(self.custom_stoploss(pair, trade, current_time, None, None))
            return trade.open_rate * (1 + sl_pct * self.reward_multiplier.value)
        if order.exit_reason == 'exit_cancel':
            return None
        # Giữ nguyên
        return current_order_rate

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Sinh cột exit_long, exit_tag với TP dynamic và cancel-condition
        """
        dataframe['exit_long'] = 0
        dataframe['exit_tag']  = ''
        
        # Tính SL_pct tĩnh từ parameter mẫu (giả sử SL% = sl_buffer.value)
        sl_pct = self.sl_buffer.value  
        tp_mult = self.reward_multiplier.value
        
        for idx in range(1, len(dataframe)):
            price = dataframe['close'].iat[idx]
            entry_rate = dataframe['open'].iat[idx]  # khoảng entry giả định
            tp_thresh = entry_rate * (1 + sl_pct * tp_mult)
            # 1) TP
            if price >= tp_thresh:
                dataframe.at[dataframe.index[idx], 'exit_long'] = 1
                dataframe.at[dataframe.index[idx], 'exit_tag']  = 'exit_tp'
            # 2) Cancel-condition
            peaks   = dataframe['peak'].ffill().dropna().iloc[:idx].tail(2).values
            troughs = dataframe['trough'].ffill().dropna().iloc[:idx].tail(2).values
            if len(peaks)==2 and len(troughs)==2 and peaks[1]<peaks[0] and troughs[1]<troughs[0]:
                dataframe.at[dataframe.index[idx], 'exit_long'] = 1
                dataframe.at[dataframe.index[idx], 'exit_tag']  = 'exit_cancel'
        
        return dataframe
