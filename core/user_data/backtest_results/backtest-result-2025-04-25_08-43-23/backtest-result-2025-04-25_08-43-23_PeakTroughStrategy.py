# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these imports ---
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pandas import DataFrame
from typing import Dict, Optional, Union, Tuple

from freqtrade.strategy import (
    IStrategy,
    Trade,
    Order,
    PairLocks,
    informative,  # @informative decorator
    # Hyperopt Parameters
    BooleanParameter,
    CategoricalParameter,
    DecimalParameter,
    IntParameter,
    RealParameter,
    # timeframe helpers
    timeframe_to_minutes,
    timeframe_to_next_date,
    timeframe_to_prev_date,
    # Strategy helper functions
    merge_informative_pair,
    stoploss_from_absolute,
    stoploss_from_open,
)

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import pandas_ta as pta
from technical import qtpylib
from technical.pivots_points import pivots_points


class PeakTroughStrategy(IStrategy):
    """
    This is a strategy template to get you started.
    More information in https://www.freqtrade.io/en/latest/strategy-customization/

    You can:
        :return: a Dataframe with all mandatory indicators for the strategies
    - Rename the class name (Do not forget to update class_name)
    - Add any methods you want to build your strategy
    - Add any lib you need to build your strategy

    You must keep:
    - the lib in the section "Do not remove these libs"
    - the methods: populate_indicators, populate_entry_trend, populate_exit_trend
    You should keep:
    - timeframe, minimal_roi, stoploss, trailing_*
    """

    # Cấu hình chung
    minimal_roi = {"0": 10}         # Không dùng ROI mặc định
    stoploss = -0.10            # Stoploss tạm, override via callback 

    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 3

    # Optimal timeframe for the strategy.
    timeframe = "30m"

    # Can this strategy go short?
    can_short: bool = False

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    # minimal_roi = {
    #     "60": 0.01,
    #     "30": 0.02,
    #     "0": 0.04
    # }

    # Trailing stoploss
    trailing_stop = False
    # trailing_only_offset_is_reached = False
    # trailing_stop_positive = 0.01
    # trailing_stop_positive_offset = 0.0  # Disabled / not configured

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # These values can be overridden in the config.
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 30

    # Strategy parameters
    buy_rsi = IntParameter(10, 40, default=30, space="buy")
    sell_rsi = IntParameter(60, 90, default=70, space="sell")

    def informative_pairs(self):
        """
        Define additional, informative pair/interval combinations to be cached from the exchange.
        These pair/interval combinations are non-tradeable, unless they are part
        of the whitelist as well.
        For more information, please consult the documentation
        :return: List of tuples in the format (pair, interval)
            Sample: return [("ETH/USDT", "5m"),
                            ("BTC/USDT", "15m"),
                            ]
        """
        return []
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Xác định đỉnh (peak) và đáy (trough) dựa trên Pivot Points:
        - pivot: điểm trung tâm
        - r1: Resistance #1 (lấy làm peak)
        - s1: Support  #1 (lấy làm trough)
        """
        # 1. Tính pivot points với timeperiod=30 và chỉ lấy 1 cấp level
        piv = pivots_points(dataframe, timeperiod=30, levels=1)
        # 2. Gán r1 làm peak, s1 làm trough
        dataframe['peak']   = np.where(piv['r1'].notna(), piv['r1'], np.nan)
        dataframe['trough'] = np.where(piv['s1'].notna(), piv['s1'], np.nan)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Cột enter_long = 1 khi thỏa điều kiện
        dataframe['enter_long'] = 0
        for idx in range(2, len(dataframe)):
            row = dataframe.iloc[idx]
            last_peak = dataframe['peak'].ffill().iloc[idx-1]
            if row['close'] > last_peak:
                dataframe.at[dataframe.index[idx], 'enter_long'] = 1
        return dataframe
    
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: str | None, side: str,
                 **kwargs) -> float:
        """
        Customize leverage for each new trade. This method is only called in futures mode.

        :param pair: Pair that's currently analyzed
        :param current_time: datetime object, containing the current datetime
        :param current_rate: Rate, calculated based on pricing settings in exit_pricing.
        :param proposed_leverage: A leverage proposed by the bot.
        :param max_leverage: Max leverage allowed on this pair
        :param entry_tag: Optional entry_tag (buy_tag) if provided with the buy signal.
        :param side: "long" or "short" - indicating the direction of the proposed trade
        :return: A leverage amount, which is between 1.0 and max_leverage.
        """
        return 2.5
    
    # custom_entry_price
    def customize_entry(self, pair: str, trade: Trade, order: Order, **kwargs) -> None:
        df = self.dp.get(pair, self.timeframe).df
        last_peak = df['peak'].ffill().iloc[-2]
        last_trough = df['trough'].ffill().iloc[-2]
        buy_price = last_peak * 1.05
        stop_price = last_trough * 0.95
        take_profit_price = buy_price * 2

        # Override orders
        self.modify_order(trade, price=buy_price, side='buy', order_type='limit')
        self.stop_loss(pair, price=stop_price)
        self.take_profit(pair, price=take_profit_price)

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
        """
        Override giá limit entry:
        Giá BUY = last_peak * 1.05
        """
        # Lấy last_peak từ DataFrame
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        # Tính pivots nếu chưa có
        piv = pivots_points(dataframe, timeperiod=30, levels=1)
        last_peak = piv['r1'].ffill().iloc[-2]  # r1 = peak
        # Nếu không có điều kiện entry trong candle hiện tại, giữ proposed_rate
        if entry_tag != "enter_long":
            return proposed_rate
        # Tính giá mới
        new_price = last_peak * 1.05
        return new_price

    # custom_exit
    def check_cancel(self, pair: str, trade: Trade, **kwargs) -> bool:
        df = self.dp.get(pair, self.timeframe).df
        peaks = df['peak'].dropna().tail(2).values
        troughs = df['trough'].dropna().tail(2).values
        if len(peaks)==2 and len(troughs)==2:
            if peaks[1] < peaks[0] and troughs[1] < troughs[0]:
                return True
        return False
    
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
        Sinh exit-signal khi:
         1) Take-profit: close >= buy_price * 2
         2) Cancel-condition: 2 đỉnh & 2 đáy mới thấp hơn
        Trả về 'exit_tp' hoặc 'exit_cancel' để bot tự động close.
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        # 1) Take Profit
        buy_price = trade.open_rate  # entry_price đã dùng ở custom_entry_price
        if current_rate >= buy_price * 2:
            return "exit_tp"

        # 2) Cancel-Condition
        piv = pivots_points(dataframe, timeperiod=30, levels=1)
        peaks   = piv['r1'].ffill().dropna().tail(2).values
        troughs = piv['s1'].ffill().dropna().tail(2).values
        if len(peaks)==2 and len(troughs)==2 and peaks[1]<peaks[0] and troughs[1]<troughs[0]:
            return "exit_cancel"

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
        - Với exit_tp: đặt limit tại buy_price * 2
        - Với exit_cancel: trả về None để hủy order exit (và bot sẽ thực hiện market exit)
        """
        # Lấy exit_reason
        exit_reason = order.exit_reason  # Freqtrade gắn exit_reason từ custom_exit
        buy_price = trade.open_rate

        if exit_reason == "exit_tp":
            return buy_price * 2
        if exit_reason == "exit_cancel":
            return None

        # Mặc định giữ proposed_rate
        return current_order_rate

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Sinh tín hiệu exit_long dựa trên:
         1) Take-profit: close >= last_peak * 1.05 * 2
         2) Cancel-condition: hai đỉnh & hai đáy mới thấp hơn
        """
        # Khởi tạo cột
        dataframe['exit_long'] = 0
        dataframe['exit_tag']  = ''

        # Duyệt từng dòng
        for idx in range(1, len(dataframe)):
            row = dataframe.iloc[idx]
            # 1) Take-Profit
            # last_peak đã được tính trong populate_indicators(), fill-forward
            last_peak = dataframe['peak'].ffill().iloc[idx-1]
            tp_threshold = last_peak * 1.05 * 2
            if row['close'] >= tp_threshold:
                dataframe.at[dataframe.index[idx], 'exit_long'] = 1
                dataframe.at[dataframe.index[idx], 'exit_tag']  = 'exit_tp'

            # 2) Cancel-Condition
            peaks   = dataframe['peak'].ffill().dropna().iloc[:idx].tail(2).values
            troughs = dataframe['trough'].ffill().dropna().iloc[:idx].tail(2).values
            if len(peaks)==2 and len(troughs)==2:
                if (peaks[1] < peaks[0]) and (troughs[1] < troughs[0]):
                    dataframe.at[dataframe.index[idx], 'exit_long'] = 1
                    dataframe.at[dataframe.index[idx], 'exit_tag']  = 'exit_cancel'

        return dataframe