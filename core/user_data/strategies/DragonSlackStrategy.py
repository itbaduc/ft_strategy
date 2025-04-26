# --- Imports và cấu hình chung ---
import numpy as np
import pandas as pd
from pandas import DataFrame
from datetime import datetime
from typing import Optional

import talib.abstract as ta  # TA-Lib
from freqtrade.strategy import IStrategy, Trade, Order
from freqtrade.strategy import DecimalParameter, IntParameter

from technical.pivots_points import pivots_points

class DragonSlackStrategy(IStrategy):
    """
    DragonSlackStrategy
    - Timeframe: 1d
    - Futures isolated margin, tập trung vào 2 trades cùng lúc
    - Mục tiêu: avg 20–30% lợi nhuận mỗi tháng

Result for strategy DragonSlackStrategy
                                                BACKTESTING REPORT
┏━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┓
┃          Pair ┃ Trades ┃ Avg Profit % ┃ Tot Profit USDT ┃ Tot Profit % ┃ Avg Duration ┃  Win  Draw  Loss  Win% ┃
┡━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━┩
│ BTC/USDT:USDT │     16 │         8.79 │         130.650 │        13.07 │      6:00:00 │    9     0     7  56.2 │
│         TOTAL │     16 │         8.79 │         130.650 │        13.07 │      6:00:00 │    9     0     7  56.2 │
└───────────────┴────────┴──────────────┴─────────────────┴──────────────┴──────────────┴────────────────────────┘
                                         LEFT OPEN TRADES REPORT
┏━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┓
┃  Pair ┃ Trades ┃ Avg Profit % ┃ Tot Profit USDT ┃ Tot Profit % ┃ Avg Duration ┃  Win  Draw  Loss  Win% ┃
┡━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━┩
│ TOTAL │      0 │          0.0 │           0.000 │          0.0 │         0:00 │    0     0     0     0 │
└───────┴────────┴──────────────┴─────────────────┴──────────────┴──────────────┴────────────────────────┘
                                                ENTER TAG STATS
┏━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Enter Tag ┃ Entries ┃ Avg Profit % ┃ Tot Profit USDT ┃ Tot Profit % ┃ Avg Duration ┃  Win  Draw  Loss  Win% ┃
┡━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━┩
│     OTHER │      16 │         8.79 │         130.650 │        13.07 │      6:00:00 │    9     0     7  56.2 │
│     TOTAL │      16 │         8.79 │         130.650 │        13.07 │      6:00:00 │    9     0     7  56.2 │
└───────────┴─────────┴──────────────┴─────────────────┴──────────────┴──────────────┴────────────────────────┘
                                               EXIT REASON STATS
┏━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Exit Reason ┃ Exits ┃ Avg Profit % ┃ Tot Profit USDT ┃ Tot Profit % ┃ Avg Duration ┃  Win  Draw  Loss  Win% ┃
┡━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━┩
│ exit_cancel │    16 │         8.79 │         130.650 │        13.07 │      6:00:00 │    9     0     7  56.2 │
│       TOTAL │    16 │         8.79 │         130.650 │        13.07 │      6:00:00 │    9     0     7  56.2 │
└─────────────┴───────┴──────────────┴─────────────────┴──────────────┴──────────────┴────────────────────────┘
                                                           MIXED TAG STATS
┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┓
┃           Enter Tag ┃ Exit Reason ┃ Trades ┃ Avg Profit % ┃ Tot Profit USDT ┃ Tot Profit % ┃ Avg Duration ┃  Win  Draw  Loss  Win% ┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━┩
│ ('', 'exit_cancel') │             │     16 │         8.79 │         130.650 │        13.07 │      6:00:00 │    9     0     7  56.2 │
│               TOTAL │             │     16 │         8.79 │         130.650 │        13.07 │      6:00:00 │    9     0     7  56.2 │
└─────────────────────┴─────────────┴────────┴──────────────┴─────────────────┴──────────────┴──────────────┴────────────────────────┘
                    SUMMARY METRICS
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Metric                      ┃ Value                 ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━┩
│ Backtesting from            │ 2025-03-01 00:00:00   │
│ Backtesting to              │ 2025-04-25 09:00:00   │
│ Trading Mode                │ Isolated Futures      │
│ Max open trades             │ 1                     │
│                             │                       │
│ Total/Daily Avg Trades      │ 16 / 0.29             │
│ Starting balance            │ 1000 USDT             │
│ Final balance               │ 1130.65 USDT          │
│ Absolute profit             │ 130.65 USDT           │
│ Total profit %              │ 13.07%                │
│ CAGR %                      │ 125.89%               │
│ Sortino                     │ 8.74                  │
│ Sharpe                      │ 2.41                  │
│ Calmar                      │ 292.23                │
│ SQN                         │ 1.68                  │
│ Profit factor               │ 5.18                  │
│ Expectancy (Ratio)          │ 8.17 (1.83)           │
│ Avg. daily profit %         │ 0.24%                 │
│ Avg. stake amount           │ 94.698 USDT           │
│ Total trade volume          │ 29169.192 USDT        │
│                             │                       │
│ Best Pair                   │ BTC/USDT:USDT 13.07%  │
│ Worst Pair                  │ BTC/USDT:USDT 13.07%  │
│ Best trade                  │ BTC/USDT:USDT 64.73%  │
│ Worst trade                 │ BTC/USDT:USDT -17.65% │
│ Best day                    │ 59.892 USDT           │
│ Worst day                   │ -11.5 USDT            │
│ Days win/draw/lose          │ 8 / 40 / 6            │
│ Avg. Duration Winners       │ 8:33:00               │
│ Avg. Duration Loser         │ 2:43:00               │
│ Max Consecutive Wins / Loss │ 3 / 2                 │
│ Rejected Entry signals      │ 0                     │
│ Entry/Exit Timeouts         │ 0 / 0                 │
│                             │                       │
│ Min balance                 │ 997.513 USDT          │
│ Max balance                 │ 1130.65 USDT          │
│ Max % of account underwater │ 1.55%                 │
│ Absolute Drawdown (Account) │ 1.55%                 │
│ Absolute Drawdown           │ 17.159 USDT           │
│ Drawdown high               │ 104.932 USDT          │
│ Drawdown low                │ 87.773 USDT           │
│ Drawdown Start              │ 2025-04-21 05:00:00   │
│ Drawdown End                │ 2025-04-21 18:00:00   │
│ Market change               │ 1.78%                 │
└─────────────────────────────┴───────────────────────┘

Backtested 2025-03-01 00:00:00 -> 2025-04-25 09:00:00 | Max open trades : 1
                                                             STRATEGY SUMMARY
┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┓
┃                     ┃        ┃              ┃                 ┃              ┃              ┃     Win  Draw  Loss ┃                    ┃
┃            Strategy ┃ Trades ┃ Avg Profit % ┃ Tot Profit USDT ┃ Tot Profit % ┃ Avg Duration ┃                Win% ┃           Drawdown ┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━┩
│ DragonSlackStrategy │     16 │         8.79 │         130.650 │        13.07 │      6:00:00 │       9     0     7 │ 17.159 USDT  1.55% │
│                     │        │              │                 │              │              │                56.2 │                    │
└─────────────────────┴────────┴──────────────┴─────────────────┴──────────────┴──────────────┴─────────────────────┴────────────────────┘
    """

    INTERFACE_VERSION = 3
    timeframe = "1d"
    minimal_roi = {"0": 3}         # TP tối thiểu 30% (hyperopt tối ưu)
    stoploss = -0.50                  # fallback SL −50% nếu callback lỗi
    trailing_stop = False
    process_only_new_candles = True
    startup_candle_count = 20

    # Hyperopt params
    sl_buffer = DecimalParameter(0.01, 0.10, default=0.05, space='stoploss')      # SL buffer 1–10% :contentReference[oaicite:5]{index=5}
    reward_multiplier = DecimalParameter(2, 5, default=4, space='exit')          # RR từ 2 đến 5

    # chỉ chạy max 2 trades để tập trung vốn
    max_open_trades = 5

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Donchian Channel 55
        dataframe['dc_high'] = dataframe['high'].rolling(window=30).max()
        dataframe['dc_low']  = dataframe['low'].rolling(window=30).min()
        # ATR 14
        dataframe['atr']     = ta.ATR(dataframe, timeperiod=14)
        # ADX 14
        adx = ta.ADX(dataframe, timeperiod=14)  # returns single series :contentReference[oaicite:6]{index=6}
        dataframe['adx']     = adx

        # Pivots (dùng để custom_stoploss lấy trough)
        piv = pivots_points(dataframe, timeperiod=15, levels=1)
        dataframe['trough']  = piv['s1']
        dataframe['peak'] = piv['r1']

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['enter_long'] = 0

        for idx in range(1, len(dataframe)):
            row = dataframe.iloc[idx]
            prev = dataframe.iloc[idx-1]

            # Breakout Donchian trên + sufficient volatility + ADX
            if (
                row['close'] > prev['dc_high'] and
                # (row['atr'] / row['close']) > 0.015 and
                row['adx'] > 20
            ):
                dataframe.at[dataframe.index[idx], 'enter_long'] = 1

        return dataframe

    def custom_stoploss(
        self,
        pair: str,
        trade: Optional[Trade],
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs
    ) -> float:
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        # Xác định entry_rate và entry_time
        if trade and hasattr(trade, "open_date"):
            entry_rate = trade.open_rate
            entry_time = trade.open_date
        else:
            entry_rate = current_rate
            entry_time = current_time or datetime.utcnow()

        # Chuẩn hóa entry_time sang UTC-aware
        entry_time = pd.to_datetime(entry_time)
        if entry_time.tzinfo is None:
            entry_time = entry_time.tz_localize("UTC")

        # Lọc candles trước entry_time
        df_entry = df[df.index <= entry_time] if isinstance(df.index, pd.DatetimeIndex) \
                   else df[df['date'] <= entry_time]

        # Lấy trough gần nhất
        troughs = df_entry['trough'].dropna()
        last_trough = float(troughs.iloc[-1]) if not troughs.empty else float(df_entry['low'].iloc[-1])

        sl_buffer = getattr(self.sl_buffer, 'value', self.sl_buffer)
        sl_price = last_trough * (1 - sl_buffer)
        return (sl_price - entry_rate) / entry_rate

    def leverage(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        entry_tag: Optional[str],
        side: str,
        **kwargs
    ) -> float:
        sl_pct = abs(self.custom_stoploss(
            pair, None, current_time, current_rate, 0.0
        )) * 100
        lev = 1.0 if sl_pct == 0 else max(1.0, (100.0 / sl_pct) - 5.0)
        return min(max_leverage, lev)

    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs
    ) -> Optional[str]:
        # Tính dynamic TP theo Reward Multiplier
        sl_pct = abs(self.custom_stoploss(pair, trade, current_time, current_rate, current_profit))
        reward_mult = getattr(self.reward_multiplier, 'value', self.reward_multiplier)
        tp_price = trade.open_rate * (1 + sl_pct * reward_mult)
        if current_rate >= tp_price:
            return 'exit_tp'

        # Cancel-condition: 2 lower pivots
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
        entry_tag: Optional[str],
        side: str,
        **kwargs
    ) -> Optional[float]:
        if order.exit_reason == 'exit_tp':
            # Tính lại TP price
            sl_pct = abs(self.custom_stoploss(pair, trade, current_time, None, None))
            reward_mult = getattr(self.reward_multiplier, 'value', self.reward_multiplier)
            return trade.open_rate * (1 + sl_pct * reward_mult)
        return None

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['exit_long'] = 0
        dataframe['exit_tag']  = ''
        sl_buf = getattr(self.sl_buffer, 'value', self.sl_buffer)
        rp = getattr(self.reward_multiplier, 'value', self.reward_multiplier)

        for idx in range(1, len(dataframe)):
            price = dataframe['close'].iat[idx]
            entry_rate = dataframe['open'].iat[idx]  # giả định entry tại open
            tp_thresh = entry_rate * (1 + sl_buf * rp)
            if price >= tp_thresh:
                dataframe.at[dataframe.index[idx], 'exit_long'] = 1
                dataframe.at[dataframe.index[idx], 'exit_tag']  = 'exit_tp'

            # Cancel pivots
            peaks   = dataframe['peak'].ffill().dropna().iloc[:idx].tail(2).values
            troughs = dataframe['trough'].ffill().dropna().iloc[:idx].tail(2).values
            if len(peaks)==2 and len(troughs)==2 and peaks[1]<peaks[0] and troughs[1]<troughs[0]:
                dataframe.at[dataframe.index[idx], 'exit_long'] = 1
                dataframe.at[dataframe.index[idx], 'exit_tag']  = 'exit_cancel'

        return dataframe
