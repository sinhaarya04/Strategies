"""
Backtrader Strategy: Heikin Ashi EMA Band + RSI

Buy:  HA close > 20 EMA(High) AND RSI > 50
Sell: HA close < 20 EMA(Low)  AND RSI < 50
"""

import backtrader as bt
from config import (
    EMA_PERIOD, RSI_PERIOD, ATR_PERIOD,
    RSI_BUY_THRESHOLD, RSI_SELL_THRESHOLD,
    ES_POINT_VALUE, ATR_RISK_MULTIPLIER, ATR_TARGET_MULTIPLIER, RISK_PER_TRADE_PCT,
)


class HeikinAshiClose(bt.Indicator):
    """Compute Heikin Ashi Close: (O + H + L + C) / 4"""
    lines = ("ha_close",)

    def __init__(self):
        self.lines.ha_close = (
            self.data.open + self.data.high + self.data.low + self.data.close
        ) / 4.0


class HeikinAshiEMAStrategy(bt.Strategy):
    params = dict(
        ema_period=EMA_PERIOD,
        rsi_period=RSI_PERIOD,
        atr_period=ATR_PERIOD,
        rsi_buy=RSI_BUY_THRESHOLD,
        rsi_sell=RSI_SELL_THRESHOLD,
        atr_risk_mult=ATR_RISK_MULTIPLIER,
        atr_target_mult=ATR_TARGET_MULTIPLIER,
        risk_pct=RISK_PER_TRADE_PCT,
        point_value=ES_POINT_VALUE,
    )

    def __init__(self):
        # Heikin Ashi close (simplified — full HA needs recursive open,
        # but HA close is the main signal component)
        self.ha_close = HeikinAshiClose(self.data)

        # EMA bands on regular High and Low
        self.ema_high = bt.indicators.EMA(self.data.high, period=self.p.ema_period)
        self.ema_low = bt.indicators.EMA(self.data.low, period=self.p.ema_period)

        # RSI on regular close
        self.rsi = bt.indicators.RSI(self.data.close, period=self.p.rsi_period)

        # ATR on regular OHLC
        self.atr = bt.indicators.ATR(self.data, period=self.p.atr_period)

        # Track orders
        self.order = None
        self.entry_price = None
        self.stop_price = None
        self.target_price = None

    def log(self, msg):
        dt = self.data.datetime.datetime(0)
        print(f"  [{dt:%Y-%m-%d %H:%M}] {msg}")

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f"BUY  @ {order.executed.price:.2f} x{order.executed.size:.0f}")
                self.entry_price = order.executed.price
            else:
                self.log(f"SELL @ {order.executed.price:.2f} x{abs(order.executed.size):.0f}")
                self.entry_price = order.executed.price
        self.order = None

    def notify_trade(self, trade):
        if trade.isclosed:
            self.log(f"P&L: ${trade.pnl:.2f} (net: ${trade.pnlcomm:.2f})")

    def _calc_size(self):
        """ATR-based position sizing: risk X% of portfolio per trade."""
        atr_val = self.atr[0]
        if atr_val <= 0:
            return 1
        cash = self.broker.getvalue()
        risk_dollars = cash * self.p.risk_pct
        stop_distance = atr_val * self.p.atr_risk_mult
        dollar_risk_per_contract = stop_distance * self.p.point_value
        if dollar_risk_per_contract <= 0:
            return 1
        size = int(risk_dollars / dollar_risk_per_contract)
        return max(1, size)

    def next(self):
        if self.order:
            return

        ha_c = self.ha_close.ha_close[0]
        ema_h = self.ema_high[0]
        ema_l = self.ema_low[0]
        rsi = self.rsi[0]
        atr_val = self.atr[0]

        # Check stop loss and take profit
        if self.position:
            if self.position.size > 0:
                if self.stop_price and self.data.close[0] <= self.stop_price:
                    self.log(f"STOP HIT @ {self.data.close[0]:.2f}")
                    self.order = self.close()
                    self.stop_price = None
                    self.target_price = None
                    return
                if self.target_price and self.data.close[0] >= self.target_price:
                    self.log(f"TARGET HIT @ {self.data.close[0]:.2f}")
                    self.order = self.close()
                    self.stop_price = None
                    self.target_price = None
                    return
            elif self.position.size < 0:
                if self.stop_price and self.data.close[0] >= self.stop_price:
                    self.log(f"STOP HIT @ {self.data.close[0]:.2f}")
                    self.order = self.close()
                    self.stop_price = None
                    self.target_price = None
                    return
                if self.target_price and self.data.close[0] <= self.target_price:
                    self.log(f"TARGET HIT @ {self.data.close[0]:.2f}")
                    self.order = self.close()
                    self.stop_price = None
                    self.target_price = None
                    return

        if not self.position:
            # No position — look for entry
            if ha_c > ema_h and rsi > self.p.rsi_buy:
                size = self._calc_size()
                self.order = self.buy(size=size)
                self.stop_price = self.data.close[0] - atr_val * self.p.atr_risk_mult
                self.target_price = self.data.close[0] + atr_val * self.p.atr_target_mult

            elif ha_c < ema_l and rsi < self.p.rsi_sell:
                size = self._calc_size()
                self.order = self.sell(size=size)
                self.stop_price = self.data.close[0] + atr_val * self.p.atr_risk_mult
                self.target_price = self.data.close[0] - atr_val * self.p.atr_target_mult

        else:
            # In position — look for exit (reverse signal)
            if self.position.size > 0:
                # Long — exit if HA close drops below lower EMA and RSI < 50
                if ha_c < ema_l and rsi < self.p.rsi_sell:
                    self.order = self.close()
                    self.stop_price = None
                    self.target_price = None

            elif self.position.size < 0:
                # Short — exit if HA close rises above upper EMA and RSI > 50
                if ha_c > ema_h and rsi > self.p.rsi_buy:
                    self.order = self.close()
                    self.stop_price = None
                    self.target_price = None
