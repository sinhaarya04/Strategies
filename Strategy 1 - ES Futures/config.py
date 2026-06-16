# ES E-mini Futures — Heikin Ashi EMA Band + RSI Strategy Config

# Timeframe
INTERVAL = "5m"
LOOKBACK_DAYS = 365  # 1 year via Polygon.io (60d fallback with yfinance)

# Indicators
EMA_PERIOD = 20       # EMA of High / EMA of Low
RSI_PERIOD = 14       # RSI on regular close
ATR_PERIOD = 14       # ATR for sizing/stops

# Entry rules
RSI_BUY_THRESHOLD = 55    # RSI must be above this to go long
RSI_SELL_THRESHOLD = 45   # RSI must be below this to go short

# Risk
INITIAL_CAPITAL = 50000.0
COMMISSION_PER_CONTRACT = 1.24  # CME ES round trip
MARGIN_PER_CONTRACT = 500.0     # day trading margin
ES_POINT_VALUE = 50.0           # $50 per point for ES
ATR_RISK_MULTIPLIER = 1.5       # stop distance = ATR * multiplier
ATR_TARGET_MULTIPLIER = 1.0     # take profit = ATR * multiplier
RISK_PER_TRADE_PCT = 0.02       # risk 2% of capital per trade
