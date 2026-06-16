"""Indicators: Heikin Ashi, EMA bands, RSI, ATR."""

import numpy as np
import pandas as pd
from config import EMA_PERIOD, RSI_PERIOD, ATR_PERIOD


def to_heikin_ashi(df):
    """
    Convert regular OHLCV DataFrame to Heikin Ashi candles.
    Input df must have columns: Open, High, Low, Close
    Returns DataFrame with: HA_Open, HA_High, HA_Low, HA_Close
    """
    ha = pd.DataFrame(index=df.index)

    ha["HA_Close"] = (df["Open"] + df["High"] + df["Low"] + df["Close"]) / 4.0

    ha["HA_Open"] = 0.0
    ha.iloc[0, ha.columns.get_loc("HA_Open")] = (df["Open"].iloc[0] + df["Close"].iloc[0]) / 2.0
    for i in range(1, len(ha)):
        ha.iloc[i, ha.columns.get_loc("HA_Open")] = (
            ha.iloc[i - 1, ha.columns.get_loc("HA_Open")] +
            ha.iloc[i - 1, ha.columns.get_loc("HA_Close")]
        ) / 2.0

    ha["HA_High"] = pd.concat([df["High"], ha["HA_Open"], ha["HA_Close"]], axis=1).max(axis=1)
    ha["HA_Low"] = pd.concat([df["Low"], ha["HA_Open"], ha["HA_Close"]], axis=1).min(axis=1)

    return ha


def compute_ema(series, period=EMA_PERIOD):
    """Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()


def compute_rsi(close, period=RSI_PERIOD):
    """RSI on regular close prices (not Heikin Ashi)."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def compute_atr(high, low, close, period=ATR_PERIOD):
    """Average True Range on regular OHLC."""
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def compute_fisher(series, period=9):
    """
    Fisher Transform (John Ehlers).
    Normalizes price to -1..+1 over `period` bars, applies Fisher transform.
    Returns (fisher, signal) where signal = fisher shifted by 1 bar.
    """
    n = len(series)
    vals = series.values.astype(np.float64)

    # Rolling min/max normalization
    roll_min = series.rolling(window=period, min_periods=period).min().values
    roll_max = series.rolling(window=period, min_periods=period).max().values
    denom = roll_max - roll_min
    denom[denom == 0] = np.nan
    x_raw = 2.0 * (vals - roll_min) / denom - 1.0

    # Clamp and smooth (EMA alpha=0.33) — recursive, must loop on arrays
    x = np.zeros(n)
    for i in range(n):
        if np.isnan(x_raw[i]):
            x[i] = 0.67 * (x[i - 1] if i > 0 else 0.0)
        else:
            x[i] = 0.33 * np.clip(x_raw[i], -0.999, 0.999) + 0.67 * (x[i - 1] if i > 0 else 0.0)

    # Fisher transform — recursive, loop on arrays
    fisher_arr = np.zeros(n)
    for i in range(n):
        v = x[i]
        if abs(v) < 0.999:
            f = 0.5 * np.log((1.0 + v) / (1.0 - v))
        else:
            f = 0.0
        fisher_arr[i] = f + 0.5 * (fisher_arr[i - 1] if i > 0 else 0.0)

    fisher = pd.Series(fisher_arr, index=series.index)
    signal = fisher.shift(1)
    return fisher, signal


def compute_all(df):
    """
    Compute all indicators on a regular OHLCV DataFrame.
    Returns the original df augmented with indicator columns.
    """
    result = df.copy()

    # Heikin Ashi
    ha = to_heikin_ashi(df)
    result["HA_Open"] = ha["HA_Open"]
    result["HA_High"] = ha["HA_High"]
    result["HA_Low"] = ha["HA_Low"]
    result["HA_Close"] = ha["HA_Close"]

    # EMA bands on regular High/Low
    result["EMA_High"] = compute_ema(df["High"], EMA_PERIOD)
    result["EMA_Low"] = compute_ema(df["Low"], EMA_PERIOD)

    # RSI on regular Close
    result["RSI"] = compute_rsi(df["Close"], RSI_PERIOD)

    # ATR on regular OHLC
    result["ATR"] = compute_atr(df["High"], df["Low"], df["Close"], ATR_PERIOD)

    # Fisher Transform on regular Close
    result["Fisher"], result["Fisher_Signal"] = compute_fisher(df["Close"])

    # Fisher Transform on HA Close
    result["Fisher_HA"], result["Fisher_HA_Signal"] = compute_fisher(ha["HA_Close"])

    return result


if __name__ == "__main__":
    from data_feed import fetch_es_data
    df = fetch_es_data()
    result = compute_all(df)
    print(result[["Close", "HA_Close", "EMA_High", "EMA_Low", "RSI", "ATR"]].tail(20))
