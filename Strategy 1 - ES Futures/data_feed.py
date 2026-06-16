"""Fetch ES/SPY intraday bars from Polygon.io (1yr+) or yfinance (60d fallback)."""

import os
import time
import pandas as pd
from datetime import datetime, timedelta

POLYGON_KEY = os.environ.get("POLYGON_API_KEY", "")  # set POLYGON_API_KEY env var if using Polygon path
CACHE_DIR = os.path.dirname(os.path.abspath(__file__))


def fetch_polygon(symbol="SPY", interval="5", days=365):
    """
    Pull intraday bars from Polygon.io.
    Free tier: 5 req/min, 2yr history, 15-min delay.
    Uses SPY as ES proxy (identical during RTH).
    """
    import requests

    end = datetime.now()
    start = end - timedelta(days=days)

    cache_file = os.path.join(CACHE_DIR, f"polygon_{symbol}_{interval}m_{days}d.parquet")
    if os.path.exists(cache_file):
        age_hours = (time.time() - os.path.getmtime(cache_file)) / 3600
        if age_hours < 12:
            df = pd.read_parquet(cache_file)
            print(f"[DATA] Loaded {len(df)} cached bars from {cache_file}")
            print(f"[DATA] Range: {df.index[0]} -> {df.index[-1]}")
            return df

    all_bars = []
    # Polygon free tier: 5 req/min. Fetch in 30-day chunks.
    chunk_start = start
    page = 0
    while chunk_start < end:
        chunk_end = min(chunk_start + timedelta(days=30), end)
        s_str = chunk_start.strftime("%Y-%m-%d")
        e_str = chunk_end.strftime("%Y-%m-%d")

        page += 1
        print(f"[DATA] Fetching {symbol} {interval}m: {s_str} to {e_str} (page {page})...")

        r = requests.get(
            f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/{interval}/minute/{s_str}/{e_str}",
            params={"apiKey": POLYGON_KEY, "limit": 50000, "sort": "asc"},
            timeout=30,
        )

        if r.status_code == 429:
            print("[DATA] Rate limited, waiting 60s...")
            time.sleep(60)
            continue

        r.raise_for_status()
        data = r.json()
        results = data.get("results", [])
        all_bars.extend(results)
        print(f"[DATA]   Got {len(results)} bars")

        chunk_start = chunk_end + timedelta(days=1)
        time.sleep(13)  # stay under 5 req/min

    if not all_bars:
        raise ValueError("No data returned from Polygon")

    df = pd.DataFrame(all_bars)
    df["datetime"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_convert("US/Eastern")
    df = df.set_index("datetime")
    df = df.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"})
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]

    # Cache
    df.to_parquet(cache_file)

    print(f"[DATA] Fetched {len(df)} bars of {symbol} @ {interval}m")
    print(f"[DATA] Range: {df.index[0]} -> {df.index[-1]}")
    return df


def fetch_es_data(symbol="SPY", interval=None, days=None):
    """
    Loader priority:
      1. Bloomberg ES 5m parquet (real futures, full 24h Globex) — when ES_5m_bbg.parquet exists
         and USE_BBG_ES env var is "1" (default).
      2. Polygon.io SPY proxy (legacy, 04:00-19:55 ET only).
      3. yfinance fallback (60 days max).
    """
    from config import INTERVAL, LOOKBACK_DAYS
    if interval is None:
        interval = INTERVAL.replace("m", "")  # "5m" -> "5"
    if days is None:
        days = LOOKBACK_DAYS

    use_bbg = os.environ.get("USE_BBG_ES", "1") == "1"
    bbg_path = os.path.join(CACHE_DIR, "ES_5m_bbg.parquet")
    if use_bbg and os.path.exists(bbg_path):
        df = pd.read_parquet(bbg_path)
        print(f"[DATA] Loaded {len(df):,} ES 5m bars from Bloomberg parquet (real futures, 24h)")
        print(f"[DATA] Range: {df.index[0]} -> {df.index[-1]}")
        return df

    try:
        return fetch_polygon(symbol=symbol, interval=interval, days=days)
    except Exception as e:
        print(f"[DATA] Polygon failed ({e}), falling back to yfinance...")
        import yfinance as yf
        ticker = yf.Ticker("ES=F" if symbol == "SPY" else symbol)
        yf_interval = f"{interval}m"
        yf_days = min(days, 60)
        df = ticker.history(period=f"{yf_days}d", interval=yf_interval)
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.dropna(inplace=True)
        print(f"[DATA] Fetched {len(df)} bars of ES=F @ {yf_interval} (yfinance fallback)")
        print(f"[DATA] Range: {df.index[0]} -> {df.index[-1]}")
        return df


if __name__ == "__main__":
    df = fetch_es_data()
    print(df.tail(10))
