"""
Pull ES=F continuous futures data from yfinance across multiple timeframes.

yfinance interval/period limits:
  5m  -> up to 60 days
  15m -> up to 60 days
  60m -> up to 730 days (~2yr)
  1d  -> max (decades)
  1wk -> max (decades)

Saves each timeframe to its own parquet file. Prints coverage stats so we can
sanity-check what we actually got back vs SPY (which only had 04:00-19:55 ET).
"""
import yfinance as yf
import pandas as pd
from pathlib import Path

OUT_DIR = Path(__file__).parent
SYMBOL = "ES=F"

JOBS = [
    # (interval, period, label)
    ("5m",  "60d",  "ES_5m_60d"),
    ("15m", "60d",  "ES_15m_60d"),
    ("60m", "730d", "ES_60m_730d"),
    ("1d",  "10y",  "ES_1d_10y"),
    ("1wk", "10y",  "ES_1wk_10y"),
]

def pull(interval, period, label):
    print(f"\n=== {label}  interval={interval} period={period} ===")
    df = yf.download(
        SYMBOL,
        interval=interval,
        period=period,
        auto_adjust=False,
        progress=False,
        prepost=False,
    )
    if df.empty:
        print("  EMPTY")
        return None

    # yfinance may return MultiIndex columns when downloading single ticker; flatten
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Normalize timezone to US/Eastern for intraday (yfinance gives UTC for intraday, naive for daily)
    if df.index.tz is not None:
        df.index = df.index.tz_convert("US/Eastern")

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df = df.sort_index()

    out_path = OUT_DIR / f"{label}.parquet"
    df.to_parquet(out_path)

    print(f"  Rows: {len(df):,}")
    print(f"  Range: {df.index.min()}  ->  {df.index.max()}")
    if interval in ("5m", "15m", "60m"):
        per_hour = df.groupby(df.index.hour).size()
        print(f"  Hours present: {sorted(per_hour.index.tolist())}")
        print(f"  Bars per hour (head): {per_hour.head(6).to_dict()}")
        print(f"  Bars per hour (overnight 20-03): "
              f"{ {h: int(per_hour.get(h,0)) for h in [20,21,22,23,0,1,2,3]} }")
    print(f"  Saved: {out_path.name}")
    return df

for interval, period, label in JOBS:
    try:
        pull(interval, period, label)
    except Exception as e:
        print(f"  ERROR pulling {label}: {e}")
