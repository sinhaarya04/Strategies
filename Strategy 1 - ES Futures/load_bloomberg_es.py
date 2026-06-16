"""
Load Bloomberg ES 5m / 15m / 60m bars from ~/Desktop/data (1).xlsx.

Notes on the source file:
- No header row; column 0 = datetime (ET), 1-4 = OHLC, 5 = Volume
- Sheet1 = 5m OHLCV (37,854 rows)
- Sheet2 = 15m OHLCV (12,618 rows)
- Sheet3 = 60m CLOSE-ONLY (3,157 rows). We'll resample 5m -> 60m for full OHLCV instead.
- Timezone: ET (confirmed by user)
- One stray Excel serial in Sheet1 needs converting
"""
import pandas as pd
from pathlib import Path

SRC = Path("/Users/aryansinha/Desktop/data (1).xlsx")
OUT_DIR = Path(__file__).parent

def load_ohlcv(sheet, label):
    df = pd.read_excel(SRC, sheet_name=sheet, header=None,
                       names=["datetime", "Open", "High", "Low", "Close", "Volume"])
    # Convert any Excel serial floats to datetime
    bad = ~df["datetime"].apply(lambda x: isinstance(x, pd.Timestamp) or hasattr(x, "year"))
    if bad.any():
        df.loc[bad, "datetime"] = pd.to_datetime(
            df.loc[bad, "datetime"].astype(float), unit="D", origin="1899-12-30"
        )
    df["datetime"] = pd.to_datetime(df["datetime"])
    # Localize to ET
    df["datetime"] = df["datetime"].dt.tz_localize("US/Eastern", ambiguous="NaT", nonexistent="shift_forward")
    df = df.set_index("datetime").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    return df

print("Loading Sheet1 (5m)...")
es_5m = load_ohlcv("Sheet1", "5m")
print(f"  rows={len(es_5m):,}  range={es_5m.index.min()} -> {es_5m.index.max()}")

print("\nLoading Sheet2 (15m)...")
es_15m = load_ohlcv("Sheet2", "15m")
print(f"  rows={len(es_15m):,}  range={es_15m.index.min()} -> {es_15m.index.max()}")

print("\nResampling 5m -> 60m for full OHLCV (Sheet3 is close-only)...")
es_60m = es_5m.resample("60min").agg({
    "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"
}).dropna()
print(f"  rows={len(es_60m):,}  range={es_60m.index.min()} -> {es_60m.index.max()}")

# Save
for label, df in [("ES_5m_bbg", es_5m), ("ES_15m_bbg", es_15m), ("ES_60m_bbg", es_60m)]:
    out = OUT_DIR / f"{label}.parquet"
    df.to_parquet(out)
    print(f"  saved {out.name}  ({len(df):,} rows)")

# Coverage sanity check
print("\n=== Coverage sanity (5m bars per hour ET) ===")
per_hour = es_5m.groupby(es_5m.index.hour).size()
for h in range(24):
    cnt = per_hour.get(h, 0)
    bar = "#" * int(cnt / max(per_hour) * 40)
    print(f"  {h:02d}:00  {cnt:>6,}  {bar}")

# Check daily session break
print(f"\nHour 17 (daily break) bar count: {per_hour.get(17, 0)}  (expect ~0)")
print(f"Hour 18 (Globex reopen) bar count: {per_hour.get(18, 0)}")
