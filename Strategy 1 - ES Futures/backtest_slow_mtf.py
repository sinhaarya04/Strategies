"""
Slow-timeframe MTF backtests: HA+RSI 3-lot scaled exit on ES.

Two pairs (Gary's email):
  1. Daily entries, Weekly bias    ("D + W")
  2. 60-min entries, Daily bias    ("60m + D")

Same signal as the intraday haRSI scripts:
  LONG  : HA_close > EMA_High AND RSI > 55
  SHORT : HA_close < EMA_Low  AND RSI < 45
Position: 3 contracts. Stop 1.5xATR, TP1/2/3 = 1/2/3xATR.
Ladder: TP1 -> BE on lots 2&3. TP2 -> trail lot 3 to +1xATR. TP3 -> exit.

Data sources:
  60m  : ES_60m_730d.parquet     (yfinance, ~2 yr)
  Daily: ES_1d_10y.parquet       (yfinance, ~10 yr)
  Wk   : ES_1wk_10y.parquet      (yfinance, ~10 yr)

Commissions applied side-by-side: $0 / $4.90 / $3.00 per contract round-trip.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from indicators import compute_all
from config import RSI_BUY_THRESHOLD, RSI_SELL_THRESHOLD

HERE = Path(__file__).parent
ACCOUNT_SIZE = 50_000.0
POINT_VALUE = 50.0  # ES = $50/pt
LOTS_PER_ENTRY = 3
COMMS = [("Gross",        0.00),
         ("Net @ $4.90",  4.90),
         ("Net @ $3.00",  3.00)]


# ---------- Bias signal (slower TF) ---------------------------------------
def compute_bias(df_slow):
    bias = pd.Series(0, index=df_slow.index, dtype=int)
    ha = df_slow["HA_Close"].values
    eh = df_slow["EMA_High"].values
    el = df_slow["EMA_Low"].values
    rs = df_slow["RSI"].values
    for i in range(len(df_slow)):
        if np.isnan(ha[i]) or np.isnan(eh[i]) or np.isnan(rs[i]):
            continue
        if ha[i] > eh[i] and rs[i] > RSI_BUY_THRESHOLD:
            bias.iloc[i] = 1
        elif ha[i] < el[i] and rs[i] < RSI_SELL_THRESHOLD:
            bias.iloc[i] = -1
    return bias


# ---------- 3-lot scaled-exit simulation ----------------------------------
def simulate(df_entry, bias_arr):
    direction = 0
    entry_price = 0.0
    atr_at_entry = 0.0
    lots_remaining = 0
    stop_price = 0.0
    tp1 = tp2 = tp3 = 0.0
    lot1_hit = lot2_hit = False

    pts_total = 0.0
    equities = [0.0]
    n_entries = 0
    wins = losses = 0
    lot_pnls_pts = []
    worst_entry_pts = 0.0
    best_entry_pts = 0.0
    entry_pts_running = 0.0

    ha_c = df_entry["HA_Close"].values
    ema_h = df_entry["EMA_High"].values
    ema_l = df_entry["EMA_Low"].values
    rsi  = df_entry["RSI"].values
    atr  = df_entry["ATR"].values
    open_ = df_entry["Open"].values
    high  = df_entry["High"].values
    low   = df_entry["Low"].values
    close = df_entry["Close"].values

    for i in range(1, len(df_entry)):
        bias = bias_arr[i]

        # --- Manage open position --------------------------------------
        if direction != 0:
            h, l = high[i], low[i]

            if direction == 1:
                # Stop first
                if l <= stop_price:
                    lot_pnl = (stop_price - entry_price) * lots_remaining
                    pts_total += lot_pnl
                    entry_pts_running += lot_pnl
                    lot_pnls_pts.append(stop_price - entry_price)
                    if stop_price > entry_price: wins += 1
                    else: losses += 1
                    direction = 0
                    lots_remaining = 0
                else:
                    # TP1
                    if not lot1_hit and h >= tp1:
                        lot_pnl = (tp1 - entry_price) * 1
                        pts_total += lot_pnl
                        entry_pts_running += lot_pnl
                        lot_pnls_pts.append(tp1 - entry_price)
                        wins += 1
                        lots_remaining -= 1
                        lot1_hit = True
                        stop_price = entry_price  # BE on remaining
                    # TP2
                    if lot1_hit and not lot2_hit and h >= tp2:
                        lot_pnl = (tp2 - entry_price) * 1
                        pts_total += lot_pnl
                        entry_pts_running += lot_pnl
                        lot_pnls_pts.append(tp2 - entry_price)
                        wins += 1
                        lots_remaining -= 1
                        lot2_hit = True
                        stop_price = entry_price + atr_at_entry  # trail to +1 ATR
                    # TP3
                    if lot2_hit and h >= tp3:
                        lot_pnl = (tp3 - entry_price) * 1
                        pts_total += lot_pnl
                        entry_pts_running += lot_pnl
                        lot_pnls_pts.append(tp3 - entry_price)
                        wins += 1
                        lots_remaining -= 1
                        direction = 0
            else:  # direction == -1
                if h >= stop_price:
                    lot_pnl = (entry_price - stop_price) * lots_remaining
                    pts_total += lot_pnl
                    entry_pts_running += lot_pnl
                    lot_pnls_pts.append(entry_price - stop_price)
                    if stop_price < entry_price: wins += 1
                    else: losses += 1
                    direction = 0
                    lots_remaining = 0
                else:
                    if not lot1_hit and l <= tp1:
                        lot_pnl = (entry_price - tp1) * 1
                        pts_total += lot_pnl
                        entry_pts_running += lot_pnl
                        lot_pnls_pts.append(entry_price - tp1)
                        wins += 1
                        lots_remaining -= 1
                        lot1_hit = True
                        stop_price = entry_price
                    if lot1_hit and not lot2_hit and l <= tp2:
                        lot_pnl = (entry_price - tp2) * 1
                        pts_total += lot_pnl
                        entry_pts_running += lot_pnl
                        lot_pnls_pts.append(entry_price - tp2)
                        wins += 1
                        lots_remaining -= 1
                        lot2_hit = True
                        stop_price = entry_price - atr_at_entry
                    if lot2_hit and l <= tp3:
                        lot_pnl = (entry_price - tp3) * 1
                        pts_total += lot_pnl
                        entry_pts_running += lot_pnl
                        lot_pnls_pts.append(entry_price - tp3)
                        wins += 1
                        lots_remaining -= 1
                        direction = 0

            if direction == 0:
                worst_entry_pts = min(worst_entry_pts, entry_pts_running)
                best_entry_pts = max(best_entry_pts, entry_pts_running)
                entry_pts_running = 0.0

        equities.append(pts_total)

        # --- New entry --------------------------------------------------
        if direction == 0 and bias != 0 and i + 1 < len(df_entry):
            if np.isnan(atr[i]) or np.isnan(ha_c[i]) or np.isnan(ema_h[i]) or np.isnan(rsi[i]):
                continue
            long_sig  = ha_c[i] > ema_h[i] and rsi[i] > RSI_BUY_THRESHOLD
            short_sig = ha_c[i] < ema_l[i] and rsi[i] < RSI_SELL_THRESHOLD
            if long_sig and bias == 1:
                direction = 1
                entry_price = open_[i + 1] if not np.isnan(open_[i + 1]) else close[i]
                atr_at_entry = atr[i]
                stop_price = entry_price - 1.5 * atr_at_entry
                tp1 = entry_price + 1.0 * atr_at_entry
                tp2 = entry_price + 2.0 * atr_at_entry
                tp3 = entry_price + 3.0 * atr_at_entry
                lots_remaining = 3
                lot1_hit = lot2_hit = False
                n_entries += 1
            elif short_sig and bias == -1:
                direction = -1
                entry_price = open_[i + 1] if not np.isnan(open_[i + 1]) else close[i]
                atr_at_entry = atr[i]
                stop_price = entry_price + 1.5 * atr_at_entry
                tp1 = entry_price - 1.0 * atr_at_entry
                tp2 = entry_price - 2.0 * atr_at_entry
                tp3 = entry_price - 3.0 * atr_at_entry
                lots_remaining = 3
                lot1_hit = lot2_hit = False
                n_entries += 1

    total_usd = pts_total * POINT_VALUE
    avg_atr = float(np.nanmean(atr))
    return {
        "entries": n_entries,
        "exits": len(lot_pnls_pts),
        "total_pts": pts_total,
        "total_usd": total_usd,
        "return_pct": (total_usd / ACCOUNT_SIZE) * 100.0,
        "wr": (wins / max(wins + losses, 1)) * 100.0,
        "pf": (sum(p for p in lot_pnls_pts if p > 0) /
               max(-sum(p for p in lot_pnls_pts if p < 0), 1e-9)),
        "avg_atr": avg_atr,
        "avg_stop_pts": 1.5 * avg_atr,
        "avg_stop_usd": 1.5 * avg_atr * POINT_VALUE,
        "worst_entry_pts": worst_entry_pts,
        "best_entry_pts": best_entry_pts,
        "equities": equities,
    }


# ---------- Driver --------------------------------------------------------
def run_pair(label, df_entry, df_slow):
    print(f"\n=== {label} ===")
    print(f"  Entry bars: {len(df_entry):,}  ({df_entry.index.min()} -> {df_entry.index.max()})")
    print(f"  Bias bars : {len(df_slow):,}   ({df_slow.index.min()} -> {df_slow.index.max()})")

    bias_slow = compute_bias(df_slow)
    bias_aligned = bias_slow.reindex(df_entry.index, method="ffill").fillna(0).astype(int).values
    buy = int((bias_aligned == 1).sum())
    sell = int((bias_aligned == -1).sum())
    neut = int((bias_aligned == 0).sum())
    print(f"  Bias dist on entry bars: BUY={buy}  SELL={sell}  NEUTRAL={neut}")

    stats = simulate(df_entry, bias_aligned)
    print(f"  Entries={stats['entries']}  Exits={stats['exits']}")
    print(f"  Gross  P&L = {stats['total_pts']:+,.2f} pts  =  ${stats['total_usd']:+,.0f}  ({stats['return_pct']:+.1f}%)")
    print(f"  WR={stats['wr']:.1f}%   PF={stats['pf']:.3f}   AvgATR={stats['avg_atr']:.2f} pts   AvgStop={stats['avg_stop_usd']:,.0f} $/lot")
    print(f"  WorstEntry={stats['worst_entry_pts']:+.1f} pts = ${stats['worst_entry_pts']*POINT_VALUE:+,.0f}")

    # Commission table
    rows = []
    for cname, c in COMMS:
        haircut = stats["entries"] * LOTS_PER_ENTRY * c
        net = stats["total_usd"] - haircut
        rows.append((cname, c, haircut, net))
        print(f"   {cname:<14}: haircut=${haircut:>10,.0f}   net = ${net:>+12,.0f}")
    return {
        "label": label,
        "entries": stats["entries"],
        "exits": stats["exits"],
        "gross_usd": stats["total_usd"],
        "return_pct": stats["return_pct"],
        "wr": stats["wr"],
        "pf": stats["pf"],
        "avg_atr": stats["avg_atr"],
        "avg_stop_usd": stats["avg_stop_usd"],
        "worst_entry_usd": stats["worst_entry_pts"] * POINT_VALUE,
        "comms": [(cn, c, h, n) for (cn, c, h, n) in rows],
    }


def main():
    # Load bars
    df_60m_raw = pd.read_parquet(HERE / "ES_60m_730d.parquet")
    df_1d_raw  = pd.read_parquet(HERE / "ES_1d_10y.parquet")
    df_1wk_raw = pd.read_parquet(HERE / "ES_1wk_10y.parquet")

    # Strip TZ on daily/weekly to keep alignment simple
    for d in (df_1d_raw, df_1wk_raw):
        if getattr(d.index, "tz", None) is not None:
            d.index = d.index.tz_localize(None)
    if getattr(df_60m_raw.index, "tz", None) is not None:
        df_60m_raw.index = df_60m_raw.index.tz_convert("UTC").tz_localize(None)

    print("Computing indicators...")
    df_60m = compute_all(df_60m_raw)
    df_1d  = compute_all(df_1d_raw)
    df_1wk = compute_all(df_1wk_raw)

    results = []
    results.append(run_pair(
        "Daily entries + Weekly bias (ES, 10yr yfinance)",
        df_entry=df_1d, df_slow=df_1wk,
    ))
    # For 60m+D, limit daily to the 60m window so alignment is clean
    daily_aligned = df_1d.loc[df_1d.index >= df_60m.index.min().normalize()]
    results.append(run_pair(
        "60m entries + Daily bias (ES, 2yr yfinance)",
        df_entry=df_60m, df_slow=daily_aligned,
    ))

    out = HERE / "slow_mtf_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved {out.name}")


if __name__ == "__main__":
    main()
