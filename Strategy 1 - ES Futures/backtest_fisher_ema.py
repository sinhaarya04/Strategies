"""
Strategy: HA EMA Band + Fisher Transform (replaces RSI)
60m bias filter, 5m and 15m entries.
Entry: HA close crosses EMA band AND Fisher crosses signal line (same direction).
Stops/targets: both ATR-based and linear point-based grids.
All P&L in POINTS.
"""

import numpy as np
import pandas as pd

from data_feed import fetch_es_data
from indicators import compute_all
from config import ES_POINT_VALUE

ACCOUNT_SIZE = 50_000.0

# ATR-based grid
ATR_STOP_RANGE = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]
ATR_TARGET_RANGE = [None, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

# Linear point-based grid
PT_STOP_RANGE = [2, 3, 5, 8, 10, 15]
PT_TARGET_RANGE = [None, 2, 3, 5, 8, 10, 15]

SESSIONS = [
    ("All Day", 0, 0, 23, 59),
    ("Overnight (00:00-04:00)", 0, 0, 4, 0),
    ("US Morning (04:00-13:00)", 4, 0, 13, 0),
    ("US Afternoon (13:00-17:00)", 13, 0, 17, 0),
]

TIMEFRAMES = [
    ("5m", None),
    ("15m", "15min"),
]

# Fisher on HA Close won the standalone test — use both here to confirm
FISHER_SOURCES = [
    ("Fisher(Close)", "Fisher", "Fisher_Signal"),
    ("Fisher(HA)", "Fisher_HA", "Fisher_HA_Signal"),
]


def resample(df_5m, freq):
    return df_5m.resample(freq).agg({
        "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum",
    }).dropna()


def get_60m_bias(df_60m):
    """60m directional bias using HA EMA + Fisher (instead of RSI)."""
    bias = pd.Series(0, index=df_60m.index, dtype=int)
    fisher = df_60m["Fisher_HA"].values
    signal = df_60m["Fisher_HA_Signal"].values
    ha_c = df_60m["HA_Close"].values
    ema_h = df_60m["EMA_High"].values
    ema_l = df_60m["EMA_Low"].values

    for i in range(1, len(df_60m)):
        if np.isnan(ema_h[i]) or np.isnan(fisher[i]) or np.isnan(signal[i]):
            continue
        # Bullish: HA close > EMA high AND Fisher > Signal
        if ha_c[i] > ema_h[i] and fisher[i] > signal[i]:
            bias.iloc[i] = 1
        # Bearish: HA close < EMA low AND Fisher < Signal
        elif ha_c[i] < ema_l[i] and fisher[i] < signal[i]:
            bias.iloc[i] = -1
    return bias


def align_bias(bias_60m, df_lower):
    """Map 60m bias to lower timeframe bars."""
    aligned = pd.Series(0, index=df_lower.index, dtype=int)
    bias_times = bias_60m.index
    for i, ts in enumerate(df_lower.index):
        mask = bias_times <= ts
        if mask.any():
            aligned.iloc[i] = bias_60m.loc[bias_times[mask][-1]]
    return aligned


def simulate(df, bias, f_col, s_col, stop_val, target_val, is_atr,
             s_start_h, s_start_m, s_end_h, s_end_m):
    """
    Entry: HA close crosses EMA band AND Fisher crossover, aligned with 60m bias.
    Stop/target: ATR-based or fixed points depending on is_atr flag.
    """
    position = 0
    entry_price = stop_price = target_price = 0.0
    equity_pts = 0.0

    trades = []
    equities = [equity_pts]

    start_min = s_start_h * 60 + s_start_m
    end_min = s_end_h * 60 + s_end_m
    filter_all = (start_min == 0 and end_min == 23 * 60 + 59)

    ha_c = df["HA_Close"].values
    ema_h = df["EMA_High"].values
    ema_l = df["EMA_Low"].values
    atr = df["ATR"].values
    close = df["Close"].values
    fisher = df[f_col].values
    signal = df[s_col].values
    bias_arr = bias.values

    for i in range(2, len(df)):
        c = close[i]
        f_now, s_now = fisher[i], signal[i]
        f_prev, s_prev = fisher[i - 1], signal[i - 1]

        if np.isnan(ema_h[i]) or np.isnan(f_now) or np.isnan(s_now) or np.isnan(f_prev) or np.isnan(atr[i]):
            equities.append(equity_pts)
            continue

        bar_min = df.index[i].hour * 60 + df.index[i].minute
        in_session = filter_all or (start_min <= bar_min < end_min)
        b = bias_arr[i]

        # --- Exits (always active) ---
        if position == 1 and c <= stop_price:
            equity_pts += c - entry_price
            trades.append(c - entry_price)
            position = 0
        elif position == -1 and c >= stop_price:
            equity_pts += entry_price - c
            trades.append(entry_price - c)
            position = 0

        if target_val is not None:
            if position == 1 and c >= target_price:
                equity_pts += c - entry_price
                trades.append(c - entry_price)
                position = 0
            elif position == -1 and c <= target_price:
                equity_pts += entry_price - c
                trades.append(entry_price - c)
                position = 0

        # Reversal exit: opposite EMA cross + Fisher cross
        if position == 1:
            if ha_c[i] < ema_l[i] and f_now < s_now and f_prev >= s_prev:
                equity_pts += c - entry_price
                trades.append(c - entry_price)
                position = 0
        elif position == -1:
            if ha_c[i] > ema_h[i] and f_now > s_now and f_prev <= s_prev:
                equity_pts += entry_price - c
                trades.append(entry_price - c)
                position = 0

        # --- Entry: EMA cross + Fisher crossover + 60m bias alignment ---
        if position == 0 and in_session:
            # Long: HA close > EMA high, Fisher crosses above signal, bias bullish
            if (b == 1 and ha_c[i] > ema_h[i] and
                    f_now > s_now and f_prev <= s_prev):
                position = 1
                entry_price = c
                if is_atr:
                    stop_price = c - atr[i] * stop_val
                    target_price = c + atr[i] * target_val if target_val else 0
                else:
                    stop_price = c - stop_val
                    target_price = c + target_val if target_val else 0

            # Short: HA close < EMA low, Fisher crosses below signal, bias bearish
            elif (b == -1 and ha_c[i] < ema_l[i] and
                  f_now < s_now and f_prev >= s_prev):
                position = -1
                entry_price = c
                if is_atr:
                    stop_price = c + atr[i] * stop_val
                    target_price = c - atr[i] * target_val if target_val else 0
                else:
                    stop_price = c + stop_val
                    target_price = c - target_val if target_val else 0

        # Mark-to-market
        if position == 1:
            equities.append(equity_pts + (c - entry_price))
        elif position == -1:
            equities.append(equity_pts + (entry_price - c))
        else:
            equities.append(equity_pts)

    return equities, trades


def compute_stats(trades, equities):
    if not trades:
        return None
    wins = [t for t in trades if t > 0]
    losses = [t for t in trades if t <= 0]
    gross_w = sum(wins) if wins else 0
    gross_l = abs(sum(losses)) if losses else 0.001
    eq = np.array(equities)
    peak = np.maximum.accumulate(eq)
    dd_pts = (peak - eq).max()
    return {
        "pts": equities[-1],
        "trades": len(trades),
        "wr": len(wins) / len(trades) * 100,
        "pf": gross_w / gross_l,
        "dd_pts": dd_pts,
        "avg_w": np.mean(wins) if wins else 0,
        "avg_l": np.mean(losses) if losses else 0,
    }


def main():
    print("Fetching 5m data...")
    df_5m_raw = fetch_es_data()
    df_60m_raw = resample(df_5m_raw, "60min")
    df_60m = compute_all(df_60m_raw)

    # 60m bias using Fisher(HA) on 60m
    bias_60m = get_60m_bias(df_60m)

    # Prepare entry timeframes
    frames = {}
    for tf_label, resample_freq in TIMEFRAMES:
        if resample_freq:
            raw = resample(df_5m_raw, resample_freq)
        else:
            raw = df_5m_raw
        df = compute_all(raw)
        b = align_bias(bias_60m, df)
        frames[tf_label] = (df, b)
        buy_n = (b == 1).sum()
        sell_n = (b == -1).sum()
        neutral_n = (b == 0).sum()
        print(f"{tf_label}: {len(df)} bars | 60m bias: BUY={buy_n} SELL={sell_n} NEUTRAL={neutral_n}")

    # Count combos
    n_atr = len(ATR_STOP_RANGE) * len(ATR_TARGET_RANGE)
    n_pt = len(PT_STOP_RANGE) * len(PT_TARGET_RANGE)
    n_per_slice = (n_atr + n_pt) * len(SESSIONS)
    total = n_per_slice * len(FISHER_SOURCES) * len(TIMEFRAMES)
    print(f"Running {total} combos ({n_atr} ATR + {n_pt} linear) x {len(SESSIONS)} sessions x {len(FISHER_SOURCES)} sources x {len(TIMEFRAMES)} TFs\n")

    all_results = []

    for tf_label, _ in TIMEFRAMES:
        df, bias = frames[tf_label]
        for src_name, f_col, s_col in FISHER_SOURCES:
            for sess_name, sh, sm, eh, em in SESSIONS:
                # ATR-based combos
                for stop in ATR_STOP_RANGE:
                    for target in ATR_TARGET_RANGE:
                        eq, tr = simulate(df, bias, f_col, s_col, stop, target, True, sh, sm, eh, em)
                        s = compute_stats(tr, eq)
                        if s is None:
                            continue
                        t_label = "None" if target is None else f"{target:.1f}x"
                        all_results.append(_row(tf_label, src_name, sess_name, f"{stop:.2f}x", t_label, "ATR", s))

                # Linear point-based combos
                for stop in PT_STOP_RANGE:
                    for target in PT_TARGET_RANGE:
                        eq, tr = simulate(df, bias, f_col, s_col, stop, target, False, sh, sm, eh, em)
                        s = compute_stats(tr, eq)
                        if s is None:
                            continue
                        t_label = "None" if target is None else f"{target}pt"
                        all_results.append(_row(tf_label, src_name, sess_name, f"{stop}pt", t_label, "Fixed", s))

    tbl = pd.DataFrame(all_results)
    profitable = tbl[tbl["PF"] >= 1.0].sort_values("PF", ascending=False)

    # Print top results
    if len(profitable) > 0:
        print("=" * 130)
        print(f"  HA EMA + FISHER (replaces RSI) — PROFITABLE COMBOS (PF >= 1.0) — {len(profitable)} of {len(tbl)}  [POINTS]")
        print("=" * 130)
        print(profitable.head(30).to_string(index=False, float_format="%.2f"))
    else:
        print("No profitable combos found.")

    # Top per TF + source
    for tf_label, _ in TIMEFRAMES:
        for src_name, _, _ in FISHER_SOURCES:
            subset = tbl[(tbl["TF"] == tf_label) & (tbl["Source"] == src_name)]
            top = subset.sort_values("PF", ascending=False).head(10)
            print(f"\n{'=' * 130}")
            print(f"  TOP 10: {tf_label} / {src_name}  [POINTS]")
            print("=" * 130)
            print(top.to_string(index=False, float_format="%.2f"))

    # Compare vs RSI baseline
    print(f"\n{'=' * 130}")
    print("  REFERENCE: Best RSI-based MTF combos were PF 2.09 (60m+5m, US Morning, SL=3.0x, no TP)")
    print("=" * 130)

    build_html(tbl, profitable)


def _row(tf, src, sess, stop, target, mode, s):
    return {
        "TF": tf,
        "Source": src,
        "Session": sess,
        "Stop": stop,
        "Target": target,
        "Mode": mode,
        "Points": s["pts"],
        "$P&L(50k)": s["pts"] * ES_POINT_VALUE,
        "Return%": s["pts"] * ES_POINT_VALUE / ACCOUNT_SIZE * 100,
        "Trades": s["trades"],
        "Win%": s["wr"],
        "PF": s["pf"],
        "MaxDD(pts)": s["dd_pts"],
        "$MaxDD": s["dd_pts"] * ES_POINT_VALUE,
        "AvgW(pts)": s["avg_w"],
        "AvgL(pts)": s["avg_l"],
    }


def build_html(tbl, profitable):
    tbl_sorted = tbl.sort_values("PF", ascending=False)

    def color_row(row):
        if row["PF"] >= 1.05:
            return ["background-color: #1a4d1a"] * len(row)
        elif row["PF"] >= 1.0:
            return ["background-color: #2d3d1a"] * len(row)
        elif row["PF"] >= 0.95:
            return ["background-color: #3d3d1a"] * len(row)
        else:
            return ["background-color: #4d1a1a"] * len(row)

    styled = (tbl_sorted.style
        .apply(color_row, axis=1)
        .format({
            "Points": "{:+,.1f} pts",
            "$P&L(50k)": "${:+,.0f}",
            "Return%": "{:+.1f}%",
            "Trades": "{:,.0f}",
            "Win%": "{:.1f}%",
            "PF": "{:.3f}",
            "MaxDD(pts)": "{:.1f} pts",
            "$MaxDD": "${:,.0f}",
            "AvgW(pts)": "{:+.2f} pts",
            "AvgL(pts)": "{:+.2f} pts",
        })
        .set_properties(**{
            "font-family": "monospace", "font-size": "13px",
            "padding": "4px 8px", "border": "1px solid #333", "color": "#e0e0e0",
        })
        .set_table_styles([
            {"selector": "th", "props": [
                ("background-color", "#1a1a2e"), ("color", "#e0e0e0"),
                ("font-size", "13px"), ("padding", "6px 8px"),
                ("border", "1px solid #333"), ("position", "sticky"), ("top", "0"),
            ]},
            {"selector": "table", "props": [("border-collapse", "collapse"), ("width", "100%")]},
        ])
    )

    best_pf = tbl_sorted.iloc[0]
    best_pnl = tbl_sorted.sort_values("$P&L(50k)", ascending=False).iloc[0]

    html = f"""<!DOCTYPE html>
<html><head><title>HA EMA + Fisher Transform — {len(tbl_sorted)} combos</title>
<style>
  body {{ background: #0d1117; color: #e0e0e0; font-family: monospace; padding: 20px; }}
  h1 {{ color: #e599f7; font-size: 20px; }}
  h2 {{ color: #8b949e; font-size: 16px; margin-top: 5px; }}
  .summary {{ display: flex; gap: 20px; margin: 15px 0; flex-wrap: wrap; }}
  .card {{ background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 12px 16px; }}
  .card .num {{ font-size: 24px; font-weight: bold; }}
  .green {{ color: #3fb950; }} .red {{ color: #f85149; }} .yellow {{ color: #d29922; }} .purple {{ color: #e599f7; }}
</style></head><body>
<h1>HA EMA Band + Fisher Transform (replaces RSI) | 60m Bias Filter</h1>
<h2>Entry: HA cross EMA band + Fisher crossover + 60m bias | ATR + Fixed-point stops/targets</h2>
<h2>{len(tbl_sorted)} combos | 5m + 15m | Fisher on Close + HA Close</h2>
<div class="summary">
  <div class="card"><div class="green num">{len(profitable)}</div><div>Profitable (PF &ge; 1.0)</div></div>
  <div class="card"><div class="yellow num">{len(tbl_sorted[(tbl_sorted['PF'] >= 0.95) & (tbl_sorted['PF'] < 1.0)])}</div><div>Near breakeven</div></div>
  <div class="card"><div class="red num">{len(tbl_sorted[tbl_sorted['PF'] < 0.95])}</div><div>Losing (PF &lt; 0.95)</div></div>
  <div class="card"><div class="purple num">PF {best_pf['PF']:.2f}</div><div>Best PF: {best_pf['TF']} {best_pf['Source']} {best_pf['Session']} SL={best_pf['Stop']} TP={best_pf['Target']}</div></div>
  <div class="card"><div class="green num">${best_pnl['$P&L(50k)']:+,.0f}</div><div>Best P&amp;L: {best_pnl['TF']} {best_pnl['Source']} {best_pnl['Session']} SL={best_pnl['Stop']} TP={best_pnl['Target']}</div></div>
  <div class="card"><div class="{'green' if best_pnl['Return%'] > 0 else 'red'} num">{best_pnl['Return%']:+.1f}%</div><div>Best Return on $50k</div></div>
</div>
{styled.to_html()}
</body></html>"""

    output = "backtest_fisher_ema_chart.html"
    with open(output, "w") as f:
        f.write(html)

    import webbrowser, os
    webbrowser.open("file://" + os.path.abspath(output))
    print(f"\nChart saved to {output} and opened in browser.")


if __name__ == "__main__":
    main()
