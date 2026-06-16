"""
Strategy: Fisher Transform Crossover with Linear Stops/Targets
Entry: Fisher line crosses signal line (on Close or HA Close).
Stop/Target: Fixed point values (not ATR-based).
All P&L reported in POINTS.
"""

import numpy as np
import pandas as pd

from data_feed import fetch_es_data
from indicators import compute_all
from config import ES_POINT_VALUE

ACCOUNT_SIZE = 50_000.0

# Linear point-based grid
STOP_RANGE = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15]
TARGET_RANGE = [None, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15]

SESSIONS = [
    ("All Day", 0, 0, 23, 59),
    ("Overnight (00:00-04:00)", 0, 0, 4, 0),
    ("US Morning (04:00-13:00)", 4, 0, 13, 0),
    ("US Afternoon (13:00-17:00)", 13, 0, 17, 0),
]

FISHER_SOURCES = [
    ("Close", "Fisher", "Fisher_Signal"),
    ("HA Close", "Fisher_HA", "Fisher_HA_Signal"),
]

TIMEFRAMES = [
    ("5m", None),
    ("15m", "15min"),
]


def resample(df_5m, freq):
    return df_5m.resample(freq).agg({
        "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum",
    }).dropna()


def simulate(df, fisher_col, signal_col, stop_pts, target_pts,
             s_start_h, s_start_m, s_end_h, s_end_m):
    """Fisher crossover entries with fixed-point stops/targets. P&L in points."""
    position = 0
    entry_price = stop_price = target_price = 0.0
    equity_pts = 0.0

    trades = []
    equities = [equity_pts]

    start_min = s_start_h * 60 + s_start_m
    end_min = s_end_h * 60 + s_end_m
    filter_all = (start_min == 0 and end_min == 23 * 60 + 59)

    fisher = df[fisher_col].values
    signal = df[signal_col].values
    close = df["Close"].values

    for i in range(2, len(df)):
        f_now = fisher[i]
        s_now = signal[i]
        f_prev = fisher[i - 1]
        s_prev = signal[i - 1]
        c = close[i]

        if np.isnan(f_now) or np.isnan(s_now) or np.isnan(f_prev) or np.isnan(s_prev):
            equities.append(equity_pts)
            continue

        bar_min = df.index[i].hour * 60 + df.index[i].minute
        in_session = filter_all or (start_min <= bar_min < end_min)

        # Exits always active
        if position == 1 and c <= stop_price:
            equity_pts += c - entry_price
            trades.append(c - entry_price)
            position = 0
        elif position == -1 and c >= stop_price:
            equity_pts += entry_price - c
            trades.append(entry_price - c)
            position = 0

        if target_pts is not None:
            if position == 1 and c >= target_price:
                equity_pts += c - entry_price
                trades.append(c - entry_price)
                position = 0
            elif position == -1 and c <= target_price:
                equity_pts += entry_price - c
                trades.append(entry_price - c)
                position = 0

        # Reverse signal exit
        if position == 1:
            if f_now < s_now and f_prev >= s_prev:
                equity_pts += c - entry_price
                trades.append(c - entry_price)
                position = 0
        elif position == -1:
            if f_now > s_now and f_prev <= s_prev:
                equity_pts += entry_price - c
                trades.append(entry_price - c)
                position = 0

        # Entry: Fisher crossover, in session only
        if position == 0 and in_session:
            if f_now > s_now and f_prev <= s_prev:
                position = 1
                entry_price = c
                stop_price = c - stop_pts
                target_price = c + target_pts if target_pts else 0
            elif f_now < s_now and f_prev >= s_prev:
                position = -1
                entry_price = c
                stop_price = c + stop_pts
                target_price = c - target_pts if target_pts else 0

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

    # Prepare both timeframes
    frames = {}
    for tf_label, resample_freq in TIMEFRAMES:
        if resample_freq:
            raw = resample(df_5m_raw, resample_freq)
        else:
            raw = df_5m_raw
        frames[tf_label] = compute_all(raw)
        print(f"{tf_label}: {len(frames[tf_label])} bars")

    total = len(STOP_RANGE) * len(TARGET_RANGE) * len(SESSIONS) * len(FISHER_SOURCES) * len(TIMEFRAMES)
    print(f"Running {total} combos...\n")

    all_results = []

    for tf_label, _ in TIMEFRAMES:
        df = frames[tf_label]
        for src_name, f_col, s_col in FISHER_SOURCES:
            for sess_name, sh, sm, eh, em in SESSIONS:
                for stop in STOP_RANGE:
                    for target in TARGET_RANGE:
                        eq, tr = simulate(df, f_col, s_col, stop, target, sh, sm, eh, em)
                        s = compute_stats(tr, eq)
                        if s is None:
                            continue
                        t_label = "None" if target is None else f"{target}pt"
                        dollar_pnl = s["pts"] * ES_POINT_VALUE
                        dollar_dd = s["dd_pts"] * ES_POINT_VALUE
                        ret_pct = dollar_pnl / ACCOUNT_SIZE * 100
                        all_results.append({
                            "TF": tf_label,
                            "Source": src_name,
                            "Session": sess_name,
                            "Stop": f"{stop}pt",
                            "Target": t_label,
                            "Points": s["pts"],
                            "$P&L(50k)": dollar_pnl,
                            "Return%": ret_pct,
                            "Trades": s["trades"],
                            "Win%": s["wr"],
                            "PF": s["pf"],
                            "MaxDD(pts)": s["dd_pts"],
                            "$MaxDD": dollar_dd,
                            "AvgW(pts)": s["avg_w"],
                            "AvgL(pts)": s["avg_l"],
                        })

    tbl = pd.DataFrame(all_results)
    profitable = tbl[tbl["PF"] >= 1.0].sort_values("PF", ascending=False)

    if len(profitable) > 0:
        print("=" * 120)
        print(f"  FISHER TRANSFORM — PROFITABLE COMBOS (PF >= 1.0) — {len(profitable)} of {len(tbl)}  [POINTS]")
        print("=" * 120)
        print(profitable.head(30).to_string(index=False, float_format="%.2f"))
    else:
        print("No profitable combos found.")

    # Top per source + timeframe
    for tf_label, _ in TIMEFRAMES:
        for src_name, _, _ in FISHER_SOURCES:
            subset = tbl[(tbl["TF"] == tf_label) & (tbl["Source"] == src_name)]
            top = subset.sort_values("PF", ascending=False).head(10)
            print(f"\n{'=' * 120}")
            print(f"  TOP 10: {tf_label} / {src_name}  [POINTS]")
            print("=" * 120)
            print(top.to_string(index=False, float_format="%.2f"))

    # HTML output
    build_html(tbl, profitable)


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
<html><head><title>Fisher Transform Grid — {len(tbl_sorted)} combos</title>
<style>
  body {{ background: #0d1117; color: #e0e0e0; font-family: monospace; padding: 20px; }}
  h1 {{ color: #e599f7; font-size: 20px; }}
  h2 {{ color: #8b949e; font-size: 16px; margin-top: 5px; }}
  .summary {{ display: flex; gap: 20px; margin: 15px 0; flex-wrap: wrap; }}
  .card {{ background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 12px 16px; }}
  .card .num {{ font-size: 24px; font-weight: bold; }}
  .green {{ color: #3fb950; }} .red {{ color: #f85149; }} .yellow {{ color: #d29922; }} .purple {{ color: #e599f7; }}
</style></head><body>
<h1>Fisher Transform Crossover — Linear Stops/Targets [ALL P&L IN POINTS]</h1>
<h2>Entry: Fisher crosses signal | Stops/targets in fixed points | Close + HA Close variants | 5m + 15m</h2>
<h2>{len(tbl_sorted)} combos | {len(STOP_RANGE)} stops x {len(TARGET_RANGE)} targets x {len(SESSIONS)} sessions x 2 sources x 2 TFs</h2>
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

    output = "backtest_fisher_chart.html"
    with open(output, "w") as f:
        f.write(html)

    import webbrowser, os
    webbrowser.open("file://" + os.path.abspath(output))
    print(f"\nChart saved to {output} and opened in browser.")


if __name__ == "__main__":
    main()
