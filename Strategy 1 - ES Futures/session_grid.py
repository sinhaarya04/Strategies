"""
Full grid search (stop x target) across each session.
Shows which combo + session is actually profitable.
"""

import numpy as np
import pandas as pd

from data_feed import fetch_es_data
from indicators import compute_all
from config import (
    RSI_BUY_THRESHOLD, RSI_SELL_THRESHOLD,
    INITIAL_CAPITAL, ES_POINT_VALUE,
)

STOP_RANGE = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
TARGET_RANGE = [None, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]

SESSIONS = [
    ("All Day", 0, 0, 23, 59),
    ("Overnight (00:00-04:00)", 0, 0, 4, 0),
    ("US Morning (04:00-13:00)", 4, 0, 13, 0),
    ("US Afternoon (13:00-17:00)", 13, 0, 17, 0),
]


def simulate(df, atr_stop_mult, atr_target_mult, s_start_h, s_start_m, s_end_h, s_end_m):
    position = 0
    entry_price = stop_price = target_price = 0.0
    equity = INITIAL_CAPITAL
    trades = []
    equities = [equity]

    start_min = s_start_h * 60 + s_start_m
    end_min = s_end_h * 60 + s_end_m
    filter_all = (start_min == 0 and end_min == 23 * 60 + 59)

    for i in range(1, len(df)):
        ha_c = df["HA_Close"].iloc[i]
        ema_h = df["EMA_High"].iloc[i]
        ema_l = df["EMA_Low"].iloc[i]
        rsi = df["RSI"].iloc[i]
        atr = df["ATR"].iloc[i]
        close = df["Close"].iloc[i]

        if np.isnan(rsi) or np.isnan(ema_h) or np.isnan(atr):
            equities.append(equity)
            continue

        bar_min = df.index[i].hour * 60 + df.index[i].minute
        in_session = filter_all or (start_min <= bar_min < end_min)

        # Exits always active
        if position == 1 and close <= stop_price:
            equity += (close - entry_price) * ES_POINT_VALUE
            trades.append((close - entry_price) * ES_POINT_VALUE)
            position = 0
        elif position == -1 and close >= stop_price:
            equity += (entry_price - close) * ES_POINT_VALUE
            trades.append((entry_price - close) * ES_POINT_VALUE)
            position = 0

        if atr_target_mult is not None:
            if position == 1 and close >= target_price:
                equity += (close - entry_price) * ES_POINT_VALUE
                trades.append((close - entry_price) * ES_POINT_VALUE)
                position = 0
            elif position == -1 and close <= target_price:
                equity += (entry_price - close) * ES_POINT_VALUE
                trades.append((entry_price - close) * ES_POINT_VALUE)
                position = 0

        if position == 1:
            if ha_c < ema_l and rsi < RSI_SELL_THRESHOLD:
                equity += (close - entry_price) * ES_POINT_VALUE
                trades.append((close - entry_price) * ES_POINT_VALUE)
                position = 0
        elif position == -1:
            if ha_c > ema_h and rsi > RSI_BUY_THRESHOLD:
                equity += (entry_price - close) * ES_POINT_VALUE
                trades.append((entry_price - close) * ES_POINT_VALUE)
                position = 0

        # Entry only in session
        if position == 0 and in_session:
            if ha_c > ema_h and rsi > RSI_BUY_THRESHOLD:
                position = 1; entry_price = close
                stop_price = close - atr * atr_stop_mult
                target_price = close + atr * atr_target_mult if atr_target_mult else 0
            elif ha_c < ema_l and rsi < RSI_SELL_THRESHOLD:
                position = -1; entry_price = close
                stop_price = close + atr * atr_stop_mult
                target_price = close - atr * atr_target_mult if atr_target_mult else 0

        if position == 1:
            equities.append(equity + (close - entry_price) * ES_POINT_VALUE)
        elif position == -1:
            equities.append(equity + (entry_price - close) * ES_POINT_VALUE)
        else:
            equities.append(equity)

    return equities, trades


def compute_stats(trades, equities):
    if not trades:
        return None
    wins = [t for t in trades if t > 0]
    losses = [t for t in trades if t <= 0]
    gross_w = sum(wins) if wins else 0
    gross_l = abs(sum(losses)) if losses else 0.001
    pnl = equities[-1] - INITIAL_CAPITAL
    eq = np.array(equities)
    dd = ((np.maximum.accumulate(eq) - eq) / np.maximum.accumulate(eq) * 100).max()
    return {
        "pnl": pnl,
        "trades": len(trades),
        "wr": len(wins) / len(trades) * 100,
        "pf": gross_w / gross_l,
        "dd": dd,
    }


def main():
    print("Fetching data...")
    df = fetch_es_data()
    df = compute_all(df)

    total_combos = len(STOP_RANGE) * len(TARGET_RANGE) * len(SESSIONS)
    print(f"Running {total_combos} combos ({len(STOP_RANGE)} stops x {len(TARGET_RANGE)} targets x {len(SESSIONS)} sessions)\n")

    all_results = []

    for sess_name, sh, sm, eh, em in SESSIONS:
        for stop in STOP_RANGE:
            for target in TARGET_RANGE:
                eq, tr = simulate(df, stop, target, sh, sm, eh, em)
                s = compute_stats(tr, eq)
                if s is None:
                    continue
                t_label = "None" if target is None else f"{target:.2f}x"
                all_results.append({
                    "Session": sess_name,
                    "Stop": f"{stop:.2f}x",
                    "Target": t_label,
                    "P&L": s["pnl"],
                    "Trades": s["trades"],
                    "Win%": s["wr"],
                    "PF": s["pf"],
                    "MaxDD%": s["dd"],
                })

    tbl = pd.DataFrame(all_results)

    # Show profitable combos first
    profitable = tbl[tbl["PF"] >= 1.0].sort_values("PF", ascending=False)

    if len(profitable) > 0:
        print("=" * 95)
        print(f"  PROFITABLE COMBOS (PF >= 1.0) — {len(profitable)} found!")
        print("=" * 95)
        print(profitable.head(30).to_string(index=False, float_format="%.2f"))
    else:
        print("No profitable combos found.")

    # Top 15 per session
    for sess_name, _, _, _, _ in SESSIONS:
        subset = tbl[tbl["Session"] == sess_name].sort_values("PF", ascending=False).head(15)
        print(f"\n{'=' * 95}")
        print(f"  TOP 15: {sess_name}")
        print("=" * 95)
        print(subset.to_string(index=False, float_format="%.2f"))

    # Export full table to HTML
    tbl_sorted = tbl.sort_values("PF", ascending=False)

    # Color code rows
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
            "P&L": "${:+,.0f}",
            "Trades": "{:,.0f}",
            "Win%": "{:.1f}%",
            "PF": "{:.3f}",
            "MaxDD%": "{:.1f}%",
        })
        .set_properties(**{
            "font-family": "monospace",
            "font-size": "13px",
            "padding": "4px 8px",
            "border": "1px solid #333",
            "color": "#e0e0e0",
        })
        .set_table_styles([
            {"selector": "th", "props": [
                ("background-color", "#1a1a2e"),
                ("color", "#e0e0e0"),
                ("font-size", "13px"),
                ("padding", "6px 8px"),
                ("border", "1px solid #333"),
                ("position", "sticky"),
                ("top", "0"),
            ]},
            {"selector": "table", "props": [
                ("border-collapse", "collapse"),
                ("width", "100%"),
            ]},
        ])
    )

    html = f"""<!DOCTYPE html>
<html>
<head>
<title>ES 5m Session Grid — {len(tbl)} combos</title>
<style>
  body {{ background: #0d1117; color: #e0e0e0; font-family: monospace; padding: 20px; }}
  h1 {{ color: #58a6ff; font-size: 20px; }}
  h2 {{ color: #8b949e; font-size: 16px; margin-top: 5px; }}
  .summary {{ display: flex; gap: 20px; margin: 15px 0; }}
  .card {{ background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 12px 16px; }}
  .card .num {{ font-size: 24px; font-weight: bold; }}
  .green {{ color: #3fb950; }}
  .red {{ color: #f85149; }}
  .yellow {{ color: #d29922; }}
  table {{ margin-top: 10px; }}
</style>
</head>
<body>
<h1>ES E-mini 5m — Full Session Grid Search</h1>
<h2>{len(STOP_RANGE)} stops x {len(TARGET_RANGE)} targets x {len(SESSIONS)} sessions = {len(tbl)} combos</h2>

<div class="summary">
  <div class="card">
    <div class="green num">{len(profitable)}</div>
    <div>Profitable (PF &ge; 1.0)</div>
  </div>
  <div class="card">
    <div class="yellow num">{len(tbl[(tbl['PF'] >= 0.95) & (tbl['PF'] < 1.0)])}</div>
    <div>Near breakeven (0.95-1.0)</div>
  </div>
  <div class="card">
    <div class="red num">{len(tbl[tbl['PF'] < 0.95])}</div>
    <div>Losing (PF &lt; 0.95)</div>
  </div>
  <div class="card">
    <div class="green num">${tbl_sorted.iloc[0]['P&L']:+,.0f}</div>
    <div>Best P&amp;L: {tbl_sorted.iloc[0]['Session']} SL={tbl_sorted.iloc[0]['Stop']} TP={tbl_sorted.iloc[0]['Target']}</div>
  </div>
</div>

{styled.to_html()}
</body>
</html>"""

    output = "session_grid.html"
    with open(output, "w") as f:
        f.write(html)

    import webbrowser, os
    webbrowser.open("file://" + os.path.abspath(output))
    print(f"\nFull table saved to {output} and opened in browser.")


if __name__ == "__main__":
    main()
