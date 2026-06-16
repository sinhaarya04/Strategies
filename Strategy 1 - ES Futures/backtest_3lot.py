"""
Strategy 3: 3-Lot Scaled Exit (1/2/3 ATR runner)
Entry: 3 lots when HA close crosses above/below EMA band (NO RSI filter).
Stop: 1 ATR from entry.
  Lot 1: TP at 1 ATR → move stop to breakeven on lots 2 & 3
  Lot 2: TP at 2 ATR → lot 3 stop stays at breakeven
  Lot 3: TP at 3 ATR or stopped at breakeven
All P&L reported in POINTS (not dollars).
"""

import numpy as np
import pandas as pd

from data_feed import fetch_es_data
from indicators import compute_all

SESSIONS = [
    ("All Day", 0, 0, 23, 59),
    ("Overnight (00:00-04:00)", 0, 0, 4, 0),
    ("US Morning (04:00-13:00)", 4, 0, 13, 0),
    ("US Afternoon (13:00-17:00)", 13, 0, 17, 0),
]

TIMEFRAMES = ["15min", "60min"]


def resample(df_5m, freq):
    return df_5m.resample(freq).agg({
        "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum",
    }).dropna()


def simulate_3lot(df, s_start_h, s_start_m, s_end_h, s_end_m):
    """
    3-lot scaled exit strategy. Returns per-trade log and equity curve (in points).

    Each trade enters 3 lots. Exits are:
      Lot 1: +1 ATR  (then stop → breakeven for lots 2,3)
      Lot 2: +2 ATR  (lot 3 stop stays breakeven)
      Lot 3: +3 ATR  or stopped at breakeven
    """
    start_min = s_start_h * 60 + s_start_m
    end_min = s_end_h * 60 + s_end_m
    filter_all = (start_min == 0 and end_min == 23 * 60 + 59)

    # State
    direction = 0       # +1 long, -1 short, 0 flat
    entry_price = 0.0
    atr_at_entry = 0.0
    lots_remaining = 0  # 3, 2, or 1
    stop_price = 0.0
    # TP levels for each lot
    tp1 = tp2 = tp3 = 0.0
    lot1_hit = lot2_hit = False

    equity_pts = 0.0
    equities = [0.0]
    trade_log = []  # list of dicts

    for i in range(1, len(df)):
        ha_c = df["HA_Close"].iloc[i]
        ema_h = df["EMA_High"].iloc[i]
        ema_l = df["EMA_Low"].iloc[i]
        atr = df["ATR"].iloc[i]
        close = df["Close"].iloc[i]
        high = df["High"].iloc[i]
        low = df["Low"].iloc[i]

        if np.isnan(ema_h) or np.isnan(atr) or atr == 0:
            equities.append(equity_pts)
            continue

        bar_min = df.index[i].hour * 60 + df.index[i].minute
        in_session = filter_all or (start_min <= bar_min < end_min)

        # ---- EXIT LOGIC (always active) ----
        if direction != 0 and lots_remaining > 0:
            if direction == 1:  # LONG
                # Check stop first
                if low <= stop_price:
                    # All remaining lots stopped
                    pts_per_lot = stop_price - entry_price
                    total_pts = pts_per_lot * lots_remaining
                    equity_pts += total_pts
                    trade_log.append({
                        "entry_time": entry_time,
                        "exit_time": df.index[i],
                        "direction": "LONG",
                        "entry": entry_price,
                        "exit": stop_price,
                        "lots_closed": lots_remaining,
                        "pts_per_lot": pts_per_lot,
                        "total_pts": total_pts,
                        "exit_type": "STOP" if not lot1_hit else "BE_STOP",
                    })
                    direction = 0
                    lots_remaining = 0
                else:
                    # Check TP levels (lot 3 first so we don't double-count)
                    if lots_remaining >= 3 and not lot1_hit and high >= tp1:
                        pts = tp1 - entry_price
                        equity_pts += pts
                        trade_log.append({
                            "entry_time": entry_time,
                            "exit_time": df.index[i],
                            "direction": "LONG",
                            "entry": entry_price,
                            "exit": tp1,
                            "lots_closed": 1,
                            "pts_per_lot": pts,
                            "total_pts": pts,
                            "exit_type": "TP1",
                        })
                        lot1_hit = True
                        lots_remaining = 2
                        stop_price = entry_price  # move to breakeven

                    if lots_remaining >= 2 and lot1_hit and not lot2_hit and high >= tp2:
                        pts = tp2 - entry_price
                        equity_pts += pts
                        trade_log.append({
                            "entry_time": entry_time,
                            "exit_time": df.index[i],
                            "direction": "LONG",
                            "entry": entry_price,
                            "exit": tp2,
                            "lots_closed": 1,
                            "pts_per_lot": pts,
                            "total_pts": pts,
                            "exit_type": "TP2",
                        })
                        lot2_hit = True
                        lots_remaining = 1

                    if lots_remaining >= 1 and lot2_hit and high >= tp3:
                        pts = tp3 - entry_price
                        equity_pts += pts
                        trade_log.append({
                            "entry_time": entry_time,
                            "exit_time": df.index[i],
                            "direction": "LONG",
                            "entry": entry_price,
                            "exit": tp3,
                            "lots_closed": 1,
                            "pts_per_lot": pts,
                            "total_pts": pts,
                            "exit_type": "TP3",
                        })
                        lots_remaining = 0
                        direction = 0

            elif direction == -1:  # SHORT
                if high >= stop_price:
                    pts_per_lot = entry_price - stop_price
                    total_pts = pts_per_lot * lots_remaining
                    equity_pts += total_pts
                    trade_log.append({
                        "entry_time": entry_time,
                        "exit_time": df.index[i],
                        "direction": "SHORT",
                        "entry": entry_price,
                        "exit": stop_price,
                        "lots_closed": lots_remaining,
                        "pts_per_lot": pts_per_lot,
                        "total_pts": total_pts,
                        "exit_type": "STOP" if not lot1_hit else "BE_STOP",
                    })
                    direction = 0
                    lots_remaining = 0
                else:
                    if lots_remaining >= 3 and not lot1_hit and low <= tp1:
                        pts = entry_price - tp1
                        equity_pts += pts
                        trade_log.append({
                            "entry_time": entry_time,
                            "exit_time": df.index[i],
                            "direction": "SHORT",
                            "entry": entry_price,
                            "exit": tp1,
                            "lots_closed": 1,
                            "pts_per_lot": pts,
                            "total_pts": pts,
                            "exit_type": "TP1",
                        })
                        lot1_hit = True
                        lots_remaining = 2
                        stop_price = entry_price

                    if lots_remaining >= 2 and lot1_hit and not lot2_hit and low <= tp2:
                        pts = entry_price - tp2
                        equity_pts += pts
                        trade_log.append({
                            "entry_time": entry_time,
                            "exit_time": df.index[i],
                            "direction": "SHORT",
                            "entry": entry_price,
                            "exit": tp2,
                            "lots_closed": 1,
                            "pts_per_lot": pts,
                            "total_pts": pts,
                            "exit_type": "TP2",
                        })
                        lot2_hit = True
                        lots_remaining = 1

                    if lots_remaining >= 1 and lot2_hit and low <= tp3:
                        pts = entry_price - tp3
                        equity_pts += pts
                        trade_log.append({
                            "entry_time": entry_time,
                            "exit_time": df.index[i],
                            "direction": "SHORT",
                            "entry": entry_price,
                            "exit": tp3,
                            "lots_closed": 1,
                            "pts_per_lot": pts,
                            "total_pts": pts,
                            "exit_type": "TP3",
                        })
                        lots_remaining = 0
                        direction = 0

        # ---- ENTRY LOGIC (no RSI filter, just HA cross over/under EMA band) ----
        if direction == 0 and in_session:
            if ha_c > ema_h:
                direction = 1
                entry_price = close
                atr_at_entry = atr
                lots_remaining = 3
                lot1_hit = lot2_hit = False
                stop_price = close - atr
                tp1 = close + 1 * atr
                tp2 = close + 2 * atr
                tp3 = close + 3 * atr
                entry_time = df.index[i]
            elif ha_c < ema_l:
                direction = -1
                entry_price = close
                atr_at_entry = atr
                lots_remaining = 3
                lot1_hit = lot2_hit = False
                stop_price = close + atr
                tp1 = close - 1 * atr
                tp2 = close - 2 * atr
                tp3 = close - 3 * atr
                entry_time = df.index[i]

        # Mark-to-market
        if direction == 1 and lots_remaining > 0:
            equities.append(equity_pts + (close - entry_price) * lots_remaining)
        elif direction == -1 and lots_remaining > 0:
            equities.append(equity_pts + (entry_price - close) * lots_remaining)
        else:
            equities.append(equity_pts)

    return equities, trade_log


def summarize_trades(trade_log):
    """Compute summary stats from trade log."""
    if not trade_log:
        return None

    all_pts = [t["total_pts"] for t in trade_log]
    wins = [p for p in all_pts if p > 0]
    losses = [p for p in all_pts if p <= 0]
    gross_w = sum(wins) if wins else 0
    gross_l = abs(sum(losses)) if losses else 0.001

    # Count full "trade groups" (each entry spawns up to 3 exit events)
    entry_times = set(t["entry_time"] for t in trade_log)
    n_entries = len(entry_times)

    # Exit type breakdown
    tp1_count = sum(1 for t in trade_log if t["exit_type"] == "TP1")
    tp2_count = sum(1 for t in trade_log if t["exit_type"] == "TP2")
    tp3_count = sum(1 for t in trade_log if t["exit_type"] == "TP3")
    stop_count = sum(1 for t in trade_log if t["exit_type"] == "STOP")
    be_count = sum(1 for t in trade_log if t["exit_type"] == "BE_STOP")

    total_pts = sum(all_pts)

    return {
        "entries": n_entries,
        "exit_events": len(trade_log),
        "total_pts": total_pts,
        "avg_pts_per_entry": total_pts / n_entries if n_entries else 0,
        "wr": len(wins) / len(all_pts) * 100 if all_pts else 0,
        "pf": gross_w / gross_l,
        "tp1": tp1_count,
        "tp2": tp2_count,
        "tp3": tp3_count,
        "stop": stop_count,
        "be_stop": be_count,
    }


def main():
    print("Fetching 5m data...")
    df_5m = fetch_es_data()

    all_results = []

    for tf_label in TIMEFRAMES:
        df_resampled = resample(df_5m, tf_label)
        df = compute_all(df_resampled)
        print(f"\n{'='*80}")
        print(f"  Timeframe: {tf_label} | {len(df)} bars")
        print(f"{'='*80}")

        for sess_name, sh, sm, eh, em in SESSIONS:
            eq, log = simulate_3lot(df, sh, sm, eh, em)
            stats = summarize_trades(log)
            if stats is None:
                continue

            print(f"\n  {sess_name} ({tf_label})")
            print(f"    Entries: {stats['entries']}  |  Exit events: {stats['exit_events']}")
            print(f"    Total P&L: {stats['total_pts']:+.2f} pts  |  Avg per entry: {stats['avg_pts_per_entry']:+.2f} pts")
            print(f"    Win%: {stats['wr']:.1f}%  |  PF: {stats['pf']:.3f}")
            print(f"    TP1: {stats['tp1']}  TP2: {stats['tp2']}  TP3: {stats['tp3']}  Stop: {stats['stop']}  BE Stop: {stats['be_stop']}")

            all_results.append({
                "TF": tf_label,
                "Session": sess_name,
                "Entries": stats["entries"],
                "Exits": stats["exit_events"],
                "Total(pts)": stats["total_pts"],
                "Avg/Entry(pts)": stats["avg_pts_per_entry"],
                "Win%": stats["wr"],
                "PF": stats["pf"],
                "TP1": stats["tp1"],
                "TP2": stats["tp2"],
                "TP3": stats["tp3"],
                "Stop": stats["stop"],
                "BE_Stop": stats["be_stop"],
                "equities": eq,
            })

    # ---- Summary table ----
    tbl = pd.DataFrame([{k: v for k, v in r.items() if k != "equities"} for r in all_results])
    print(f"\n{'='*110}")
    print("  3-LOT SCALED EXIT — ALL COMBOS  [P&L in POINTS]")
    print(f"{'='*110}")
    print(tbl.sort_values("PF", ascending=False).to_string(index=False, float_format="%.2f"))

    # ---- HTML chart with equity curves + table ----
    build_html(all_results, tbl)


def build_html(all_results, tbl):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # One equity subplot per timeframe
    tf_list = list(dict.fromkeys(r["TF"] for r in all_results))
    fig = make_subplots(
        rows=len(tf_list), cols=1,
        subplot_titles=[f"Equity Curve — {tf}" for tf in tf_list],
        vertical_spacing=0.12,
    )

    colors = ["#58a6ff", "#3fb950", "#d29922", "#f85149"]

    for row_idx, tf in enumerate(tf_list, 1):
        subset = [r for r in all_results if r["TF"] == tf]
        for j, r in enumerate(subset):
            fig.add_trace(go.Scatter(
                y=r["equities"],
                mode="lines",
                name=f"{r['Session']} ({tf})",
                line=dict(color=colors[j % len(colors)], width=1.5),
                showlegend=True,
            ), row=row_idx, col=1)
        fig.update_yaxes(title_text="Points", row=row_idx, col=1)

    fig.update_layout(
        template="plotly_dark",
        height=400 * len(tf_list),
        title="Strategy 3: 3-Lot Scaled Exit — Equity Curves [POINTS]",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    chart_html = fig.to_html(full_html=False, include_plotlyjs="cdn")

    # Styled table
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
            "Total(pts)": "{:+,.1f}",
            "Avg/Entry(pts)": "{:+,.2f}",
            "Win%": "{:.1f}%",
            "PF": "{:.3f}",
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

    best = tbl_sorted.iloc[0] if len(tbl_sorted) > 0 else None
    best_label = f"{best['Total(pts)']:+,.1f} pts — {best['Session']} ({best['TF']})" if best is not None else "N/A"

    html = f"""<!DOCTYPE html>
<html><head><title>ES 3-Lot Scaled Exit</title>
<style>
  body {{ background: #0d1117; color: #e0e0e0; font-family: monospace; padding: 20px; }}
  h1 {{ color: #58a6ff; font-size: 20px; }}
  h2 {{ color: #8b949e; font-size: 16px; margin-top: 5px; }}
  .summary {{ display: flex; gap: 20px; margin: 15px 0; flex-wrap: wrap; }}
  .card {{ background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 12px 16px; }}
  .card .num {{ font-size: 24px; font-weight: bold; }}
  .green {{ color: #3fb950; }} .red {{ color: #f85149; }} .yellow {{ color: #d29922; }}
</style></head><body>
<h1>Strategy 3: 3-Lot Scaled Exit (1/2/3 ATR) [ALL P&L IN POINTS]</h1>
<h2>Entry: HA close crosses EMA band (no RSI) | Stop: 1 ATR | TP: 1/2/3 ATR scaled</h2>
<div class="summary">
  <div class="card"><div class="green num">{len(tbl_sorted[tbl_sorted['PF'] >= 1.0])}</div><div>Profitable (PF &ge; 1.0)</div></div>
  <div class="card"><div class="red num">{len(tbl_sorted[tbl_sorted['PF'] < 1.0])}</div><div>Losing (PF &lt; 1.0)</div></div>
  <div class="card"><div class="green num">{best_label}</div><div>Best combo</div></div>
</div>
{chart_html}
<h2 style="margin-top:30px;">Results Table</h2>
{styled.to_html()}
</body></html>"""

    output = "backtest_3lot_chart.html"
    with open(output, "w") as f:
        f.write(html)

    import webbrowser, os
    webbrowser.open("file://" + os.path.abspath(output))
    print(f"\nChart saved to {output} and opened in browser.")


if __name__ == "__main__":
    main()
