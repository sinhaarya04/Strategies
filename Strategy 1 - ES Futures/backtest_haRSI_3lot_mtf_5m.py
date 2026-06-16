"""
Strategy: HA EMA + RSI 3-Lot Scaled Exit — MTF (60m bias + 5m entries)

Combines:
  - The HA EMA + RSI signal from backtest_mtf_5m.py (proven +$12k/year strategy)
  - The 3-lot scaled exit ladder from backtest_fisher_3lot.py (Gary spec)
  - Multi-timeframe gate: 60m bias filters out counter-trend entries

Signal (5m AND 60m use the same rules):
  LONG  bias/entry: HA close > 20-EMA(High) AND RSI > 55
  SHORT bias/entry: HA close < 20-EMA(Low)  AND RSI < 45
  Neutral: skip

Bias source: 60m.
Entry trigger: 5m signal that ALIGNS with 60m bias.

Position: 3 contracts on the entry signal.
Stop:     1.5 × ATR(5m)
Targets:  1×/2×/3× ATR(5m) scaled
Ladder:   TP1 hit -> stop to BE on lots 2 & 3
          TP2 hit -> trail stop on lot 3 to +1 ATR
          TP3 hit -> lot 3 closes (or trail-stopped first)

All P&L in points and dollars ($50/pt). $50k account base.
"""

import numpy as np
import pandas as pd

from data_feed import fetch_es_data
from indicators import compute_all
from config import RSI_BUY_THRESHOLD, RSI_SELL_THRESHOLD

ACCOUNT_SIZE = 50_000.0
POINT_VALUE = 50.0

SESSIONS = [
    ("All Day", 0, 0, 23, 59),
    ("Overnight (00:00-04:00)", 0, 0, 4, 0),
    ("US Morning (04:00-13:00)", 4, 0, 13, 0),
    ("US Afternoon (13:00-17:00)", 13, 0, 17, 0),
]


def resample(df_5m, freq):
    return df_5m.resample(freq).agg({
        "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum",
    }).dropna()


def get_60m_bias(df_60m):
    """
    +1 long bias  : HA close > EMA_High AND RSI > 55
    -1 short bias : HA close < EMA_Low  AND RSI < 45
     0 neutral
    """
    bias = pd.Series(0, index=df_60m.index, dtype=int)
    ha = df_60m["HA_Close"].values
    eh = df_60m["EMA_High"].values
    el = df_60m["EMA_Low"].values
    rs = df_60m["RSI"].values
    for i in range(len(df_60m)):
        if np.isnan(ha[i]) or np.isnan(eh[i]) or np.isnan(rs[i]):
            continue
        if ha[i] > eh[i] and rs[i] > RSI_BUY_THRESHOLD:
            bias.iloc[i] = 1
        elif ha[i] < el[i] and rs[i] < RSI_SELL_THRESHOLD:
            bias.iloc[i] = -1
    return bias


def align_bias_to_5m(bias_60m, df_5m):
    bias = bias_60m.reindex(df_5m.index, method="ffill").fillna(0).astype(int)
    return bias.values


def simulate(df_5m, bias_arr, s_start_h, s_start_m, s_end_h, s_end_m):
    """3-lot scaled exit, 5m HA+RSI entries gated by 60m bias."""
    start_min = s_start_h * 60 + s_start_m
    end_min   = s_end_h * 60 + s_end_m
    filter_all = (start_min == 0 and end_min == 23 * 60 + 59)

    direction = 0
    entry_price = 0.0
    atr_at_entry = 0.0
    lots_remaining = 0
    stop_price = 0.0
    tp1 = tp2 = tp3 = 0.0
    lot1_hit = lot2_hit = False
    entry_time = None

    equity_pts = 0.0
    equities = [0.0]
    trade_log = []
    entry_groups = []

    ha_c = df_5m["HA_Close"].values
    ema_h = df_5m["EMA_High"].values
    ema_l = df_5m["EMA_Low"].values
    rsi  = df_5m["RSI"].values
    atr_arr   = df_5m["ATR"].values
    close_arr = df_5m["Close"].values
    high_arr  = df_5m["High"].values
    low_arr   = df_5m["Low"].values
    idx       = df_5m.index

    for i in range(1, len(df_5m)):
        atr   = atr_arr[i]
        close = close_arr[i]
        high  = high_arr[i]
        low   = low_arr[i]

        if np.isnan(ha_c[i]) or np.isnan(ema_h[i]) or np.isnan(rsi[i]) or np.isnan(atr) or atr == 0:
            equities.append(equity_pts)
            continue

        bar_min = idx[i].hour * 60 + idx[i].minute
        in_session = filter_all or (start_min <= bar_min < end_min)

        # ---- EXIT LOGIC ----
        if direction != 0 and lots_remaining > 0:
            if direction == 1:
                if low <= stop_price:
                    pts_per_lot = stop_price - entry_price
                    total_pts   = pts_per_lot * lots_remaining
                    equity_pts += total_pts
                    exit_type   = "STOP" if not lot1_hit else ("BE_STOP" if not lot2_hit else "TRAIL_STOP")
                    trade_log.append({"entry_time": entry_time, "exit_time": idx[i],
                                      "direction": "LONG", "entry": entry_price, "exit": stop_price,
                                      "lots_closed": lots_remaining, "pts_per_lot": pts_per_lot,
                                      "total_pts": total_pts, "exit_type": exit_type})
                    direction = 0; lots_remaining = 0
                else:
                    if lots_remaining >= 3 and not lot1_hit and high >= tp1:
                        pts = tp1 - entry_price
                        equity_pts += pts
                        trade_log.append({"entry_time": entry_time, "exit_time": idx[i],
                                          "direction": "LONG", "entry": entry_price, "exit": tp1,
                                          "lots_closed": 1, "pts_per_lot": pts, "total_pts": pts,
                                          "exit_type": "TP1"})
                        lot1_hit = True; lots_remaining = 2; stop_price = entry_price
                    if lots_remaining >= 2 and lot1_hit and not lot2_hit and high >= tp2:
                        pts = tp2 - entry_price
                        equity_pts += pts
                        trade_log.append({"entry_time": entry_time, "exit_time": idx[i],
                                          "direction": "LONG", "entry": entry_price, "exit": tp2,
                                          "lots_closed": 1, "pts_per_lot": pts, "total_pts": pts,
                                          "exit_type": "TP2"})
                        lot2_hit = True; lots_remaining = 1; stop_price = tp1
                    if lots_remaining >= 1 and lot2_hit and high >= tp3:
                        pts = tp3 - entry_price
                        equity_pts += pts
                        trade_log.append({"entry_time": entry_time, "exit_time": idx[i],
                                          "direction": "LONG", "entry": entry_price, "exit": tp3,
                                          "lots_closed": 1, "pts_per_lot": pts, "total_pts": pts,
                                          "exit_type": "TP3"})
                        lots_remaining = 0; direction = 0
            else:
                if high >= stop_price:
                    pts_per_lot = entry_price - stop_price
                    total_pts   = pts_per_lot * lots_remaining
                    equity_pts += total_pts
                    exit_type   = "STOP" if not lot1_hit else ("BE_STOP" if not lot2_hit else "TRAIL_STOP")
                    trade_log.append({"entry_time": entry_time, "exit_time": idx[i],
                                      "direction": "SHORT", "entry": entry_price, "exit": stop_price,
                                      "lots_closed": lots_remaining, "pts_per_lot": pts_per_lot,
                                      "total_pts": total_pts, "exit_type": exit_type})
                    direction = 0; lots_remaining = 0
                else:
                    if lots_remaining >= 3 and not lot1_hit and low <= tp1:
                        pts = entry_price - tp1
                        equity_pts += pts
                        trade_log.append({"entry_time": entry_time, "exit_time": idx[i],
                                          "direction": "SHORT", "entry": entry_price, "exit": tp1,
                                          "lots_closed": 1, "pts_per_lot": pts, "total_pts": pts,
                                          "exit_type": "TP1"})
                        lot1_hit = True; lots_remaining = 2; stop_price = entry_price
                    if lots_remaining >= 2 and lot1_hit and not lot2_hit and low <= tp2:
                        pts = entry_price - tp2
                        equity_pts += pts
                        trade_log.append({"entry_time": entry_time, "exit_time": idx[i],
                                          "direction": "SHORT", "entry": entry_price, "exit": tp2,
                                          "lots_closed": 1, "pts_per_lot": pts, "total_pts": pts,
                                          "exit_type": "TP2"})
                        lot2_hit = True; lots_remaining = 1; stop_price = tp1
                    if lots_remaining >= 1 and lot2_hit and low <= tp3:
                        pts = entry_price - tp3
                        equity_pts += pts
                        trade_log.append({"entry_time": entry_time, "exit_time": idx[i],
                                          "direction": "SHORT", "entry": entry_price, "exit": tp3,
                                          "lots_closed": 1, "pts_per_lot": pts, "total_pts": pts,
                                          "exit_type": "TP3"})
                        lots_remaining = 0; direction = 0

        # ---- ENTRY: 5m HA+RSI signal aligned with 60m bias ----
        if direction == 0 and in_session:
            bias = bias_arr[i]
            go_long  = (bias == 1)  and (ha_c[i] > ema_h[i]) and (rsi[i] > RSI_BUY_THRESHOLD)
            go_short = (bias == -1) and (ha_c[i] < ema_l[i]) and (rsi[i] < RSI_SELL_THRESHOLD)

            if go_long:
                direction = 1
                entry_price = close; atr_at_entry = atr
                lots_remaining = 3; lot1_hit = lot2_hit = False
                stop_price = close - 1.5 * atr
                tp1, tp2, tp3 = close + atr, close + 2 * atr, close + 3 * atr
                entry_time = idx[i]
                entry_groups.append({"entry_time": entry_time, "direction": "LONG",
                                     "entry_price": entry_price, "atr_at_entry": atr_at_entry,
                                     "stop_dist_pts": 1.5 * atr_at_entry})
            elif go_short:
                direction = -1
                entry_price = close; atr_at_entry = atr
                lots_remaining = 3; lot1_hit = lot2_hit = False
                stop_price = close + 1.5 * atr
                tp1, tp2, tp3 = close - atr, close - 2 * atr, close - 3 * atr
                entry_time = idx[i]
                entry_groups.append({"entry_time": entry_time, "direction": "SHORT",
                                     "entry_price": entry_price, "atr_at_entry": atr_at_entry,
                                     "stop_dist_pts": 1.5 * atr_at_entry})

        if direction == 1 and lots_remaining > 0:
            equities.append(equity_pts + (close - entry_price) * lots_remaining)
        elif direction == -1 and lots_remaining > 0:
            equities.append(equity_pts + (entry_price - close) * lots_remaining)
        else:
            equities.append(equity_pts)

    return equities, trade_log, entry_groups


def summarize(trade_log, entry_groups):
    if not trade_log:
        return None
    all_pts = [t["total_pts"] for t in trade_log]
    wins   = [p for p in all_pts if p > 0]
    losses = [p for p in all_pts if p <= 0]
    gross_w = sum(wins) if wins else 0
    gross_l = abs(sum(losses)) if losses else 0.001
    cnt = lambda kind: sum(1 for t in trade_log if t["exit_type"] == kind)

    entry_pnl = {}
    for t in trade_log:
        entry_pnl[t["entry_time"]] = entry_pnl.get(t["entry_time"], 0.0) + t["total_pts"]
    per_entry_pts = list(entry_pnl.values())
    worst_entry = min(per_entry_pts) if per_entry_pts else 0.0
    best_entry  = max(per_entry_pts) if per_entry_pts else 0.0

    atrs  = [e["atr_at_entry"] for e in entry_groups]
    stops = [e["stop_dist_pts"] for e in entry_groups]
    avg_atr      = sum(atrs) / len(atrs) if atrs else 0.0
    avg_stop_pts = sum(stops) / len(stops) if stops else 0.0
    max_stop_pts = max(stops) if stops else 0.0

    total_pts = sum(all_pts)
    total_usd = total_pts * POINT_VALUE
    return_pct = total_usd / ACCOUNT_SIZE * 100.0
    final_acct = ACCOUNT_SIZE + total_usd

    return {
        "entries": len(entry_groups), "exit_events": len(trade_log),
        "total_pts": total_pts,
        "avg_per_entry": total_pts / len(entry_groups) if entry_groups else 0,
        "wr": len(wins) / len(all_pts) * 100 if all_pts else 0,
        "pf": gross_w / gross_l,
        "tp1": cnt("TP1"), "tp2": cnt("TP2"), "tp3": cnt("TP3"),
        "stop": cnt("STOP"), "be_stop": cnt("BE_STOP"), "trail_stop": cnt("TRAIL_STOP"),
        "avg_atr": avg_atr, "avg_stop_pts": avg_stop_pts, "max_stop_pts": max_stop_pts,
        "best_entry_pts": best_entry, "worst_entry_pts": worst_entry,
        "total_usd": total_usd, "return_pct": return_pct, "final_acct": final_acct,
    }


def main():
    print("Fetching 5m data...")
    df_5m_raw = fetch_es_data()

    print("Computing 5m indicators (entry timeframe)...")
    df_5m = compute_all(df_5m_raw)

    print("Computing 60m indicators (bias)...")
    df_60m = compute_all(resample(df_5m_raw, "60min"))

    print(f"5m bars: {len(df_5m)} | 60m bars: {len(df_60m)}")
    print(f"RSI thresholds: Buy > {RSI_BUY_THRESHOLD}, Sell < {RSI_SELL_THRESHOLD}")

    bias_60m = get_60m_bias(df_60m)
    bias_arr = align_bias_to_5m(bias_60m, df_5m)
    buy_bars  = int((bias_arr == 1).sum())
    sell_bars = int((bias_arr == -1).sum())
    neutral_bars = int((bias_arr == 0).sum())
    print(f"60m bias distribution on 5m bars: BUY={buy_bars} SELL={sell_bars} NEUTRAL={neutral_bars}")
    print(f"{'=' * 100}")

    all_results = []
    for sess_name, sh, sm, eh, em in SESSIONS:
        eq, log, entry_groups = simulate(df_5m, bias_arr, sh, sm, eh, em)
        stats = summarize(log, entry_groups)
        if stats is None:
            print(f"  {sess_name:30s}  NO TRADES")
            continue
        print(f"  {sess_name:30s}  "
              f"E={stats['entries']:5d}  "
              f"P&L={stats['total_pts']:+9.2f}pts  "
              f"${stats['total_usd']:+10,.0f}  "
              f"Ret={stats['return_pct']:+6.1f}%  "
              f"WR={stats['wr']:5.1f}%  PF={stats['pf']:5.3f}  "
              f"AvgStop={stats['avg_stop_pts']:5.2f}pts(${stats['avg_stop_pts']*POINT_VALUE:,.0f})  "
              f"Worst={stats['worst_entry_pts']:+7.2f}pts")
        all_results.append({
            "Session": sess_name,
            "Entries": stats["entries"], "Exits": stats["exit_events"],
            "Total(pts)": stats["total_pts"],
            "Total($)": stats["total_usd"],
            "Return(%)": stats["return_pct"],
            "FinalAcct($)": stats["final_acct"],
            "Avg/Entry(pts)": stats["avg_per_entry"],
            "Win%": stats["wr"], "PF": stats["pf"],
            "AvgATR(pts)": stats["avg_atr"],
            "AvgStop(pts)": stats["avg_stop_pts"],
            "AvgStop($/lot)": stats["avg_stop_pts"] * POINT_VALUE,
            "MaxStop(pts)": stats["max_stop_pts"],
            "BestEntry(pts)": stats["best_entry_pts"],
            "WorstEntry(pts)": stats["worst_entry_pts"],
            "WorstEntry($)": stats["worst_entry_pts"] * POINT_VALUE,
            "TP1": stats["tp1"], "TP2": stats["tp2"], "TP3": stats["tp3"],
            "Stop": stats["stop"], "BE_Stop": stats["be_stop"], "Trail_Stop": stats["trail_stop"],
            "equities": eq,
        })

    tbl = pd.DataFrame([{k: v for k, v in r.items() if k != "equities"} for r in all_results])
    print(f"\n{'=' * 140}")
    print("  HA+RSI 3-LOT MTF (60m bias + 5m entries) — ALL RUNS (sorted by PF)")
    print(f"{'=' * 140}")
    print(tbl.sort_values("PF", ascending=False).to_string(index=False, float_format="%.2f"))

    build_html(all_results, tbl)


def build_html(all_results, tbl):
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=len(all_results), cols=1,
            subplot_titles=[r["Session"] for r in all_results],
            vertical_spacing=0.05,
        )
        colors = ["#3fb950", "#58a6ff", "#d29922", "#f85149"]
        for row_idx, r in enumerate(all_results, 1):
            fig.add_trace(go.Scatter(
                y=r["equities"], mode="lines",
                name=r["Session"],
                line=dict(color=colors[(row_idx - 1) % len(colors)], width=1.5),
                showlegend=True,
            ), row=row_idx, col=1)
            fig.update_yaxes(title_text="Points", row=row_idx, col=1)

        fig.update_layout(template="plotly_dark", height=300 * len(all_results),
                          title="HA+RSI 3-Lot MTF (60m bias + 5m entries) — Equity Curves")
        chart_html = fig.to_html(full_html=False, include_plotlyjs="cdn")
    except ImportError:
        chart_html = "<p style='color:#d29922'>(plotly not installed — equity curves omitted; table only)</p>"

    tbl_sorted = tbl.sort_values("PF", ascending=False)

    def color_row(row):
        if row["PF"] >= 1.05: return ["background-color: #1a4d1a"] * len(row)
        elif row["PF"] >= 1.0: return ["background-color: #2d3d1a"] * len(row)
        elif row["PF"] >= 0.95: return ["background-color: #3d3d1a"] * len(row)
        else: return ["background-color: #4d1a1a"] * len(row)

    styled = (tbl_sorted.style
        .apply(color_row, axis=1)
        .format({
            "Total(pts)": "{:+,.1f}", "Total($)": "${:+,.0f}",
            "Return(%)": "{:+.1f}%", "FinalAcct($)": "${:,.0f}",
            "Avg/Entry(pts)": "{:+,.2f}", "Win%": "{:.1f}%", "PF": "{:.3f}",
            "AvgATR(pts)": "{:.2f}", "AvgStop(pts)": "{:.2f}",
            "AvgStop($/lot)": "${:,.0f}", "MaxStop(pts)": "{:.2f}",
            "BestEntry(pts)": "{:+.2f}", "WorstEntry(pts)": "{:+.2f}",
            "WorstEntry($)": "${:+,.0f}",
        })
        .set_properties(**{
            "font-family": "monospace", "font-size": "12px",
            "padding": "4px 8px", "border": "1px solid #333", "color": "#e0e0e0",
        })
        .set_table_styles([
            {"selector": "th", "props": [
                ("background-color", "#1a1a2e"), ("color", "#e0e0e0"),
                ("font-size", "12px"), ("padding", "6px 8px"),
                ("border", "1px solid #333"), ("position", "sticky"), ("top", "0"),
            ]},
            {"selector": "table", "props": [("border-collapse", "collapse"), ("width", "100%")]},
        ])
    )

    best = tbl_sorted.iloc[0] if len(tbl_sorted) > 0 else None
    best_label = (f"{best['Total(pts)']:+,.1f} pts (${best['Total($)']:+,.0f}, {best['Return(%)']:+.1f}%) — {best['Session']}"
                  if best is not None else "N/A")

    html = f"""<!DOCTYPE html>
<html><head><title>ES HA+RSI 3-Lot MTF (60m+5m)</title>
<style>
  body {{ background: #0d1117; color: #e0e0e0; font-family: monospace; padding: 20px; }}
  h1 {{ color: #58a6ff; font-size: 20px; }}
  h2 {{ color: #8b949e; font-size: 16px; margin-top: 5px; }}
  .summary {{ display: flex; gap: 20px; margin: 15px 0; flex-wrap: wrap; }}
  .card {{ background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 12px 16px; }}
  .card .num {{ font-size: 22px; font-weight: bold; }}
  .green {{ color: #3fb950; }} .red {{ color: #f85149; }} .yellow {{ color: #d29922; }}
</style></head><body>
<h1>HA EMA + RSI 3-Lot Scaled Exit — MTF (60m bias + 5m entries)</h1>
<h2>Bias: 60m HA cross EMA + RSI 55/45 | Entry: 5m same signal aligned with 60m bias | Stop: 1.5×ATR | TP: 1/2/3×ATR | Ladder: BE → +1ATR</h2>
<h2>4 sessions = {len(tbl_sorted)} runs | Account base = ${ACCOUNT_SIZE:,.0f} | ES point = ${POINT_VALUE}</h2>
<div class="summary">
  <div class="card"><div class="green num">{len(tbl_sorted[tbl_sorted['PF'] >= 1.0])}</div><div>Profitable (PF ≥ 1.0)</div></div>
  <div class="card"><div class="red num">{len(tbl_sorted[tbl_sorted['PF'] < 1.0])}</div><div>Losing</div></div>
  <div class="card"><div class="green num">{best_label}</div><div>Best run</div></div>
</div>
{chart_html}
<h2 style="margin-top:30px;">Results Table</h2>
{styled.to_html()}
</body></html>"""

    output = "backtest_haRSI_3lot_mtf_5m_chart.html"
    with open(output, "w") as f:
        f.write(html)

    import webbrowser, os
    webbrowser.open("file://" + os.path.abspath(output))
    print(f"\nChart saved to {output} and opened in browser.")


if __name__ == "__main__":
    main()
