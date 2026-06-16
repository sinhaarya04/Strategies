"""
Strategy: Fisher Transform 3-Lot Scaled Exit (Gary spec)

Entry signal: Fisher Transform crosses its 20-period SMA (signal line).
Position: 3 contracts on the cross.
Initial stop: 1.5 x ATR  (total initial risk = 3 x 1.5 ATR)
Targets (scaled, one lot each):
  TP1 = entry +/- 1 x ATR
  TP2 = entry +/- 2 x ATR
  TP3 = entry +/- 3 x ATR
Stop ladder:
  TP1 hit -> move stop on lots 2 & 3 to breakeven (entry)
  TP2 hit -> move stop on lot 3 to TP1 level (+1 ATR trail)
  TP3 hit -> lot 3 closes

Four variants:
  V1: FT(Close),    no RSI filter
  V2: FT(HA Close), no RSI filter
  V3: FT(Close),    RSI > 50 long / RSI < 50 short
  V4: FT(HA Close), RSI > 50 long / RSI < 50 short

All P&L in POINTS.
"""

import numpy as np
import pandas as pd

from data_feed import fetch_es_data
from indicators import compute_all

FT_SMA_PERIOD = 20
ACCOUNT_SIZE = 50_000.0
POINT_VALUE = 50.0  # ES = $50 per point

SESSIONS = [
    ("All Day", 0, 0, 23, 59),
    ("Overnight (00:00-04:00)", 0, 0, 4, 0),
    ("US Morning (04:00-13:00)", 4, 0, 13, 0),
    ("US Afternoon (13:00-17:00)", 13, 0, 17, 0),
]

TIMEFRAMES = ["5min", "15min", "60min"]

VARIANTS = [
    ("V1 FT(Close) no-RSI",  "Fisher",    False),
    ("V2 FT(HA)    no-RSI",  "Fisher_HA", False),
    ("V3 FT(Close) RSI><50", "Fisher",    True),
    ("V4 FT(HA)    RSI><50", "Fisher_HA", True),
]


def resample(df_5m, freq):
    return df_5m.resample(freq).agg({
        "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum",
    }).dropna()


def simulate(df, ft_col, sma_col, use_rsi_filter,
             s_start_h, s_start_m, s_end_h, s_end_m):
    """3-lot scaled exit with FT-vs-SMA20 cross entry. P&L in points."""
    start_min = s_start_h * 60 + s_start_m
    end_min   = s_end_h * 60 + s_end_m
    filter_all = (start_min == 0 and end_min == 23 * 60 + 59)

    # state
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
    entry_groups = []  # list of dicts: {entry_time, atr_at_entry, stop_dist_pts (per lot)}

    ft   = df[ft_col].values
    sma  = df[sma_col].values
    rsi  = df["RSI"].values
    atr_arr   = df["ATR"].values
    close_arr = df["Close"].values
    high_arr  = df["High"].values
    low_arr   = df["Low"].values
    idx       = df.index

    for i in range(1, len(df)):
        atr   = atr_arr[i]
        close = close_arr[i]
        high  = high_arr[i]
        low   = low_arr[i]

        if np.isnan(ft[i]) or np.isnan(sma[i]) or np.isnan(atr) or atr == 0:
            equities.append(equity_pts)
            continue

        bar_min = idx[i].hour * 60 + idx[i].minute
        in_session = filter_all or (start_min <= bar_min < end_min)

        # ---- EXIT LOGIC (always active) ----
        if direction != 0 and lots_remaining > 0:
            if direction == 1:  # LONG
                if low <= stop_price:
                    pts_per_lot = stop_price - entry_price
                    total_pts   = pts_per_lot * lots_remaining
                    equity_pts += total_pts
                    exit_type   = "STOP" if not lot1_hit else ("BE_STOP" if not lot2_hit else "TRAIL_STOP")
                    trade_log.append({
                        "entry_time": entry_time, "exit_time": idx[i],
                        "direction": "LONG", "entry": entry_price, "exit": stop_price,
                        "lots_closed": lots_remaining, "pts_per_lot": pts_per_lot,
                        "total_pts": total_pts, "exit_type": exit_type,
                        "atr_at_entry": atr_at_entry,
                    })
                    direction = 0; lots_remaining = 0
                else:
                    if lots_remaining >= 3 and not lot1_hit and high >= tp1:
                        pts = tp1 - entry_price
                        equity_pts += pts
                        trade_log.append({
                            "entry_time": entry_time, "exit_time": idx[i],
                            "direction": "LONG", "entry": entry_price, "exit": tp1,
                            "lots_closed": 1, "pts_per_lot": pts, "total_pts": pts,
                            "exit_type": "TP1",
                        })
                        lot1_hit = True
                        lots_remaining = 2
                        stop_price = entry_price  # -> breakeven

                    if lots_remaining >= 2 and lot1_hit and not lot2_hit and high >= tp2:
                        pts = tp2 - entry_price
                        equity_pts += pts
                        trade_log.append({
                            "entry_time": entry_time, "exit_time": idx[i],
                            "direction": "LONG", "entry": entry_price, "exit": tp2,
                            "lots_closed": 1, "pts_per_lot": pts, "total_pts": pts,
                            "exit_type": "TP2",
                        })
                        lot2_hit = True
                        lots_remaining = 1
                        stop_price = tp1  # trail to +1 ATR

                    if lots_remaining >= 1 and lot2_hit and high >= tp3:
                        pts = tp3 - entry_price
                        equity_pts += pts
                        trade_log.append({
                            "entry_time": entry_time, "exit_time": idx[i],
                            "direction": "LONG", "entry": entry_price, "exit": tp3,
                            "lots_closed": 1, "pts_per_lot": pts, "total_pts": pts,
                            "exit_type": "TP3",
                        })
                        lots_remaining = 0
                        direction = 0

            else:  # SHORT
                if high >= stop_price:
                    pts_per_lot = entry_price - stop_price
                    total_pts   = pts_per_lot * lots_remaining
                    equity_pts += total_pts
                    exit_type   = "STOP" if not lot1_hit else ("BE_STOP" if not lot2_hit else "TRAIL_STOP")
                    trade_log.append({
                        "entry_time": entry_time, "exit_time": idx[i],
                        "direction": "SHORT", "entry": entry_price, "exit": stop_price,
                        "lots_closed": lots_remaining, "pts_per_lot": pts_per_lot,
                        "total_pts": total_pts, "exit_type": exit_type,
                    })
                    direction = 0; lots_remaining = 0
                else:
                    if lots_remaining >= 3 and not lot1_hit and low <= tp1:
                        pts = entry_price - tp1
                        equity_pts += pts
                        trade_log.append({
                            "entry_time": entry_time, "exit_time": idx[i],
                            "direction": "SHORT", "entry": entry_price, "exit": tp1,
                            "lots_closed": 1, "pts_per_lot": pts, "total_pts": pts,
                            "exit_type": "TP1",
                        })
                        lot1_hit = True
                        lots_remaining = 2
                        stop_price = entry_price  # -> breakeven

                    if lots_remaining >= 2 and lot1_hit and not lot2_hit and low <= tp2:
                        pts = entry_price - tp2
                        equity_pts += pts
                        trade_log.append({
                            "entry_time": entry_time, "exit_time": idx[i],
                            "direction": "SHORT", "entry": entry_price, "exit": tp2,
                            "lots_closed": 1, "pts_per_lot": pts, "total_pts": pts,
                            "exit_type": "TP2",
                        })
                        lot2_hit = True
                        lots_remaining = 1
                        stop_price = tp1  # trail to +1 ATR

                    if lots_remaining >= 1 and lot2_hit and low <= tp3:
                        pts = entry_price - tp3
                        equity_pts += pts
                        trade_log.append({
                            "entry_time": entry_time, "exit_time": idx[i],
                            "direction": "SHORT", "entry": entry_price, "exit": tp3,
                            "lots_closed": 1, "pts_per_lot": pts, "total_pts": pts,
                            "exit_type": "TP3",
                        })
                        lots_remaining = 0
                        direction = 0

        # ---- ENTRY: FT crosses its 20-SMA ----
        if direction == 0 and in_session:
            ft_prev, ft_now = ft[i - 1], ft[i]
            sma_prev, sma_now = sma[i - 1], sma[i]

            if not (np.isnan(ft_prev) or np.isnan(sma_prev)):
                cross_up = (ft_prev <= sma_prev) and (ft_now > sma_now)
                cross_dn = (ft_prev >= sma_prev) and (ft_now < sma_now)

                go_long = cross_up
                go_short = cross_dn

                if use_rsi_filter:
                    r = rsi[i]
                    if np.isnan(r):
                        go_long = go_short = False
                    else:
                        go_long  = go_long  and (r > 50)
                        go_short = go_short and (r < 50)

                if go_long:
                    direction = 1
                    entry_price = close
                    atr_at_entry = atr
                    lots_remaining = 3
                    lot1_hit = lot2_hit = False
                    stop_price = close - 1.5 * atr
                    tp1 = close + 1 * atr
                    tp2 = close + 2 * atr
                    tp3 = close + 3 * atr
                    entry_time = idx[i]
                    entry_groups.append({
                        "entry_time": entry_time, "direction": "LONG",
                        "entry_price": entry_price, "atr_at_entry": atr_at_entry,
                        "stop_dist_pts": 1.5 * atr_at_entry,
                    })
                elif go_short:
                    direction = -1
                    entry_price = close
                    atr_at_entry = atr
                    lots_remaining = 3
                    lot1_hit = lot2_hit = False
                    stop_price = close + 1.5 * atr
                    tp1 = close - 1 * atr
                    tp2 = close - 2 * atr
                    tp3 = close - 3 * atr
                    entry_time = idx[i]
                    entry_groups.append({
                        "entry_time": entry_time, "direction": "SHORT",
                        "entry_price": entry_price, "atr_at_entry": atr_at_entry,
                        "stop_dist_pts": 1.5 * atr_at_entry,
                    })

        # mark-to-market
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

    # Per-entry P&L (sum of all lot exits for each entry_time)
    entry_pnl = {}
    for t in trade_log:
        entry_pnl[t["entry_time"]] = entry_pnl.get(t["entry_time"], 0.0) + t["total_pts"]
    per_entry_pts = list(entry_pnl.values())
    worst_entry = min(per_entry_pts) if per_entry_pts else 0.0
    best_entry = max(per_entry_pts) if per_entry_pts else 0.0

    # ATR / stop stats from entry_groups
    atrs = [e["atr_at_entry"] for e in entry_groups]
    stops = [e["stop_dist_pts"] for e in entry_groups]
    avg_atr = sum(atrs) / len(atrs) if atrs else 0.0
    avg_stop_pts = sum(stops) / len(stops) if stops else 0.0
    max_stop_pts = max(stops) if stops else 0.0

    total_pts = sum(all_pts)
    total_usd = total_pts * POINT_VALUE
    return_pct = total_usd / ACCOUNT_SIZE * 100.0
    final_acct = ACCOUNT_SIZE + total_usd

    return {
        "entries": len(entry_groups),
        "exit_events": len(trade_log),
        "total_pts": total_pts,
        "avg_per_entry": total_pts / len(entry_groups) if entry_groups else 0,
        "wr": len(wins) / len(all_pts) * 100 if all_pts else 0,
        "pf": gross_w / gross_l,
        "tp1": cnt("TP1"), "tp2": cnt("TP2"), "tp3": cnt("TP3"),
        "stop": cnt("STOP"), "be_stop": cnt("BE_STOP"), "trail_stop": cnt("TRAIL_STOP"),
        # new fields
        "avg_atr": avg_atr,
        "avg_stop_pts": avg_stop_pts,         # avg stop distance per lot, in points
        "max_stop_pts": max_stop_pts,         # worst-case (highest ATR) stop distance per lot
        "best_entry_pts": best_entry,         # best single 3-lot package
        "worst_entry_pts": worst_entry,       # worst single 3-lot package
        "total_usd": total_usd,               # total P&L in $ (assumes $50/pt)
        "return_pct": return_pct,             # return on $50k account
        "final_acct": final_acct,             # final account value
    }


def main():
    print("Fetching 5m data...")
    df_5m = fetch_es_data()

    all_results = []

    for tf in TIMEFRAMES:
        df_r = resample(df_5m, tf)
        df = compute_all(df_r)
        # 20-SMA of Fisher Transform (Gary spec — replaces default 1-bar-lag signal)
        df["Fisher_SMA20"]    = df["Fisher"].rolling(window=FT_SMA_PERIOD, min_periods=FT_SMA_PERIOD).mean()
        df["Fisher_HA_SMA20"] = df["Fisher_HA"].rolling(window=FT_SMA_PERIOD, min_periods=FT_SMA_PERIOD).mean()

        print(f"\n{'=' * 100}")
        print(f"  Timeframe: {tf} | {len(df)} bars")
        print(f"{'=' * 100}")

        for var_name, ft_col, use_rsi in VARIANTS:
            sma_col = f"{ft_col}_SMA20"
            for sess_name, sh, sm, eh, em in SESSIONS:
                eq, log, entry_groups = simulate(df, ft_col, sma_col, use_rsi, sh, sm, eh, em)
                stats = summarize(log, entry_groups)
                if stats is None:
                    print(f"  {var_name:22s}  {sess_name:30s}  NO TRADES")
                    continue
                print(f"  {var_name:22s}  {sess_name:30s}  "
                      f"E={stats['entries']:4d}  "
                      f"P&L={stats['total_pts']:+8.2f}pts  "
                      f"${stats['total_usd']:+9,.0f}  "
                      f"Ret={stats['return_pct']:+6.1f}%  "
                      f"WR={stats['wr']:5.1f}%  PF={stats['pf']:5.3f}  "
                      f"AvgStop={stats['avg_stop_pts']:5.2f}pts(${stats['avg_stop_pts']*POINT_VALUE:,.0f})  "
                      f"Worst={stats['worst_entry_pts']:+7.2f}pts")
                all_results.append({
                    "TF": tf, "Variant": var_name, "Session": sess_name,
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
    print(f"\n{'=' * 130}")
    print("  FISHER 3-LOT SCALED EXIT — ALL RUNS (sorted by PF) [POINTS]")
    print(f"{'=' * 130}")
    print(tbl.sort_values("PF", ascending=False).to_string(index=False, float_format="%.2f"))

    build_html(all_results, tbl)

    # Also emit one HTML per timeframe (so dashboard can show standalone 5m / 15m / 60m separately)
    for tf in TIMEFRAMES:
        tf_results = [r for r in all_results if r["TF"] == tf]
        tf_tbl = tbl[tbl["TF"] == tf].copy()
        if len(tf_tbl) > 0:
            build_html_for_tf(tf, tf_results, tf_tbl)


def build_html_for_tf(tf, all_results, tbl):
    """Per-timeframe HTML output. Same styling as build_html but scoped to one TF."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        groups = []
        seen = set()
        for r in all_results:
            key = r["Variant"]
            if key not in seen:
                seen.add(key); groups.append(key)

        fig = make_subplots(
            rows=len(groups), cols=1,
            subplot_titles=[v for v in groups],
            vertical_spacing=0.04,
        )
        colors = ["#58a6ff", "#3fb950", "#d29922", "#f85149"]
        for row_idx, var in enumerate(groups, 1):
            subset = [r for r in all_results if r["Variant"] == var]
            for j, r in enumerate(subset):
                fig.add_trace(go.Scatter(
                    y=r["equities"], mode="lines",
                    name=f"{r['Session']}",
                    line=dict(color=colors[j % len(colors)], width=1.3),
                    legendgroup=var,
                    showlegend=True,
                ), row=row_idx, col=1)
            fig.update_yaxes(title_text="Points", row=row_idx, col=1)

        fig.update_layout(
            template="plotly_dark",
            height=320 * len(groups),
            title=f"Fisher 3-Lot Standalone — {tf} bars",
        )
        chart_html = fig.to_html(full_html=False, include_plotlyjs="cdn")
    except ImportError:
        chart_html = "<p style='color:#d29922'>(plotly not installed — equity curves omitted; table only)</p>"

    # Drop the TF column from per-TF tables — redundant
    tbl_sorted = tbl.drop(columns=["TF"]).sort_values("PF", ascending=False)

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
    best_label = (f"{best['Total(pts)']:+,.1f} pts (${best['Total($)']:+,.0f}, {best['Return(%)']:+.1f}%) — {best['Variant']} / {best['Session']}"
                  if best is not None else "N/A")

    html = f"""<!DOCTYPE html>
<html><head><title>ES Fisher 3-Lot Standalone — {tf}</title>
<style>
  body {{ background: #0d1117; color: #e0e0e0; font-family: monospace; padding: 20px; }}
  h1 {{ color: #58a6ff; font-size: 20px; }}
  h2 {{ color: #8b949e; font-size: 16px; margin-top: 5px; }}
  .summary {{ display: flex; gap: 20px; margin: 15px 0; flex-wrap: wrap; }}
  .card {{ background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 12px 16px; }}
  .card .num {{ font-size: 22px; font-weight: bold; }}
  .green {{ color: #3fb950; }} .red {{ color: #f85149; }} .yellow {{ color: #d29922; }}
</style></head><body>
<h1>Fisher 3-Lot Standalone — {tf} bars only</h1>
<h2>Entry: FT crosses its 20-SMA | Stop: 1.5×ATR | TP: 1/2/3×ATR | Ladder: BE → +1 ATR</h2>
<h2>4 variants × 4 sessions = {len(tbl_sorted)} runs | Account base = ${ACCOUNT_SIZE:,.0f} | ES point = ${POINT_VALUE}</h2>
<div class="summary">
  <div class="card"><div class="green num">{len(tbl_sorted[tbl_sorted['PF'] >= 1.0])}</div><div>Profitable (PF ≥ 1.0)</div></div>
  <div class="card"><div class="red num">{len(tbl_sorted[tbl_sorted['PF'] < 1.0])}</div><div>Losing</div></div>
  <div class="card"><div class="green num">{best_label}</div><div>Best run</div></div>
</div>
{chart_html}
<h2 style="margin-top:30px;">Results Table</h2>
{styled.to_html()}
</body></html>"""

    output = f"backtest_fisher_3lot_{tf}_chart.html"
    with open(output, "w") as f:
        f.write(html)
    print(f"Per-TF chart saved to {output}")


def build_html(all_results, tbl):
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        groups = []
        seen = set()
        for r in all_results:
            key = (r["TF"], r["Variant"])
            if key not in seen:
                seen.add(key); groups.append(key)

        fig = make_subplots(
            rows=len(groups), cols=1,
            subplot_titles=[f"{tf} — {v}" for tf, v in groups],
            vertical_spacing=0.04,
        )
        colors = ["#58a6ff", "#3fb950", "#d29922", "#f85149"]
        for row_idx, (tf, var) in enumerate(groups, 1):
            subset = [r for r in all_results if r["TF"] == tf and r["Variant"] == var]
            for j, r in enumerate(subset):
                fig.add_trace(go.Scatter(
                    y=r["equities"], mode="lines",
                    name=f"{r['Session']}",
                    line=dict(color=colors[j % len(colors)], width=1.3),
                    legendgroup=f"{tf}-{var}",
                    showlegend=True,
                ), row=row_idx, col=1)
            fig.update_yaxes(title_text="Points", row=row_idx, col=1)

        fig.update_layout(
            template="plotly_dark",
            height=320 * len(groups),
            title="Fisher Transform 3-Lot Scaled Exit (Gary spec) — 1.5 ATR stop, 1/2/3 ATR targets, BE → +1 ATR ladder",
        )
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
    best_label = (f"{best['Total(pts)']:+,.1f} pts — {best['Variant']} / {best['Session']} ({best['TF']})"
                  if best is not None else "N/A")

    html = f"""<!DOCTYPE html>
<html><head><title>ES Fisher 3-Lot Scaled Exit</title>
<style>
  body {{ background: #0d1117; color: #e0e0e0; font-family: monospace; padding: 20px; }}
  h1 {{ color: #58a6ff; font-size: 20px; }}
  h2 {{ color: #8b949e; font-size: 16px; margin-top: 5px; }}
  .summary {{ display: flex; gap: 20px; margin: 15px 0; flex-wrap: wrap; }}
  .card {{ background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 12px 16px; }}
  .card .num {{ font-size: 22px; font-weight: bold; }}
  .green {{ color: #3fb950; }} .red {{ color: #f85149; }} .yellow {{ color: #d29922; }}
</style></head><body>
<h1>Fisher Transform 3-Lot Scaled Exit (Gary spec)</h1>
<h2>Entry: FT crosses its 20-SMA | Stop: 1.5×ATR | TP: 1/2/3×ATR scaled | Stop ladder: BE → +1ATR</h2>
<h2>4 variants × 2 timeframes × 4 sessions = {len(tbl_sorted)} runs [POINTS]</h2>
<div class="summary">
  <div class="card"><div class="green num">{len(tbl_sorted[tbl_sorted['PF'] >= 1.0])}</div><div>Profitable (PF ≥ 1.0)</div></div>
  <div class="card"><div class="red num">{len(tbl_sorted[tbl_sorted['PF'] < 1.0])}</div><div>Losing</div></div>
  <div class="card"><div class="green num">{best_label}</div><div>Best run</div></div>
</div>
{chart_html}
<h2 style="margin-top:30px;">Results Table</h2>
{styled.to_html()}
</body></html>"""

    output = "backtest_fisher_3lot_chart.html"
    with open(output, "w") as f:
        f.write(html)

    import webbrowser, os
    webbrowser.open("file://" + os.path.abspath(output))
    print(f"\nChart saved to {output} and opened in browser.")


if __name__ == "__main__":
    main()
