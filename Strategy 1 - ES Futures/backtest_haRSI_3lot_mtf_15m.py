"""
Strategy: HA EMA + RSI 3-Lot Scaled Exit — MTF (60m bias + 15m entries)

Same as backtest_haRSI_3lot_mtf_5m.py but entries fire on 15m bars.
60m HA+RSI sets bias. 15m HA+RSI fires entries when aligned with bias.

Signal:
  LONG  : HA close > 20-EMA(High) AND RSI > 55
  SHORT : HA close < 20-EMA(Low)  AND RSI < 45

Position: 3 contracts. Stop 1.5×ATR(15m), TP 1×/2×/3×ATR(15m).
Ladder: TP1 -> BE, TP2 -> trail to +1 ATR, TP3 closes runner.
"""

import pandas as pd

from data_feed import fetch_es_data
from indicators import compute_all
from config import RSI_BUY_THRESHOLD, RSI_SELL_THRESHOLD

# Reuse the simulation engine from the 5m variant — logic is identical, only the
# entry DataFrame's bar size changes.
from backtest_haRSI_3lot_mtf_5m import (
    resample, get_60m_bias, simulate, summarize, build_html,
    SESSIONS, ACCOUNT_SIZE, POINT_VALUE,
)


def align_bias_to_entry(bias_60m, df_entry):
    bias = bias_60m.reindex(df_entry.index, method="ffill").fillna(0).astype(int)
    return bias.values


def main():
    print("Fetching 5m data...")
    df_5m_raw = fetch_es_data()

    print("Computing 15m indicators (entry timeframe)...")
    df_15m = compute_all(resample(df_5m_raw, "15min"))

    print("Computing 60m indicators (bias)...")
    df_60m = compute_all(resample(df_5m_raw, "60min"))

    print(f"15m bars: {len(df_15m)} | 60m bars: {len(df_60m)}")
    print(f"RSI thresholds: Buy > {RSI_BUY_THRESHOLD}, Sell < {RSI_SELL_THRESHOLD}")

    bias_60m = get_60m_bias(df_60m)
    bias_arr = align_bias_to_entry(bias_60m, df_15m)
    buy_bars  = int((bias_arr == 1).sum())
    sell_bars = int((bias_arr == -1).sum())
    neutral_bars = int((bias_arr == 0).sum())
    print(f"60m bias distribution on 15m bars: BUY={buy_bars} SELL={sell_bars} NEUTRAL={neutral_bars}")
    print(f"{'=' * 100}")

    all_results = []
    for sess_name, sh, sm, eh, em in SESSIONS:
        eq, log, entry_groups = simulate(df_15m, bias_arr, sh, sm, eh, em)
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
    print("  HA+RSI 3-LOT MTF (60m bias + 15m entries) — ALL RUNS (sorted by PF)")
    print(f"{'=' * 140}")
    print(tbl.sort_values("PF", ascending=False).to_string(index=False, float_format="%.2f"))

    # Use the same build_html but override the output filename via a small shim
    _build_html_15m(all_results, tbl)


def _build_html_15m(all_results, tbl):
    """Same as backtest_haRSI_3lot_mtf_5m.build_html but with 15m title and output filename."""
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
                          title="HA+RSI 3-Lot MTF (60m bias + 15m entries) — Equity Curves")
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
<html><head><title>ES HA+RSI 3-Lot MTF (60m+15m)</title>
<style>
  body {{ background: #0d1117; color: #e0e0e0; font-family: monospace; padding: 20px; }}
  h1 {{ color: #58a6ff; font-size: 20px; }}
  h2 {{ color: #8b949e; font-size: 16px; margin-top: 5px; }}
  .summary {{ display: flex; gap: 20px; margin: 15px 0; flex-wrap: wrap; }}
  .card {{ background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 12px 16px; }}
  .card .num {{ font-size: 22px; font-weight: bold; }}
  .green {{ color: #3fb950; }} .red {{ color: #f85149; }} .yellow {{ color: #d29922; }}
</style></head><body>
<h1>HA EMA + RSI 3-Lot Scaled Exit — MTF (60m bias + 15m entries)</h1>
<h2>Bias: 60m HA cross EMA + RSI 55/45 | Entry: 15m same signal aligned with 60m bias | Stop: 1.5×ATR | TP: 1/2/3×ATR | Ladder: BE → +1ATR</h2>
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

    output = "backtest_haRSI_3lot_mtf_15m_chart.html"
    with open(output, "w") as f:
        f.write(html)

    import webbrowser, os
    webbrowser.open("file://" + os.path.abspath(output))
    print(f"\nChart saved to {output} and opened in browser.")


if __name__ == "__main__":
    main()
