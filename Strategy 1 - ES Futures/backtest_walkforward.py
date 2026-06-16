"""
Walk-Forward Test on the top strategy: HA+RSI 3-Lot MTF (60m+15m), US Morning.

Splits the ~1-year SPY data into 4 quarters and runs the strategy independently
on each. Reports per-quarter $ P&L, PF, win-rate, max drawdown, worst entry.
The goal is to see whether the edge is consistent across regimes or concentrated
in one favorable period.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

from data_feed import fetch_es_data
from indicators import compute_all
from backtest_haRSI_3lot_mtf_5m import get_60m_bias, simulate, summarize, SESSIONS, ACCOUNT_SIZE, POINT_VALUE

FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def resample(df_5m, freq):
    return df_5m.resample(freq).agg({
        "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum",
    }).dropna()


def main():
    print("Fetching 5m data...")
    df_5m_raw = fetch_es_data()

    print("Computing 15m indicators (entry timeframe)...")
    df_15m_full = compute_all(resample(df_5m_raw, "15min"))

    print("Computing 60m indicators (bias)...")
    df_60m_full = compute_all(resample(df_5m_raw, "60min"))

    # Determine 4-quarter splits based on the 15m data range
    start = df_15m_full.index[0]
    end = df_15m_full.index[-1]
    total_days = (end - start).days
    print(f"Data range: {start} -> {end} ({total_days} days)")

    # Equal-time splits: 4 folds
    quarter_starts = pd.date_range(start=start, end=end, periods=5)
    folds = []
    for i in range(4):
        fold_start = quarter_starts[i]
        fold_end = quarter_starts[i + 1]
        folds.append((f"Q{i+1}", fold_start, fold_end))

    # Top strategy: HA+RSI 3-Lot MTF (60m+15m), US Morning session (04:00-13:00 ET)
    SESS = ("US Morning (04:00-13:00)", 4, 0, 13, 0)
    sess_name, sh, sm, eh, em = SESS

    print(f"\nWalk-forward test: HA+RSI 3-Lot MTF (60m+15m), {sess_name} only")
    print(f"{'=' * 130}")

    rows = []
    eq_per_fold = {}

    for fold_name, fs, fe in folds:
        df15 = df_15m_full[(df_15m_full.index >= fs) & (df_15m_full.index < fe)].copy()
        df60 = df_60m_full[(df_60m_full.index >= fs) & (df_60m_full.index < fe)].copy()

        if len(df15) < 100:
            print(f"  {fold_name}: NOT ENOUGH DATA")
            continue

        bias_60m = get_60m_bias(df60)
        bias_arr = bias_60m.reindex(df15.index, method="ffill").fillna(0).astype(int).values

        eq, log, entry_groups = simulate(df15, bias_arr, sh, sm, eh, em)
        stats = summarize(log, entry_groups)

        if stats is None:
            print(f"  {fold_name} ({fs.date()} -> {fe.date()}): NO TRADES")
            continue

        eq_arr = np.array(eq)
        peak = np.maximum.accumulate(eq_arr)
        max_dd_pts = float((peak - eq_arr).max())
        max_dd_usd = max_dd_pts * POINT_VALUE

        eq_per_fold[fold_name] = eq

        print(f"  {fold_name} ({fs.date()} -> {fe.date()}, {len(df15)} bars): "
              f"E={stats['entries']:4d}  "
              f"P&L=${stats['total_usd']:+9,.0f}  "
              f"Ret={stats['return_pct']:+6.1f}%  "
              f"WR={stats['wr']:5.1f}%  PF={stats['pf']:5.2f}  "
              f"MaxDD=${max_dd_usd:,.0f}  "
              f"Worst=${stats['worst_entry_pts']*POINT_VALUE:+,.0f}")

        rows.append({
            "Fold": fold_name,
            "Start": str(fs.date()),
            "End": str(fe.date()),
            "Bars": len(df15),
            "Entries": stats["entries"],
            "$ P&L": stats["total_usd"],
            "Return %": stats["return_pct"],
            "WR %": stats["wr"],
            "PF": stats["pf"],
            "MaxDD $": max_dd_usd,
            "Worst Entry $": stats["worst_entry_pts"] * POINT_VALUE,
            "Avg Stop $/lot": stats["avg_stop_pts"] * POINT_VALUE,
        })

    # Composite (all folds combined)
    total_pnl = sum(r["$ P&L"] for r in rows)
    total_entries = sum(r["Entries"] for r in rows)
    avg_pf = np.mean([r["PF"] for r in rows])
    avg_wr = np.mean([r["WR %"] for r in rows])
    profitable_folds = sum(1 for r in rows if r["$ P&L"] > 0)

    print(f"\n{'=' * 130}")
    print(f"  COMPOSITE: {profitable_folds}/{len(rows)} folds profitable | "
          f"Total = ${total_pnl:+,.0f} | Avg PF = {avg_pf:.2f} | Avg WR = {avg_wr:.1f}%")
    print(f"{'=' * 130}")

    # Plot per-fold equity curves
    fig, ax = plt.subplots(figsize=(12, 6))
    plt.rcParams.update({
        "axes.facecolor": "#0d1117", "figure.facecolor": "#0d1117",
        "axes.edgecolor": "#30363d", "axes.labelcolor": "#e0e0e0",
        "xtick.color": "#8b949e", "ytick.color": "#8b949e",
        "axes.titlecolor": "#58a6ff", "text.color": "#e0e0e0",
        "axes.grid": True, "grid.color": "#21262d", "grid.alpha": 0.7,
        "font.family": "monospace",
    })
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_facecolor("#0d1117")
    fig.patch.set_facecolor("#0d1117")
    colors = ["#58a6ff", "#3fb950", "#d29922", "#f85149"]
    for i, (fold_name, eq) in enumerate(eq_per_fold.items()):
        ax.plot(eq, label=fold_name, color=colors[i % len(colors)], linewidth=1.5)
    ax.axhline(0, color="#666", linewidth=0.6, linestyle="--")
    ax.set_title("Walk-Forward: HA+RSI 3-Lot MTF (60m+15m), US Morning — per-quarter equity",
                 fontsize=12, color="#58a6ff", pad=10)
    ax.set_xlabel("Bar number (within fold)", color="#e0e0e0")
    ax.set_ylabel("Cumulative Points", color="#e0e0e0")
    ax.legend(loc="best", frameon=False, fontsize=10, labelcolor="#e0e0e0")
    ax.tick_params(colors="#8b949e")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "walkforward_haRSI_3lot_15m_morning.png")
    plt.savefig(fig_path, dpi=120, facecolor=fig.get_facecolor())
    plt.close()
    print(f"\nChart saved to {fig_path}")

    # Write HTML output too
    df = pd.DataFrame(rows)
    df_html = df.copy()
    df_html["$ P&L"] = df_html["$ P&L"].apply(lambda x: f"${x:+,.0f}")
    df_html["Return %"] = df_html["Return %"].apply(lambda x: f"{x:+.1f}%")
    df_html["WR %"] = df_html["WR %"].apply(lambda x: f"{x:.1f}%")
    df_html["PF"] = df_html["PF"].apply(lambda x: f"{x:.2f}")
    df_html["MaxDD $"] = df_html["MaxDD $"].apply(lambda x: f"${x:,.0f}")
    df_html["Worst Entry $"] = df_html["Worst Entry $"].apply(lambda x: f"${x:+,.0f}")
    df_html["Avg Stop $/lot"] = df_html["Avg Stop $/lot"].apply(lambda x: f"${x:.0f}")
    print("\n" + df_html.to_string(index=False))

    html = f"""<!DOCTYPE html>
<html><head><title>Walk-Forward Test</title>
<style>
  body {{ background: #0d1117; color: #e0e0e0; font-family: monospace; padding: 20px; }}
  h1 {{ color: #58a6ff; }}
  table {{ border-collapse: collapse; margin-top: 20px; }}
  th, td {{ padding: 8px 14px; border: 1px solid #30363d; }}
  th {{ background: #1a1a2e; }}
  tr:nth-child(even) {{ background: #161b22; }}
</style></head><body>
<h1>Walk-Forward Test — HA+RSI 3-Lot MTF (60m+15m), US Morning</h1>
<p>Total composite P&amp;L: <b>${total_pnl:+,.0f}</b> | {profitable_folds}/{len(rows)} folds profitable | Avg PF: {avg_pf:.2f} | Avg WR: {avg_wr:.1f}%</p>
<img src="figures/walkforward_haRSI_3lot_15m_morning.png" style="max-width:100%;border:1px solid #30363d;">
{df_html.to_html(index=False, border=0)}
</body></html>"""
    with open("walkforward_chart.html", "w") as f:
        f.write(html)
    import webbrowser
    webbrowser.open("file://" + os.path.abspath("walkforward_chart.html"))
    print("\nHTML report opened in browser.")


if __name__ == "__main__":
    main()
