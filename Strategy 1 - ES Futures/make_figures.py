"""
Generate per-strategy PNG equity curve charts for the LaTeX summary doc.
Saves to ./figures/<strategy_id>.png
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_feed import fetch_es_data
from indicators import compute_all
from config import RSI_BUY_THRESHOLD, RSI_SELL_THRESHOLD

FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

POINT_VALUE = 50.0
ACCOUNT = 50_000.0
FT_SMA_PERIOD = 20

SESSIONS = [
    ("All Day", 0, 0, 23, 59),
    ("Overnight", 0, 0, 4, 0),
    ("US Morning", 4, 0, 13, 0),
    ("US Afternoon", 13, 0, 17, 0),
]

plt.rcParams.update({
    "axes.facecolor": "#0d1117",
    "figure.facecolor": "#0d1117",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#e0e0e0",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "axes.titlecolor": "#58a6ff",
    "text.color": "#e0e0e0",
    "axes.grid": True,
    "grid.color": "#21262d",
    "grid.alpha": 0.7,
    "font.family": "monospace",
})

COLORS = {
    "All Day": "#58a6ff",
    "Overnight": "#f85149",
    "US Morning": "#3fb950",
    "US Afternoon": "#d29922",
}


def save_equity_chart(fname, title, sess_to_equities, ylabel="Cumulative Points"):
    """Save a 4-session equity-curve chart."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for sess, eq in sess_to_equities.items():
        ax.plot(eq, label=sess, color=COLORS.get(sess, "#888"), linewidth=1.4)
    ax.axhline(0, color="#666", linewidth=0.6, linestyle="--")
    ax.set_title(title, fontsize=11, pad=10)
    ax.set_xlabel("Bar number")
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, fname)
    plt.savefig(path, dpi=120, facecolor=fig.get_facecolor())
    plt.close()
    print(f"  saved {path}")


def resample(df_5m, freq):
    return df_5m.resample(freq).agg({
        "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum",
    }).dropna()


# ============================================================
# 1. HA+RSI 3-Lot MTF (60m + 5m & 60m + 15m) — top strategies
# ============================================================

def run_ha_rsi_3lot_mtf(df_entry, df_60m):
    """Returns dict[session_name] = equities list"""
    from backtest_haRSI_3lot_mtf_5m import get_60m_bias, simulate
    bias_60m = get_60m_bias(df_60m)
    bias_arr = bias_60m.reindex(df_entry.index, method="ffill").fillna(0).astype(int).values
    out = {}
    for sess_name, sh, sm, eh, em in SESSIONS:
        eq, _, _ = simulate(df_entry, bias_arr, sh, sm, eh, em)
        out[sess_name] = eq
    return out


# ============================================================
# 2. Fisher 3-Lot Standalone (5m, 15m, 60m)
# ============================================================

def run_fisher_3lot_standalone(df, ft_col, sma_col, use_rsi):
    from backtest_fisher_3lot import simulate
    out = {}
    for sess_name, sh, sm, eh, em in SESSIONS:
        eq, _, _ = simulate(df, ft_col, sma_col, use_rsi, sh, sm, eh, em)
        out[sess_name] = eq
    return out


# ============================================================
# 3. Fisher 3-Lot MTF (60m + 5m / 60m + 15m)
# ============================================================

def run_fisher_3lot_mtf(df_entry, df_60m, ft_col, sma_col, use_rsi):
    from backtest_fisher_3lot_mtf import compute_60m_bias, simulate
    bias_60m = compute_60m_bias(df_60m, ft_col, sma_col)
    bias_arr = bias_60m.reindex(df_entry.index, method="ffill").fillna(0).astype(int).values
    out = {}
    for sess_name, sh, sm, eh, em in SESSIONS:
        eq, _, _ = simulate(df_entry, ft_col, sma_col, use_rsi, bias_arr, sh, sm, eh, em)
        out[sess_name] = eq
    return out


# ============================================================
# 4. MTF 60m+5m 1-contract (HA+RSI) — Best Overnight scalper
# ============================================================

def run_mtf_5m_1lot(df_5m, df_60m, stop_mult, target_mult):
    from backtest_mtf_5m import get_60m_bias, align_bias_to_5m, simulate
    bias_60m = get_60m_bias(df_60m)
    bias_5m = align_bias_to_5m(bias_60m, df_5m)
    out = {}
    for sess_name, sh, sm, eh, em in SESSIONS:
        eq, _ = simulate(df_5m, bias_5m, stop_mult, target_mult, sh, sm, eh, em)
        out[sess_name] = eq
    return out


def main():
    print("Loading data...")
    df_5m_raw = fetch_es_data()

    print("Computing indicators on 5m...")
    df_5m = compute_all(df_5m_raw)
    df_5m["Fisher_SMA20"]    = df_5m["Fisher"].rolling(FT_SMA_PERIOD).mean()
    df_5m["Fisher_HA_SMA20"] = df_5m["Fisher_HA"].rolling(FT_SMA_PERIOD).mean()

    print("Computing indicators on 15m...")
    df_15m = compute_all(resample(df_5m_raw, "15min"))
    df_15m["Fisher_SMA20"]    = df_15m["Fisher"].rolling(FT_SMA_PERIOD).mean()
    df_15m["Fisher_HA_SMA20"] = df_15m["Fisher_HA"].rolling(FT_SMA_PERIOD).mean()

    print("Computing indicators on 60m...")
    df_60m = compute_all(resample(df_5m_raw, "60min"))
    df_60m["Fisher_SMA20"]    = df_60m["Fisher"].rolling(FT_SMA_PERIOD).mean()
    df_60m["Fisher_HA_SMA20"] = df_60m["Fisher_HA"].rolling(FT_SMA_PERIOD).mean()

    # ---------- HA+RSI 3-Lot MTF 60m+15m (TOP) ----------
    print("\n[1/8] HA+RSI 3-Lot MTF (60m+15m) — TOP")
    eqs = run_ha_rsi_3lot_mtf(df_15m, df_60m)
    save_equity_chart("haRSI_3lot_mtf_15m.png",
                      "HA+RSI 3-Lot MTF (60m+15m) — Equity by Session",
                      eqs)

    # ---------- HA+RSI 3-Lot MTF 60m+5m ----------
    print("[2/8] HA+RSI 3-Lot MTF (60m+5m)")
    eqs = run_ha_rsi_3lot_mtf(df_5m, df_60m)
    save_equity_chart("haRSI_3lot_mtf_5m.png",
                      "HA+RSI 3-Lot MTF (60m+5m) — Equity by Session",
                      eqs)

    # ---------- Fisher 3-Lot Standalone 60m ----------
    print("[3/8] Fisher 3-Lot Standalone 60m (V2 FT(HA) no-RSI)")
    eqs = run_fisher_3lot_standalone(df_60m, "Fisher_HA", "Fisher_HA_SMA20", False)
    save_equity_chart("fisher_3lot_60m.png",
                      "Fisher 3-Lot Standalone 60m — V2 FT(HA) no-RSI",
                      eqs)

    # ---------- Fisher 3-Lot Standalone 15m ----------
    print("[4/8] Fisher 3-Lot Standalone 15m (V2 FT(HA) no-RSI)")
    eqs = run_fisher_3lot_standalone(df_15m, "Fisher_HA", "Fisher_HA_SMA20", False)
    save_equity_chart("fisher_3lot_15m.png",
                      "Fisher 3-Lot Standalone 15m — V2 FT(HA) no-RSI",
                      eqs)

    # ---------- Fisher 3-Lot Standalone 5m ----------
    print("[5/8] Fisher 3-Lot Standalone 5m (V4 FT(HA)+RSI)")
    eqs = run_fisher_3lot_standalone(df_5m, "Fisher_HA", "Fisher_HA_SMA20", True)
    save_equity_chart("fisher_3lot_5m.png",
                      "Fisher 3-Lot Standalone 5m — V4 FT(HA)+RSI",
                      eqs)

    # ---------- Fisher 3-Lot MTF 60m+15m ----------
    print("[6/8] Fisher 3-Lot MTF (60m+15m) — V4 FT(HA)+RSI")
    eqs = run_fisher_3lot_mtf(df_15m, df_60m, "Fisher_HA", "Fisher_HA_SMA20", True)
    save_equity_chart("fisher_3lot_mtf_15m.png",
                      "Fisher 3-Lot MTF (60m+15m) — V4 FT(HA)+RSI",
                      eqs)

    # ---------- Fisher 3-Lot MTF 60m+5m ----------
    print("[7/8] Fisher 3-Lot MTF (60m+5m) — V1 FT(Close) no-RSI")
    eqs = run_fisher_3lot_mtf(df_5m, df_60m, "Fisher", "Fisher_SMA20", False)
    save_equity_chart("fisher_3lot_mtf_5m.png",
                      "Fisher 3-Lot MTF (60m+5m) — V1 FT(Close) no-RSI",
                      eqs)

    # ---------- MTF 60m+5m 1-contract overnight scalper ----------
    print("[8/8] MTF 60m+5m 1-contract — SL 1.75x / TP 0.25x (overnight winner)")
    eqs = run_mtf_5m_1lot(df_5m, df_60m, 1.75, 0.25)
    save_equity_chart("mtf_5m_overnight_scalper.png",
                      "MTF 60m+5m 1-contract — SL 1.75x / TP 0.25x (overnight scalper)",
                      eqs)

    print("\nAll figures saved to ./figures/")


if __name__ == "__main__":
    main()
