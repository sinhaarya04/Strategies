"""
Tabbed comparison: full grid of stop x target ATR multipliers.
Dropdown tabs to switch between each strategy variant.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data_feed import fetch_es_data
from indicators import compute_all
from config import (
    RSI_BUY_THRESHOLD, RSI_SELL_THRESHOLD,
    INITIAL_CAPITAL, ES_POINT_VALUE,
)

# Full grid: stop x target
STOP_RANGE = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
TARGET_RANGE = [None, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]


def simulate(df, atr_stop_mult, atr_target_mult=None):
    position = 0
    entry_price = stop_price = target_price = 0.0
    equity = INITIAL_CAPITAL

    trades = []
    equities = [equity]
    signals = [0]

    for i in range(1, len(df)):
        ha_c = df["HA_Close"].iloc[i]
        ema_h = df["EMA_High"].iloc[i]
        ema_l = df["EMA_Low"].iloc[i]
        rsi = df["RSI"].iloc[i]
        atr = df["ATR"].iloc[i]
        close = df["Close"].iloc[i]
        signal = 0

        if np.isnan(rsi) or np.isnan(ema_h) or np.isnan(atr):
            equities.append(equity)
            signals.append(0)
            continue

        # Stop loss
        if position == 1 and close <= stop_price:
            equity += (close - entry_price) * ES_POINT_VALUE
            trades.append({"pnl": (close - entry_price) * ES_POINT_VALUE, "exit_type": "stop"})
            position = 0; signal = -1
        elif position == -1 and close >= stop_price:
            equity += (entry_price - close) * ES_POINT_VALUE
            trades.append({"pnl": (entry_price - close) * ES_POINT_VALUE, "exit_type": "stop"})
            position = 0; signal = 1

        # Take profit
        if atr_target_mult is not None:
            if position == 1 and close >= target_price:
                equity += (close - entry_price) * ES_POINT_VALUE
                trades.append({"pnl": (close - entry_price) * ES_POINT_VALUE, "exit_type": "target"})
                position = 0; signal = -1
            elif position == -1 and close <= target_price:
                equity += (entry_price - close) * ES_POINT_VALUE
                trades.append({"pnl": (entry_price - close) * ES_POINT_VALUE, "exit_type": "target"})
                position = 0; signal = 1

        # Entry
        if position == 0:
            if ha_c > ema_h and rsi > RSI_BUY_THRESHOLD:
                position = 1; entry_price = close
                stop_price = close - atr * atr_stop_mult
                target_price = close + atr * atr_target_mult if atr_target_mult else 0
                signal = 1
            elif ha_c < ema_l and rsi < RSI_SELL_THRESHOLD:
                position = -1; entry_price = close
                stop_price = close + atr * atr_stop_mult
                target_price = close - atr * atr_target_mult if atr_target_mult else 0
                signal = -1
        elif position == 1:
            if ha_c < ema_l and rsi < RSI_SELL_THRESHOLD:
                equity += (close - entry_price) * ES_POINT_VALUE
                trades.append({"pnl": (close - entry_price) * ES_POINT_VALUE, "exit_type": "reversal"})
                position = 0; signal = -1
        elif position == -1:
            if ha_c > ema_h and rsi > RSI_BUY_THRESHOLD:
                equity += (entry_price - close) * ES_POINT_VALUE
                trades.append({"pnl": (entry_price - close) * ES_POINT_VALUE, "exit_type": "reversal"})
                position = 0; signal = 1

        if position == 1:
            mtm = equity + (close - entry_price) * ES_POINT_VALUE
        elif position == -1:
            mtm = equity + (entry_price - close) * ES_POINT_VALUE
        else:
            mtm = equity

        equities.append(mtm)
        signals.append(signal)

    return equities, trades, signals


def stats(trades, equities):
    n = len(trades)
    if n == 0:
        return dict(pnl=0, ret=0, max_dd=0, trades=0, wins=0, wr=0, avg_w=0, avg_l=0, pf=0)
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    pnl = equities[-1] - INITIAL_CAPITAL
    eq = np.array(equities)
    dd = ((np.maximum.accumulate(eq) - eq) / np.maximum.accumulate(eq) * 100)
    pf = abs(sum(t["pnl"] for t in wins)) / abs(sum(t["pnl"] for t in losses)) if losses else float("inf")
    return dict(
        pnl=pnl, ret=pnl / INITIAL_CAPITAL * 100, max_dd=dd.max(),
        trades=n, wins=len(wins), wr=len(wins) / n * 100 if n else 0,
        avg_w=np.mean([t["pnl"] for t in wins]) if wins else 0,
        avg_l=np.mean([t["pnl"] for t in losses]) if losses else 0,
        pf=pf,
    )


def build_tab_traces(df, equities, signals, color_main):
    buy_idx = [i for i in range(len(df)) if signals[i] == 1]
    sell_idx = [i for i in range(len(df)) if signals[i] == -1]

    traces = []

    # Panel 1: EMA bands + price + signals
    traces.append(go.Scatter(
        x=df.index, y=df["EMA_High"], name="EMA(20) High",
        line=dict(color="rgba(0,150,255,0.4)", width=1), showlegend=False,
    ))
    traces.append(go.Scatter(
        x=df.index, y=df["EMA_Low"], name="EMA(20) Low",
        line=dict(color="rgba(0,150,255,0.4)", width=1),
        fill="tonexty", fillcolor="rgba(0,150,255,0.07)", showlegend=False,
    ))
    traces.append(go.Scatter(
        x=df.index, y=df["HA_Close"], name="HA Close",
        line=dict(color="white", width=1.5), showlegend=False,
    ))
    traces.append(go.Scatter(
        x=df.index, y=df["Close"], name="Close",
        line=dict(color="rgba(180,180,180,0.4)", width=1), showlegend=False,
    ))
    traces.append(go.Scatter(
        x=df.index[buy_idx], y=df["Close"].iloc[buy_idx],
        mode="markers", name="BUY",
        marker=dict(symbol="triangle-up", size=10, color="#00ff88",
                   line=dict(width=1, color="white")), showlegend=False,
    ))
    traces.append(go.Scatter(
        x=df.index[sell_idx], y=df["Close"].iloc[sell_idx],
        mode="markers", name="SELL",
        marker=dict(symbol="triangle-down", size=10, color="#ff4444",
                   line=dict(width=1, color="white")), showlegend=False,
    ))

    # Panel 2: Equity
    r, g, b = int(color_main.lstrip("#")[0:2], 16), int(color_main.lstrip("#")[2:4], 16), int(color_main.lstrip("#")[4:6], 16)
    traces.append(go.Scatter(
        x=df.index, y=equities, name="Equity",
        line=dict(color=color_main, width=2),
        fill="tozeroy", fillcolor=f"rgba({r},{g},{b},0.1)",
    ))

    # Panel 3: RSI
    traces.append(go.Scatter(
        x=df.index, y=df["RSI"], name="RSI",
        line=dict(color="#ffaa00", width=1.5), showlegend=False,
    ))

    # Panel 4: ATR
    traces.append(go.Scatter(
        x=df.index, y=df["ATR"], name="ATR",
        line=dict(color="#aa66ff", width=1.5),
        fill="tozeroy", fillcolor="rgba(170,102,255,0.1)", showlegend=False,
    ))

    return traces  # 10 traces per tab


def main():
    print("Fetching data...")
    df = fetch_es_data()
    df = compute_all(df)

    # Run all combos, collect results sorted by composite rank
    all_variants = []
    for stop in STOP_RANGE:
        for target in TARGET_RANGE:
            t_label = "None" if target is None else f"{target:.2f}x"
            label = f"SL={stop:.2f}x TP={t_label}"
            print(f"Running: {label}...")
            eq, tr, sig = simulate(df, atr_stop_mult=stop, atr_target_mult=target)
            s = stats(tr, eq)
            all_variants.append((label, stop, target, eq, tr, sig, s))

    # Sort by composite rank (PF + PnL + Sharpe proxy)
    for v in all_variants:
        v[6]["_pf_rank"] = 0
        v[6]["_pnl_rank"] = 0
    sorted_by_pf = sorted(all_variants, key=lambda x: x[6]["pf"], reverse=True)
    sorted_by_pnl = sorted(all_variants, key=lambda x: x[6]["pnl"], reverse=True)
    for rank, v in enumerate(sorted_by_pf):
        v[6]["_pf_rank"] = rank
    for rank, v in enumerate(sorted_by_pnl):
        v[6]["_pnl_rank"] = rank
    all_variants.sort(key=lambda x: x[6]["_pf_rank"] + x[6]["_pnl_rank"])

    # Print top 15
    print(f"\nTop 15 of {len(all_variants)} combos (by composite rank):")
    for i, (label, stop, target, eq, tr, sig, s) in enumerate(all_variants[:15]):
        print(f"  {i+1:2d}. {label:20s} P&L ${s['pnl']:+,.0f} | {s['trades']} trades | Win {s['wr']:.0f}% | DD {s['max_dd']:.1f}% | PF {s['pf']:.2f}")

    # Color palette — cycle through
    palette = [
        "#ff6b6b", "#00ff88", "#4dabf7", "#ffd43b", "#ff922b",
        "#da77f2", "#69db7c", "#74c0fc", "#f06595", "#a9e34b",
        "#e599f7", "#ffa94d", "#66d9e8", "#c0eb75", "#ff8787",
    ]

    # Build figure
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.45, 0.20, 0.15, 0.20],
        subplot_titles=("ES E-mini 5m — Price + EMA Bands + Signals",
                       "Equity Curve", "RSI (14)", "ATR (14)")
    )

    row_map = [1, 1, 1, 1, 1, 1, 2, 3, 4]
    n_traces = 9

    for vi, (label, stop, target, eq, tr, sig, s) in enumerate(all_variants):
        color = palette[vi % len(palette)]
        traces = build_tab_traces(df, eq, sig, color)
        visible = (vi == 0)
        for i, trace in enumerate(traces):
            trace.visible = visible
            fig.add_trace(trace, row=row_map[i], col=1)

    # Dropdown buttons
    buttons = []
    for vi, (label, stop, target, eq, tr, sig, s) in enumerate(all_variants):
        vis = [False] * (n_traces * len(all_variants))
        for j in range(n_traces):
            vis[vi * n_traces + j] = True
        btn_label = (f"{label} | ${s['pnl']:+,.0f} ({s['ret']:+.1f}%) | "
                     f"{s['trades']} tr | W {s['wr']:.0f}% | "
                     f"DD {s['max_dd']:.1f}% | PF {s['pf']:.2f}")
        buttons.append(dict(label=btn_label, method="update", args=[{"visible": vis}]))

    fig.update_layout(
        updatemenus=[dict(
            type="dropdown",
            direction="down",
            x=0.5, xanchor="center", y=1.08,
            buttons=buttons,
            font=dict(size=11),
            bgcolor="rgba(30,30,30,0.9)",
        )],
        template="plotly_dark",
        title=dict(
            text=f"ES 5m Backtest — {len(all_variants)} Stop x Target Combos (ranked by PF + P&L)",
            font=dict(size=18),
        ),
        height=1000,
        hovermode="x unified",
    )

    fig.add_hline(y=50, line_dash="dash", line_color="gray", row=3, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="rgba(255,0,0,0.3)", row=3, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="rgba(0,255,0,0.3)", row=3, col=1)
    fig.add_hline(y=INITIAL_CAPITAL, line_dash="dash", line_color="gray",
                  annotation_text=f"Start: ${INITIAL_CAPITAL:,.0f}", row=2, col=1)

    for i in range(1, 5):
        fig.update_yaxes(gridcolor="rgba(50,50,50,0.5)", row=i, col=1)
        fig.update_xaxes(gridcolor="rgba(50,50,50,0.5)", row=i, col=1)

    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="$", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1, range=[0, 100])
    fig.update_yaxes(title_text="ATR", row=4, col=1)

    output = "compare_chart.html"
    fig.write_html(output, auto_open=True)
    print(f"\nChart saved to {output} and opened in browser.")


if __name__ == "__main__":
    main()
