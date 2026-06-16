"""
Clean interactive visualization of ES backtest results.
Outputs an HTML file that opens in browser.
"""

import sys
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data_feed import fetch_es_data
from indicators import compute_all
from config import (
    RSI_BUY_THRESHOLD, RSI_SELL_THRESHOLD,
    ATR_RISK_MULTIPLIER, RISK_PER_TRADE_PCT,
    INITIAL_CAPITAL, ES_POINT_VALUE,
)


def simulate_trades(df):
    """
    Vectorized trade simulation matching the backtrader strategy.
    Returns df with signals and equity curve.
    """
    position = 0  # 1 = long, -1 = short, 0 = flat
    entry_price = 0.0
    stop_price = 0.0
    equity = INITIAL_CAPITAL

    trades = []
    equities = [equity]
    positions = [0]
    signals = [0]  # 1=buy, -1=sell, 0=none

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
            positions.append(position)
            signals.append(0)
            continue

        # Check stop loss
        if position == 1 and close <= stop_price:
            pnl = (close - entry_price) * ES_POINT_VALUE
            equity += pnl
            trades.append({"entry_idx": entry_idx, "exit_idx": i, "side": "long",
                          "entry": entry_price, "exit": close, "pnl": pnl})
            position = 0
            signal = -1
        elif position == -1 and close >= stop_price:
            pnl = (entry_price - close) * ES_POINT_VALUE
            equity += pnl
            trades.append({"entry_idx": entry_idx, "exit_idx": i, "side": "short",
                          "entry": entry_price, "exit": close, "pnl": pnl})
            position = 0
            signal = 1

        # Entry / exit signals
        if position == 0:
            if ha_c > ema_h and rsi > RSI_BUY_THRESHOLD:
                position = 1
                entry_price = close
                entry_idx = i
                stop_price = close - atr * ATR_RISK_MULTIPLIER
                signal = 1
            elif ha_c < ema_l and rsi < RSI_SELL_THRESHOLD:
                position = -1
                entry_price = close
                entry_idx = i
                stop_price = close + atr * ATR_RISK_MULTIPLIER
                signal = -1
        elif position == 1:
            if ha_c < ema_l and rsi < RSI_SELL_THRESHOLD:
                pnl = (close - entry_price) * ES_POINT_VALUE
                equity += pnl
                trades.append({"entry_idx": entry_idx, "exit_idx": i, "side": "long",
                              "entry": entry_price, "exit": close, "pnl": pnl})
                position = 0
                signal = -1
        elif position == -1:
            if ha_c > ema_h and rsi > RSI_BUY_THRESHOLD:
                pnl = (entry_price - close) * ES_POINT_VALUE
                equity += pnl
                trades.append({"entry_idx": entry_idx, "exit_idx": i, "side": "short",
                              "entry": entry_price, "exit": close, "pnl": pnl})
                position = 0
                signal = 1

        # Mark-to-market equity
        if position == 1:
            mtm = equity + (close - entry_price) * ES_POINT_VALUE
        elif position == -1:
            mtm = equity + (entry_price - close) * ES_POINT_VALUE
        else:
            mtm = equity

        equities.append(mtm)
        positions.append(position)
        signals.append(signal)

    df = df.copy()
    df["Equity"] = equities
    df["Position"] = positions
    df["Signal"] = signals

    return df, trades


def plot_results(df, trades, output="backtest_chart.html"):
    """Create clean interactive plotly chart."""

    buy_mask = df["Signal"] == 1
    sell_mask = df["Signal"] == -1

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.45, 0.20, 0.15, 0.20],
        subplot_titles=("ES E-mini — Price + EMA Bands + Signals",
                       "Equity Curve",
                       "RSI (14)",
                       "ATR (14)")
    )

    # --- Row 1: Price chart with HA candles + EMA bands ---

    # EMA band (shaded)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["EMA_High"], name="EMA(20) High",
        line=dict(color="rgba(0,150,255,0.4)", width=1),
        showlegend=True,
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df.index, y=df["EMA_Low"], name="EMA(20) Low",
        line=dict(color="rgba(0,150,255,0.4)", width=1),
        fill="tonexty", fillcolor="rgba(0,150,255,0.07)",
        showlegend=True,
    ), row=1, col=1)

    # Heikin Ashi candles as line
    fig.add_trace(go.Scatter(
        x=df.index, y=df["HA_Close"], name="HA Close",
        line=dict(color="white", width=1.5),
    ), row=1, col=1)

    # Regular close
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Close"], name="Close",
        line=dict(color="rgba(180,180,180,0.5)", width=1),
    ), row=1, col=1)

    # Buy signals
    fig.add_trace(go.Scatter(
        x=df.index[buy_mask], y=df["Close"][buy_mask],
        mode="markers", name="BUY",
        marker=dict(symbol="triangle-up", size=10, color="#00ff88",
                   line=dict(width=1, color="white")),
    ), row=1, col=1)

    # Sell signals
    fig.add_trace(go.Scatter(
        x=df.index[sell_mask], y=df["Close"][sell_mask],
        mode="markers", name="SELL",
        marker=dict(symbol="triangle-down", size=10, color="#ff4444",
                   line=dict(width=1, color="white")),
    ), row=1, col=1)

    # --- Row 2: Equity curve ---
    equity_color = ["#00ff88" if df["Equity"].iloc[i] >= INITIAL_CAPITAL else "#ff4444"
                    for i in range(len(df))]

    fig.add_trace(go.Scatter(
        x=df.index, y=df["Equity"], name="Equity",
        line=dict(color="#00ff88", width=2),
        fill="tozeroy", fillcolor="rgba(0,255,136,0.1)",
    ), row=2, col=1)

    fig.add_hline(y=INITIAL_CAPITAL, line_dash="dash", line_color="gray",
                  annotation_text=f"Start: ${INITIAL_CAPITAL:,.0f}", row=2, col=1)

    # --- Row 3: RSI ---
    fig.add_trace(go.Scatter(
        x=df.index, y=df["RSI"], name="RSI",
        line=dict(color="#ffaa00", width=1.5),
    ), row=3, col=1)

    fig.add_hline(y=50, line_dash="dash", line_color="gray", row=3, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="rgba(255,0,0,0.3)", row=3, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="rgba(0,255,0,0.3)", row=3, col=1)

    # --- Row 4: ATR ---
    fig.add_trace(go.Scatter(
        x=df.index, y=df["ATR"], name="ATR",
        line=dict(color="#aa66ff", width=1.5),
        fill="tozeroy", fillcolor="rgba(170,102,255,0.1)",
    ), row=4, col=1)

    # --- Stats annotation ---
    final_equity = df["Equity"].iloc[-1]
    pnl = final_equity - INITIAL_CAPITAL
    ret_pct = pnl / INITIAL_CAPITAL * 100
    max_dd = ((df["Equity"].cummax() - df["Equity"]) / df["Equity"].cummax()).max() * 100
    n_trades = len(trades)
    wins = sum(1 for t in trades if t["pnl"] > 0)
    win_rate = wins / n_trades * 100 if n_trades > 0 else 0

    stats_text = (
        f"P&L: ${pnl:+,.0f} ({ret_pct:+.1f}%) | "
        f"Trades: {n_trades} | Win: {win_rate:.0f}% | "
        f"Max DD: {max_dd:.1f}%"
    )

    # --- Layout ---
    fig.update_layout(
        template="plotly_dark",
        title=dict(
            text=f"ES E-mini Futures — HA EMA Band + RSI Backtest<br>"
                 f"<span style='font-size:14px;color:{'#00ff88' if pnl > 0 else '#ff4444'}'>"
                 f"{stats_text}</span>",
            font=dict(size=18),
        ),
        height=1000,
        showlegend=True,
        legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center"),
        hovermode="x unified",
        xaxis4=dict(title="Date"),
    )

    # Dark theme tweaks
    for i in range(1, 5):
        fig.update_yaxes(gridcolor="rgba(50,50,50,0.5)", row=i, col=1)
        fig.update_xaxes(gridcolor="rgba(50,50,50,0.5)", row=i, col=1)

    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="$", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1, range=[0, 100])
    fig.update_yaxes(title_text="ATR", row=4, col=1)

    fig.write_html(output, auto_open=True)
    print(f"\nChart saved to {output} and opened in browser.")

    return fig


if __name__ == "__main__":
    print("Fetching data...")
    df = fetch_es_data()

    print("Computing indicators...")
    df = compute_all(df)

    print("Simulating trades...")
    df, trades = simulate_trades(df)

    print(f"Total trades: {len(trades)}")
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    print(f"Won: {len(wins)} | Lost: {len(losses)}")

    print("\nGenerating chart...")
    plot_results(df, trades)
