"""
Session analysis: split the trading day into 3 parts,
run the best combo (SL=0.25x TP=0.50x) on each, see where the edge is.
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

# Sessions (Eastern Time — matching yfinance output)
SESSIONS = [
    ("Overnight (00:00-04:00)", 0, 4, 0),     # True Globex overnight only
    ("US Morning (04:00-13:00)", 4, 13, 0),    # Pre-market + open + morning trend
    ("US Afternoon (13:00-17:00)", 13, 17, 0), # Afternoon + close
]

# Best combo from grid search
SL_MULT = 0.25
TP_MULT = 0.50


def simulate_session(df, atr_stop_mult, atr_target_mult, session_start_h, session_end_h, session_end_m=0, session_start_m=0):
    """Only take entries during the specified session window."""
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

        # Time filter — only enter during this session
        bar_time = df.index[i]
        bar_hour = bar_time.hour
        bar_min = bar_time.minute
        bar_minutes = bar_hour * 60 + bar_min
        session_start_minutes = session_start_h * 60 + session_start_m
        session_end_minutes = session_end_h * 60 + session_end_m
        in_session = session_start_minutes <= bar_minutes < session_end_minutes

        if np.isnan(rsi) or np.isnan(ema_h) or np.isnan(atr):
            equities.append(equity)
            signals.append(0)
            continue

        # Stop loss (always active, even outside session)
        if position == 1 and close <= stop_price:
            equity += (close - entry_price) * ES_POINT_VALUE
            trades.append({"pnl": (close - entry_price) * ES_POINT_VALUE, "hour": bar_hour})
            position = 0; signal = -1
        elif position == -1 and close >= stop_price:
            equity += (entry_price - close) * ES_POINT_VALUE
            trades.append({"pnl": (entry_price - close) * ES_POINT_VALUE, "hour": bar_hour})
            position = 0; signal = 1

        # Take profit (always active)
        if position == 1 and close >= target_price:
            equity += (close - entry_price) * ES_POINT_VALUE
            trades.append({"pnl": (close - entry_price) * ES_POINT_VALUE, "hour": bar_hour})
            position = 0; signal = -1
        elif position == -1 and close <= target_price:
            equity += (entry_price - close) * ES_POINT_VALUE
            trades.append({"pnl": (entry_price - close) * ES_POINT_VALUE, "hour": bar_hour})
            position = 0; signal = 1

        # Reversal exit (always active)
        if position == 1:
            if ha_c < ema_l and rsi < RSI_SELL_THRESHOLD:
                equity += (close - entry_price) * ES_POINT_VALUE
                trades.append({"pnl": (close - entry_price) * ES_POINT_VALUE, "hour": bar_hour})
                position = 0; signal = -1
        elif position == -1:
            if ha_c > ema_h and rsi > RSI_BUY_THRESHOLD:
                equity += (entry_price - close) * ES_POINT_VALUE
                trades.append({"pnl": (entry_price - close) * ES_POINT_VALUE, "hour": bar_hour})
                position = 0; signal = 1

        # Entry — ONLY during session
        if position == 0 and in_session:
            if ha_c > ema_h and rsi > RSI_BUY_THRESHOLD:
                position = 1; entry_price = close
                stop_price = close - atr * atr_stop_mult
                target_price = close + atr * atr_target_mult
                signal = 1
            elif ha_c < ema_l and rsi < RSI_SELL_THRESHOLD:
                position = -1; entry_price = close
                stop_price = close + atr * atr_stop_mult
                target_price = close - atr * atr_target_mult
                signal = -1

        if position == 1:
            equities.append(equity + (close - entry_price) * ES_POINT_VALUE)
        elif position == -1:
            equities.append(equity + (entry_price - close) * ES_POINT_VALUE)
        else:
            equities.append(equity)

        signals.append(signal)

    return equities, trades, signals


def simulate_all(df, atr_stop_mult, atr_target_mult):
    """No session filter — trades any time."""
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

        bar_hour = df.index[i].hour

        if position == 1 and close <= stop_price:
            equity += (close - entry_price) * ES_POINT_VALUE
            trades.append({"pnl": (close - entry_price) * ES_POINT_VALUE, "hour": bar_hour})
            position = 0; signal = -1
        elif position == -1 and close >= stop_price:
            equity += (entry_price - close) * ES_POINT_VALUE
            trades.append({"pnl": (entry_price - close) * ES_POINT_VALUE, "hour": bar_hour})
            position = 0; signal = 1

        if position == 1 and close >= target_price:
            equity += (close - entry_price) * ES_POINT_VALUE
            trades.append({"pnl": (close - entry_price) * ES_POINT_VALUE, "hour": bar_hour})
            position = 0; signal = -1
        elif position == -1 and close <= target_price:
            equity += (entry_price - close) * ES_POINT_VALUE
            trades.append({"pnl": (entry_price - close) * ES_POINT_VALUE, "hour": bar_hour})
            position = 0; signal = 1

        if position == 1:
            if ha_c < ema_l and rsi < RSI_SELL_THRESHOLD:
                equity += (close - entry_price) * ES_POINT_VALUE
                trades.append({"pnl": (close - entry_price) * ES_POINT_VALUE, "hour": bar_hour})
                position = 0; signal = -1
        elif position == -1:
            if ha_c > ema_h and rsi > RSI_BUY_THRESHOLD:
                equity += (entry_price - close) * ES_POINT_VALUE
                trades.append({"pnl": (entry_price - close) * ES_POINT_VALUE, "hour": bar_hour})
                position = 0; signal = 1

        if position == 0:
            if ha_c > ema_h and rsi > RSI_BUY_THRESHOLD:
                position = 1; entry_price = close
                stop_price = close - atr * atr_stop_mult
                target_price = close + atr * atr_target_mult
                signal = 1
            elif ha_c < ema_l and rsi < RSI_SELL_THRESHOLD:
                position = -1; entry_price = close
                stop_price = close + atr * atr_stop_mult
                target_price = close - atr * atr_target_mult
                signal = -1

        if position == 1:
            equities.append(equity + (close - entry_price) * ES_POINT_VALUE)
        elif position == -1:
            equities.append(equity + (entry_price - close) * ES_POINT_VALUE)
        else:
            equities.append(equity)

        signals.append(signal)

    return equities, trades, signals


def stats(trades, equities):
    n = len(trades)
    if n == 0:
        return dict(pnl=0, ret=0, max_dd=0, trades=0, wins=0, wr=0, pf=0)
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    pnl = equities[-1] - INITIAL_CAPITAL
    eq = np.array(equities)
    dd = ((np.maximum.accumulate(eq) - eq) / np.maximum.accumulate(eq) * 100)
    gross_w = abs(sum(t["pnl"] for t in wins)) if wins else 0
    gross_l = abs(sum(t["pnl"] for t in losses)) if losses else 0.001
    return dict(
        pnl=pnl, ret=pnl / INITIAL_CAPITAL * 100, max_dd=dd.max(),
        trades=n, wins=len(wins), wr=len(wins) / n * 100,
        pf=gross_w / gross_l,
    )


def main():
    print("Fetching data...")
    df = fetch_es_data()
    df = compute_all(df)
    print(f"Using best combo: SL={SL_MULT}x ATR, TP={TP_MULT}x ATR\n")

    # ── Hourly P&L breakdown (no session filter) ──
    eq_all, trades_all, sig_all = simulate_all(df, SL_MULT, TP_MULT)
    s_all = stats(trades_all, eq_all)

    hourly = {}
    for t in trades_all:
        h = t["hour"]
        if h not in hourly:
            hourly[h] = {"pnl": 0, "trades": 0, "wins": 0}
        hourly[h]["pnl"] += t["pnl"]
        hourly[h]["trades"] += 1
        if t["pnl"] > 0:
            hourly[h]["wins"] += 1

    print("=" * 65)
    print("  HOURLY P&L BREAKDOWN (SL=0.25x TP=0.50x)")
    print("=" * 65)
    print(f"  {'Hour':>6s}  {'P&L':>10s}  {'Trades':>7s}  {'Win%':>6s}  {'Avg':>8s}  Bar")
    print("-" * 65)
    for h in sorted(hourly.keys()):
        d = hourly[h]
        wr = d["wins"] / d["trades"] * 100 if d["trades"] > 0 else 0
        avg = d["pnl"] / d["trades"] if d["trades"] > 0 else 0
        bar = "+" * max(1, int(abs(d["pnl"]) / 500)) if d["pnl"] > 0 else "-" * max(1, int(abs(d["pnl"]) / 500))
        print(f"  {h:02d}:00   ${d['pnl']:>+9,.0f}  {d['trades']:>6d}   {wr:>5.0f}%  ${avg:>+7.0f}  {bar}")
    print("-" * 65)
    print(f"  TOTAL  ${s_all['pnl']:>+9,.0f}  {s_all['trades']:>6d}   {s_all['wr']:>5.0f}%          PF={s_all['pf']:.2f}")

    # ── Session backtests ──
    print(f"\n{'=' * 65}")
    print("  SESSION BACKTESTS")
    print("=" * 65)

    variants = [("All Sessions", eq_all, trades_all, sig_all, s_all)]

    for name, start_h, end_h, end_m in SESSIONS:
        start_m = 30 if start_h == 9 else 0
        eq, tr, sig = simulate_session(df, SL_MULT, TP_MULT, start_h, end_h, end_m, start_m)
        s = stats(tr, eq)
        variants.append((name, eq, tr, sig, s))
        print(f"  {name:35s} P&L ${s['pnl']:>+9,.0f} | {s['trades']:>4d} trades | Win {s['wr']:.0f}% | DD {s['max_dd']:.1f}% | PF {s['pf']:.2f}")

    # ── Visual ──
    colors = ["#ffffff", "#4dabf7", "#00ff88", "#ff922b"]

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.40, 0.35, 0.25],
        subplot_titles=(
            "ES 5m — Price + Signals",
            "Equity Curves by Session",
            "Hourly P&L Heatmap",
        )
    )

    n_price_traces = 6  # EMA high, low, HA close, close, buy, sell markers
    n_equity_traces = 1
    n_per_variant = n_price_traces + n_equity_traces  # 7

    for vi, (name, eq, tr, sig, s) in enumerate(variants):
        buy_idx = [i for i in range(len(df)) if sig[i] == 1]
        sell_idx = [i for i in range(len(df)) if sig[i] == -1]
        visible = (vi == 0)
        color = colors[vi]

        # Price panel traces
        fig.add_trace(go.Scatter(x=df.index, y=df["EMA_High"], line=dict(color="rgba(0,150,255,0.4)", width=1), showlegend=False, visible=visible), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["EMA_Low"], line=dict(color="rgba(0,150,255,0.4)", width=1), fill="tonexty", fillcolor="rgba(0,150,255,0.07)", showlegend=False, visible=visible), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["HA_Close"], line=dict(color="white", width=1.5), showlegend=False, visible=visible), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["Close"], line=dict(color="rgba(180,180,180,0.4)", width=1), showlegend=False, visible=visible), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index[buy_idx], y=df["Close"].iloc[buy_idx], mode="markers", marker=dict(symbol="triangle-up", size=9, color="#00ff88", line=dict(width=1, color="white")), showlegend=False, visible=visible), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index[sell_idx], y=df["Close"].iloc[sell_idx], mode="markers", marker=dict(symbol="triangle-down", size=9, color="#ff4444", line=dict(width=1, color="white")), showlegend=False, visible=visible), row=1, col=1)

        # Equity
        r, g, b = int(color.lstrip("#")[0:2], 16), int(color.lstrip("#")[2:4], 16), int(color.lstrip("#")[4:6], 16)
        fig.add_trace(go.Scatter(x=df.index, y=eq, name=name, line=dict(color=color, width=2), fill="tozeroy", fillcolor=f"rgba({r},{g},{b},0.08)", visible=visible), row=2, col=1)

    # Hourly P&L bar chart (always visible)
    hours_sorted = sorted(hourly.keys())
    pnls = [hourly[h]["pnl"] for h in hours_sorted]
    bar_colors = ["#00ff88" if p > 0 else "#ff4444" for p in pnls]
    fig.add_trace(go.Bar(
        x=[f"{h:02d}:00" for h in hours_sorted],
        y=pnls,
        marker_color=bar_colors,
        name="Hourly P&L",
        showlegend=False,
    ), row=3, col=1)

    # Dropdown
    buttons = []
    for vi, (name, eq, tr, sig, s) in enumerate(variants):
        vis = [False] * (n_per_variant * len(variants))
        for j in range(n_per_variant):
            vis[vi * n_per_variant + j] = True
        # Hourly bar always visible (it's the last trace)
        total_session_traces = n_per_variant * len(variants)
        vis.append(True)

        btn_label = (f"{name} | ${s['pnl']:+,.0f} | "
                     f"{s['trades']} tr | W {s['wr']:.0f}% | "
                     f"DD {s['max_dd']:.1f}% | PF {s['pf']:.2f}")
        buttons.append(dict(label=btn_label, method="update", args=[{"visible": vis}]))

    fig.update_layout(
        updatemenus=[dict(
            type="dropdown", direction="down",
            x=0.5, xanchor="center", y=1.06,
            buttons=buttons,
            font=dict(size=11),
            bgcolor="rgba(30,30,30,0.9)",
        )],
        template="plotly_dark",
        title=dict(text="ES 5m Session Analysis — SL=0.25x TP=0.50x ATR", font=dict(size=18)),
        height=1100,
        hovermode="x unified",
    )

    fig.add_hline(y=INITIAL_CAPITAL, line_dash="dash", line_color="gray",
                  annotation_text=f"Start: ${INITIAL_CAPITAL:,.0f}", row=2, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=3, col=1)

    for i in range(1, 4):
        fig.update_yaxes(gridcolor="rgba(50,50,50,0.5)", row=i, col=1)
        fig.update_xaxes(gridcolor="rgba(50,50,50,0.5)", row=i, col=1)

    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Equity ($)", row=2, col=1)
    fig.update_yaxes(title_text="P&L ($)", row=3, col=1)

    output = "session_chart.html"
    fig.write_html(output, auto_open=True)
    print(f"\nChart saved to {output}")


if __name__ == "__main__":
    main()
