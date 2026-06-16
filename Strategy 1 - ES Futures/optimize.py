"""
Grid search over ATR target multiplier + ATR stop multiplier.
Ranks by Profit Factor, Net P&L, and Sharpe Ratio.
"""

import numpy as np
import pandas as pd
from data_feed import fetch_es_data
from indicators import compute_all
from config import (
    RSI_BUY_THRESHOLD, RSI_SELL_THRESHOLD,
    INITIAL_CAPITAL, ES_POINT_VALUE,
)

# ── Grid parameters ──
TARGET_RANGE = [None, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
STOP_RANGE = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]


def simulate(df, atr_stop_mult, atr_target_mult):
    """Run one backtest pass. Returns (equities, trades)."""
    position = 0
    entry_price = stop_price = target_price = 0.0
    equity = INITIAL_CAPITAL

    trades = []
    equities = [equity]

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

        # Stop loss
        if position == 1 and close <= stop_price:
            pnl = (close - entry_price) * ES_POINT_VALUE
            equity += pnl
            trades.append(pnl)
            position = 0
        elif position == -1 and close >= stop_price:
            pnl = (entry_price - close) * ES_POINT_VALUE
            equity += pnl
            trades.append(pnl)
            position = 0

        # Take profit
        if atr_target_mult is not None:
            if position == 1 and close >= target_price:
                pnl = (close - entry_price) * ES_POINT_VALUE
                equity += pnl
                trades.append(pnl)
                position = 0
            elif position == -1 and close <= target_price:
                pnl = (entry_price - close) * ES_POINT_VALUE
                equity += pnl
                trades.append(pnl)
                position = 0

        # Entry / reversal exit
        if position == 0:
            if ha_c > ema_h and rsi > RSI_BUY_THRESHOLD:
                position = 1
                entry_price = close
                stop_price = close - atr * atr_stop_mult
                target_price = close + atr * atr_target_mult if atr_target_mult else 0
            elif ha_c < ema_l and rsi < RSI_SELL_THRESHOLD:
                position = -1
                entry_price = close
                stop_price = close + atr * atr_stop_mult
                target_price = close - atr * atr_target_mult if atr_target_mult else 0
        elif position == 1:
            if ha_c < ema_l and rsi < RSI_SELL_THRESHOLD:
                pnl = (close - entry_price) * ES_POINT_VALUE
                equity += pnl
                trades.append(pnl)
                position = 0
        elif position == -1:
            if ha_c > ema_h and rsi > RSI_BUY_THRESHOLD:
                pnl = (entry_price - close) * ES_POINT_VALUE
                equity += pnl
                trades.append(pnl)
                position = 0

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
    gross_win = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0.001
    pnl = equities[-1] - INITIAL_CAPITAL

    eq = np.array(equities)
    peak = np.maximum.accumulate(eq)
    dd_pct = ((peak - eq) / peak * 100).max()

    # Sharpe on per-trade returns
    if len(trades) > 1:
        ret = np.array(trades) / INITIAL_CAPITAL
        sharpe = np.mean(ret) / np.std(ret) * np.sqrt(252 * 78)  # ~78 5-min bars/day
    else:
        sharpe = 0

    return {
        "pnl": pnl,
        "trades": len(trades),
        "win_rate": len(wins) / len(trades) * 100,
        "pf": gross_win / gross_loss,
        "max_dd": dd_pct,
        "sharpe": sharpe,
        "avg_win": np.mean(wins) if wins else 0,
        "avg_loss": np.mean(losses) if losses else 0,
    }


def main():
    print("Fetching 5m ES data...")
    df = fetch_es_data()
    df = compute_all(df)
    print(f"Running grid: {len(TARGET_RANGE)} targets x {len(STOP_RANGE)} stops = {len(TARGET_RANGE) * len(STOP_RANGE)} combos\n")

    results = []

    for stop in STOP_RANGE:
        for target in TARGET_RANGE:
            eq, trades = simulate(df, stop, target)
            s = compute_stats(trades, eq)
            if s is None:
                continue
            label_t = "None" if target is None else f"{target:.2f}"
            results.append({
                "Stop": f"{stop:.2f}x",
                "Target": label_t,
                "P&L": s["pnl"],
                "Trades": s["trades"],
                "Win%": s["win_rate"],
                "PF": s["pf"],
                "MaxDD%": s["max_dd"],
                "Sharpe": s["sharpe"],
                "AvgWin": s["avg_win"],
                "AvgLoss": s["avg_loss"],
            })

    tbl = pd.DataFrame(results)

    # Rank tables
    print("=" * 90)
    print("  TOP 15 BY PROFIT FACTOR")
    print("=" * 90)
    by_pf = tbl.sort_values("PF", ascending=False).head(15)
    print(by_pf.to_string(index=False, float_format="%.2f"))

    print("\n" + "=" * 90)
    print("  TOP 15 BY NET P&L")
    print("=" * 90)
    by_pnl = tbl.sort_values("P&L", ascending=False).head(15)
    print(by_pnl.to_string(index=False, float_format="%.2f"))

    print("\n" + "=" * 90)
    print("  TOP 15 BY SHARPE RATIO")
    print("=" * 90)
    by_sharpe = tbl.sort_values("Sharpe", ascending=False).head(15)
    print(by_sharpe.to_string(index=False, float_format="%.2f"))

    # Overall best (composite rank)
    tbl["rank_pf"] = tbl["PF"].rank(ascending=False)
    tbl["rank_pnl"] = tbl["P&L"].rank(ascending=False)
    tbl["rank_sharpe"] = tbl["Sharpe"].rank(ascending=False)
    tbl["composite"] = tbl["rank_pf"] + tbl["rank_pnl"] + tbl["rank_sharpe"]

    print("\n" + "=" * 90)
    print("  TOP 10 COMPOSITE (avg rank across all 3 metrics)")
    print("=" * 90)
    by_comp = tbl.sort_values("composite").head(10)
    print(by_comp[["Stop", "Target", "P&L", "Trades", "Win%", "PF", "MaxDD%", "Sharpe"]].to_string(index=False, float_format="%.2f"))

    print()


if __name__ == "__main__":
    main()
