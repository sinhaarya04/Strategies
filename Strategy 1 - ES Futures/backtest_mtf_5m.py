"""
Strategy: Multi-Timeframe Filter (60m + 5m)
60-min chart sets direction bias (HA EMA + RSI).
5-min chart entries only when aligned with 60-min bias.
All P&L reported in POINTS.
"""

import numpy as np
import pandas as pd

from data_feed import fetch_es_data
from indicators import compute_all
from config import RSI_BUY_THRESHOLD, RSI_SELL_THRESHOLD, ES_POINT_VALUE

ACCOUNT_SIZE = 50_000.0

STOP_RANGE = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
TARGET_RANGE = [None, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]

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
    For each 60-min bar, determine directional bias:
      +1 = buy bias (HA close > EMA high AND RSI > buy threshold)
      -1 = sell bias (HA close < EMA low AND RSI < sell threshold)
       0 = neutral
    """
    bias = pd.Series(0, index=df_60m.index, dtype=int)
    for i in range(len(df_60m)):
        ha_c = df_60m["HA_Close"].iloc[i]
        ema_h = df_60m["EMA_High"].iloc[i]
        ema_l = df_60m["EMA_Low"].iloc[i]
        rsi = df_60m["RSI"].iloc[i]

        if np.isnan(rsi) or np.isnan(ema_h):
            continue

        if ha_c > ema_h and rsi > RSI_BUY_THRESHOLD:
            bias.iloc[i] = 1
        elif ha_c < ema_l and rsi < RSI_SELL_THRESHOLD:
            bias.iloc[i] = -1
    return bias


def align_bias_to_5m(bias_60m, df_5m):
    """
    Map 60-min bias to each 5-min bar.
    Each 5-min bar gets the bias of the most recent completed 60-min bar.
    """
    aligned = pd.Series(0, index=df_5m.index, dtype=int)
    bias_times = bias_60m.index
    for i, ts in enumerate(df_5m.index):
        mask = bias_times <= ts
        if mask.any():
            aligned.iloc[i] = bias_60m.loc[bias_times[mask][-1]]
    return aligned


def simulate(df_5m, bias_5m, atr_stop_mult, atr_target_mult, s_start_h, s_start_m, s_end_h, s_end_m):
    """Run strategy: only enter when 5m signal aligns with 60m bias. P&L in points."""
    position = 0
    entry_price = stop_price = target_price = 0.0
    equity_pts = 0.0

    trades = []
    equities = [equity_pts]

    start_min = s_start_h * 60 + s_start_m
    end_min = s_end_h * 60 + s_end_m
    filter_all = (start_min == 0 and end_min == 23 * 60 + 59)

    for i in range(1, len(df_5m)):
        ha_c = df_5m["HA_Close"].iloc[i]
        ema_h = df_5m["EMA_High"].iloc[i]
        ema_l = df_5m["EMA_Low"].iloc[i]
        rsi = df_5m["RSI"].iloc[i]
        atr = df_5m["ATR"].iloc[i]
        close = df_5m["Close"].iloc[i]
        bias = bias_5m.iloc[i]

        if np.isnan(rsi) or np.isnan(ema_h) or np.isnan(atr):
            equities.append(equity_pts)
            continue

        bar_min = df_5m.index[i].hour * 60 + df_5m.index[i].minute
        in_session = filter_all or (start_min <= bar_min < end_min)

        # Exits always active
        if position == 1 and close <= stop_price:
            equity_pts += close - entry_price
            trades.append(close - entry_price)
            position = 0
        elif position == -1 and close >= stop_price:
            equity_pts += entry_price - close
            trades.append(entry_price - close)
            position = 0

        if atr_target_mult is not None:
            if position == 1 and close >= target_price:
                equity_pts += close - entry_price
                trades.append(close - entry_price)
                position = 0
            elif position == -1 and close <= target_price:
                equity_pts += entry_price - close
                trades.append(entry_price - close)
                position = 0

        # Reversal exit
        if position == 1:
            if ha_c < ema_l and rsi < RSI_SELL_THRESHOLD:
                equity_pts += close - entry_price
                trades.append(close - entry_price)
                position = 0
        elif position == -1:
            if ha_c > ema_h and rsi > RSI_BUY_THRESHOLD:
                equity_pts += entry_price - close
                trades.append(entry_price - close)
                position = 0

        # Entry: 5m signal must ALIGN with 60m bias, and be in session
        if position == 0 and in_session:
            if bias == 1 and ha_c > ema_h and rsi > RSI_BUY_THRESHOLD:
                position = 1; entry_price = close
                stop_price = close - atr * atr_stop_mult
                target_price = close + atr * atr_target_mult if atr_target_mult else 0
            elif bias == -1 and ha_c < ema_l and rsi < RSI_SELL_THRESHOLD:
                position = -1; entry_price = close
                stop_price = close + atr * atr_stop_mult
                target_price = close - atr * atr_target_mult if atr_target_mult else 0

        if position == 1:
            equities.append(equity_pts + (close - entry_price))
        elif position == -1:
            equities.append(equity_pts + (entry_price - close))
        else:
            equities.append(equity_pts)

    return equities, trades


def compute_stats(trades, equities):
    if not trades:
        return None
    wins = [t for t in trades if t > 0]
    losses = [t for t in trades if t <= 0]
    gross_w = sum(wins) if wins else 0
    gross_l = abs(sum(losses)) if losses else 0.001
    eq = np.array(equities)
    peak = np.maximum.accumulate(eq)
    dd_pts = (peak - eq).max()
    return {
        "pts": equities[-1],
        "trades": len(trades),
        "wr": len(wins) / len(trades) * 100,
        "pf": gross_w / gross_l,
        "dd_pts": dd_pts,
        "avg_w": np.mean(wins) if wins else 0,
        "avg_l": np.mean(losses) if losses else 0,
    }


def main():
    print("Fetching 5m data, resampling to 60m for bias...")
    df_5m_raw = fetch_es_data()
    df_60m_raw = resample(df_5m_raw, "60min")

    df_5m = compute_all(df_5m_raw)
    df_60m = compute_all(df_60m_raw)

    print(f"5m bars: {len(df_5m)} | 60m bars: {len(df_60m)}")
    print(f"RSI thresholds: Buy > {RSI_BUY_THRESHOLD}, Sell < {RSI_SELL_THRESHOLD}")

    # Compute 60m bias and map to 5m
    bias_60m = get_60m_bias(df_60m)
    bias_5m = align_bias_to_5m(bias_60m, df_5m)

    buy_bars = (bias_5m == 1).sum()
    sell_bars = (bias_5m == -1).sum()
    neutral_bars = (bias_5m == 0).sum()
    print(f"60m bias distribution on 5m bars: BUY={buy_bars} SELL={sell_bars} NEUTRAL={neutral_bars}")

    total = len(STOP_RANGE) * len(TARGET_RANGE) * len(SESSIONS)
    print(f"Running {total} combos...\n")

    all_results = []

    for sess_name, sh, sm, eh, em in SESSIONS:
        for stop in STOP_RANGE:
            for target in TARGET_RANGE:
                eq, tr = simulate(df_5m, bias_5m, stop, target, sh, sm, eh, em)
                s = compute_stats(tr, eq)
                if s is None:
                    continue
                t_label = "None" if target is None else f"{target:.2f}x"
                dollar_pnl = s["pts"] * ES_POINT_VALUE
                dollar_dd = s["dd_pts"] * ES_POINT_VALUE
                ret_pct = dollar_pnl / ACCOUNT_SIZE * 100
                all_results.append({
                    "Session": sess_name,
                    "Stop": f"{stop:.2f}x",
                    "Target": t_label,
                    "Points": s["pts"],
                    "$P&L(50k)": dollar_pnl,
                    "Return%": ret_pct,
                    "Trades": s["trades"],
                    "Win%": s["wr"],
                    "PF": s["pf"],
                    "MaxDD(pts)": s["dd_pts"],
                    "$MaxDD": dollar_dd,
                    "AvgW(pts)": s["avg_w"],
                    "AvgL(pts)": s["avg_l"],
                })

    tbl = pd.DataFrame(all_results)
    profitable = tbl[tbl["PF"] >= 1.0].sort_values("PF", ascending=False)

    if len(profitable) > 0:
        print("=" * 110)
        print(f"  MTF 60m+5m PROFITABLE COMBOS (PF >= 1.0) — {len(profitable)} found!  [POINTS]")
        print("=" * 110)
        print(profitable.head(30).to_string(index=False, float_format="%.2f"))
    else:
        print("No profitable combos found.")

    for sess_name, _, _, _, _ in SESSIONS:
        subset = tbl[tbl["Session"] == sess_name].sort_values("PF", ascending=False).head(15)
        print(f"\n{'=' * 110}")
        print(f"  TOP 15: {sess_name}  [POINTS]")
        print("=" * 110)
        print(subset.to_string(index=False, float_format="%.2f"))

    # HTML output
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
            "Points": "{:+,.1f} pts",
            "$P&L(50k)": "${:+,.0f}",
            "Return%": "{:+.1f}%",
            "Trades": "{:,.0f}",
            "Win%": "{:.1f}%",
            "PF": "{:.3f}",
            "MaxDD(pts)": "{:.1f} pts",
            "$MaxDD": "${:,.0f}",
            "AvgW(pts)": "{:+.2f} pts",
            "AvgL(pts)": "{:+.2f} pts",
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

    best = tbl_sorted.iloc[0]
    html = f"""<!DOCTYPE html>
<html><head><title>ES MTF (60m+5m) Grid — {len(tbl)} combos</title>
<style>
  body {{ background: #0d1117; color: #e0e0e0; font-family: monospace; padding: 20px; }}
  h1 {{ color: #58a6ff; font-size: 20px; }}
  h2 {{ color: #8b949e; font-size: 16px; margin-top: 5px; }}
  .summary {{ display: flex; gap: 20px; margin: 15px 0; flex-wrap: wrap; }}
  .card {{ background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 12px 16px; }}
  .card .num {{ font-size: 24px; font-weight: bold; }}
  .green {{ color: #3fb950; }} .red {{ color: #f85149; }} .yellow {{ color: #d29922; }}
</style></head><body>
<h1>Strategy: Multi-Timeframe — 60m Bias + 5m Entries</h1>
<h2>60m sets direction, 5m aligned entries only | RSI {RSI_BUY_THRESHOLD}/{RSI_SELL_THRESHOLD} | $50k account @ $50/pt</h2>
<h2>{len(tbl)} combos | 5m bars: {len(df_5m)} | 60m bias on 5m: BUY={buy_bars} SELL={sell_bars} NEUTRAL={neutral_bars}</h2>
<div class="summary">
  <div class="card"><div class="green num">{len(profitable)}</div><div>Profitable (PF &ge; 1.0)</div></div>
  <div class="card"><div class="yellow num">{len(tbl[(tbl['PF'] >= 0.95) & (tbl['PF'] < 1.0)])}</div><div>Near breakeven</div></div>
  <div class="card"><div class="red num">{len(tbl[tbl['PF'] < 0.95])}</div><div>Losing (PF &lt; 0.95)</div></div>
  <div class="card"><div class="green num">{best['Points']:+,.1f} pts</div><div>Best: {best['Session']} SL={best['Stop']} TP={best['Target']}</div></div>
  <div class="card"><div class="green num">${best['$P&L(50k)']:+,.0f}</div><div>Best $ P&amp;L (50k account)</div></div>
  <div class="card"><div class="{'green' if best['Return%'] > 0 else 'red'} num">{best['Return%']:+.1f}%</div><div>Best Return on $50k</div></div>
</div>
{styled.to_html()}
</body></html>"""

    output = "backtest_mtf_5m_chart.html"
    with open(output, "w") as f:
        f.write(html)

    import webbrowser, os
    webbrowser.open("file://" + os.path.abspath(output))
    print(f"\nChart saved to {output} and opened in browser.")


if __name__ == "__main__":
    main()
