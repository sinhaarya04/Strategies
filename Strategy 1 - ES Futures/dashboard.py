"""
Generate the ES Futures strategy picker dashboard.
Run: python dashboard.py
Opens dashboard.html in browser — click a strategy card to view its chart.
"""

import webbrowser
import os

STRATEGIES = [
    {
        "id": "base5m",
        "name": "Base 5-min",
        "tag": "5m",
        "file": "backtest_chart.html",
        "summary": "Single-pass HA EMA + RSI on 5-min bars with ATR stop/target.",
        "entry": "HA close crosses 20-EMA(High) up AND RSI > 55 → LONG. HA close crosses 20-EMA(Low) down AND RSI < 45 → SHORT.",
        "exit": "ATR stop (1.5×), ATR target (1.0×), or reversal signal closes the position.",
        "position": "1 contract. Size scaled to risk 2% of capital using ATR distance.",
        "sweep": "Single backtest run (no grid). 5-min bars across full ~1 year of SPY data.",
        "notes": "Baseline reference — equity curve, trade markers, drawdown. Use as the apples-to-apples comparison point.",
    },
    {
        "id": "grid5m",
        "name": "5-min Grid Search",
        "tag": "5m",
        "file": "compare_chart.html",
        "summary": "110-combo grid sweep of ATR stop × target multipliers on the 5-min strategy.",
        "entry": "Same HA EMA + RSI signal as Base 5-min.",
        "exit": "Each combo uses a different (stop ATR ×, target ATR ×) pair. Reversal exit still active.",
        "position": "1 contract per trade, ATR-sized.",
        "sweep": "10 stops (0.25× → 3.0×) × 11 targets (None + 0.25× → 3.0×) = 110 combos, ranked by composite score.",
        "notes": "Interactive tabbed Plotly chart with price / equity / RSI / ATR panels. Dropdown switches between combos.",
    },
    {
        "id": "session5m",
        "name": "5-min Session Analysis",
        "tag": "5m",
        "file": "session_chart.html",
        "summary": "Best-of-grid combo (SL 0.25×, TP 0.50×) sliced by intraday session.",
        "entry": "Same HA EMA + RSI signal, only when current bar falls in the chosen session window.",
        "exit": "Tight ATR stop (0.25×) and modest target (0.50×) — winning combo from the 5m grid.",
        "position": "1 contract per trade, ATR-sized.",
        "sweep": "Same combo run across 3 sessions: Overnight, US Morning, US Afternoon. Hourly P&L heatmap included.",
        "notes": "Identifies which hours-of-day the edge concentrates in. US Morning carries the bulk of the P&L.",
    },
    {
        "id": "sessiongrid5m",
        "name": "5-min Session Grid",
        "tag": "5m",
        "file": "session_grid.html",
        "summary": "Full session × stop × target sweep — color-coded by Profit Factor.",
        "entry": "HA EMA + RSI, only inside the chosen session window.",
        "exit": "Each cell varies stop × target ATR multipliers.",
        "position": "1 contract per trade, ATR-sized.",
        "sweep": "10 stops × 11 targets × 4 sessions = 440 combos. P&L in dollars.",
        "notes": "Big heat-table — green = profitable, red = losing. Quickest way to see where the strategy is robust vs fragile.",
    },
    {
        "id": "base15m",
        "name": "15-min Base",
        "tag": "15m",
        "file": "backtest_15m_chart.html",
        "summary": "Same HA EMA + RSI logic resampled to 15-min bars.",
        "entry": "HA close vs 20-EMA(H/L) + RSI 55/45 threshold — but bars are 15-min.",
        "exit": "ATR stop × multiplier, ATR target × multiplier, or reversal.",
        "position": "1 contract per trade.",
        "sweep": "10 stops × 11 targets × 4 sessions = 440 combos. P&L in points (not dollars).",
        "notes": "Slower timeframe reduces noise. Best PF ≈ 1.18 in the US Morning session.",
    },
    {
        "id": "mtf",
        "name": "Multi-Timeframe (60m + 15m)",
        "tag": "MTF",
        "file": "backtest_mtf_chart.html",
        "summary": "Higher-timeframe bias filter — 60m sets direction, 15m takes entries.",
        "entry": "60m HA EMA + RSI assigns BUY / SELL / NEUTRAL bias. 15m entries fire only when aligned with that bias.",
        "exit": "ATR stop / target / reversal — same as the 15m base.",
        "position": "1 contract per trade.",
        "sweep": "10 × 11 × 4 = 440 combos in points.",
        "notes": "Cuts counter-trend trades. Trade count drops but average quality improves.",
    },
    {
        "id": "mtf5m",
        "name": "Multi-Timeframe (60m + 5m)",
        "tag": "MTF",
        "file": "backtest_mtf_5m_chart.html",
        "summary": "60m bias filter combined with the faster 5m entry timeframe.",
        "entry": "60m HA EMA + RSI assigns bias. 5m entries fire only when aligned.",
        "exit": "ATR stop / target / reversal.",
        "position": "1 contract per trade.",
        "sweep": "10 × 11 × 4 = 440 combos in points.",
        "notes": "More trade signals than 60m+15m, but more noise too. Use the heatmap to find where the edge holds at this finer granularity.",
    },
    {
        "id": "3lot",
        "name": "3-Lot Scaled Exit",
        "tag": "3-Lot",
        "file": "backtest_3lot_chart.html",
        "summary": "Pyramided exit — enter 3 lots, scale out at 1× / 2× / 3× ATR.",
        "entry": "HA close crosses 20-EMA band (NO RSI filter). Enters 3 lots in the cross direction.",
        "exit": "Initial stop = 1× ATR. Lot 1 → TP at +1 ATR (then stops 2 & 3 → breakeven). Lot 2 → TP at +2 ATR. Lot 3 → TP at +3 ATR or stopped at breakeven.",
        "position": "3 contracts entered together.",
        "sweep": "15-min and 60-min × 4 sessions = 8 runs. P&L in points.",
        "notes": "First scaled-exit study. No oscillator filter — pure HA cross. TP1 gets hit often, BE stops protect the runners.",
    },
    {
        "id": "fisher",
        "name": "Fisher Transform (standalone)",
        "tag": "Fisher",
        "file": "backtest_fisher_chart.html",
        "summary": "Pure Fisher Transform crossover with fixed-point stops/targets.",
        "entry": "Fisher line crosses its 1-bar-lag signal (tested on regular Close AND Heikin Ashi Close).",
        "exit": "Linear point stops & targets (1–15 pts), not ATR.",
        "position": "1 contract per trade.",
        "sweep": "2 sources × 2 timeframes × 4 sessions × 10 stops × 11 targets = 1,760 combos.",
        "notes": "Standalone Fisher test — no trend filter. Mostly chops; clean trends are where it pays.",
    },
    {
        "id": "fisher_ema",
        "name": "HA EMA + Fisher (MTF)",
        "tag": "Fisher",
        "file": "backtest_fisher_ema_chart.html",
        "summary": "Fisher Transform replaces RSI in the HA EMA system, with 60m bias filter.",
        "entry": "HA close crosses 20-EMA band AND Fisher crosses its signal in the same direction. 60m bias must agree.",
        "exit": "Tested on both ATR-multiplier grid and fixed-point grid.",
        "position": "1 contract per trade.",
        "sweep": "Close + HA Close × 5m + 15m × 4 sessions × dual stop/target grids.",
        "notes": "Most heavily filtered variant — fewer entries, but each one has triple confirmation (HA cross + Fisher cross + 60m bias).",
    },
    {
        "id": "haRSI_3lot_mtf_15m",
        "name": "★★ HA+RSI 3-Lot MTF (60m+15m) — TOP",
        "tag": "3-Lot",
        "file": "backtest_haRSI_3lot_mtf_15m_chart.html",
        "summary": "BEST STRATEGY OVERALL. HA EMA + RSI signal with the 3-lot scaled exit, gated by 60m bias, fired on 15m bars. 15m noise reduction beats both 5m and standalone 60m.",
        "entry": "60m bias: HA close > 20-EMA(H) AND RSI > 55 = LONG; HA close < 20-EMA(L) AND RSI < 45 = SHORT. 15m entries fire when the same signal triggers on a 15m bar AND aligns with 60m bias.",
        "exit": "Stop = 1.5× ATR(15m) ≈ $88/lot in morning. TP1 = +1× ATR → stop on lots 2 & 3 → BE. TP2 = +2× ATR → trail stop on lot 3 → +1 ATR. TP3 = +3× ATR closes runner.",
        "position": "3 contracts entered together. Trade frequency: ~2/day in morning+afternoon combined.",
        "sweep": "1 variant × 4 sessions = 4 runs.",
        "notes": "★ [ES Bloomberg, 6.5mo, NEW SESSIONS 04:00-13:00]: All Day = 828 entries → +$215,084 = +430% on $50k, 67.7% WR, PF 1.36. US Morning (04:00-13:00): 531 entries → +$117,851 = +236%, 66.9% WR, PF 1.30. US Afternoon: +$53,980, PF 1.30. Overnight (00:00-04:00) NOW PROFITABLE: +$26,427, PF 1.28. Walk-fwd Q1-Q4 morning (04:00-13:00): 3/4 profitable, total +$117,851, avg PF 1.38 — Q4 went NEGATIVE (-$4,794, PF 0.96) once pre-market hours included.",
    },
    {
        "id": "haRSI_3lot_mtf_5m",
        "name": "★ HA+RSI 3-Lot MTF (60m+5m)",
        "tag": "3-Lot",
        "file": "backtest_haRSI_3lot_mtf_5m_chart.html",
        "summary": "Best strategy tested so far. Combines the HA EMA + RSI signal (proven from MTF 60m+5m) with the 3-lot scaled exit ladder. 60m bias filters out counter-trend entries.",
        "entry": "60m bias: HA close > 20-EMA(H) AND RSI > 55 = LONG; HA close < 20-EMA(L) AND RSI < 45 = SHORT. 5m entries fire when the same signal triggers on a 5m bar AND aligns with 60m bias.",
        "exit": "Stop = 1.5× ATR(5m). TP1 = +1× ATR → stop on lots 2 & 3 → BE. TP2 = +2× ATR → trail stop on lot 3 → +1 ATR. TP3 = +3× ATR closes runner.",
        "position": "3 contracts entered together. Avg stop ≈ $40-56/lot. Trade count is high — ~6 entries per day on All-Day variant.",
        "sweep": "1 variant × 4 sessions = 4 runs.",
        "notes": "★★ [ES Bloomberg, 6.5mo, NEW SESSIONS 04:00-13:00]: ALL-DAY: 1,907 entries → +$266,975 = +534% on $50k, 68.0% WR, PF 1.33. US MORNING (04:00-13:00): 936 entries → +$165,449 = +331%, 68.3% WR, PF 1.38 (top PF). Overnight (00:00-04:00): +$38,662 = +77%, PF 1.35 — now profitable on real Globex. US Afternoon weakest (PF 1.10).",
    },
    {
        "id": "fisher_3lot_5m",
        "name": "Fisher 3-Lot — Standalone 5m",
        "tag": "Fisher",
        "file": "backtest_fisher_3lot_5min_chart.html",
        "summary": "Fisher 3-Lot scaled exit run on pure 5-min bars. No MTF gate, no higher-timeframe filter — entries fire on 5m FT crosses alone.",
        "entry": "5m FT crosses its 20-SMA. Tested with FT(Close) and FT(HA Close). Variants V3/V4 add an RSI > 50 / RSI < 50 direction filter.",
        "exit": "Stop = 1.5× ATR(5m). TP1/2/3 = 1×/2×/3× ATR(5m). Stop ladder: TP1 → BE, TP2 → trail to +1 ATR.",
        "position": "3 contracts. Avg stop ≈ $40/lot (very tight — small 5m ATR). Trade count is huge: 500-2,800 per session.",
        "sweep": "4 variants × 4 sessions = 16 runs.",
        "notes": "Best run: V4 FT(HA)+RSI, US Afternoon → +41.8 pts = +$2,089 (+4.2% on $50k), 501 entries, PF 1.09 (marginal). Only 4 of 16 runs profitable. 5m noise dominates — confirms slower TFs are where the edge lives.",
    },
    {
        "id": "fisher_3lot_15m",
        "name": "Fisher 3-Lot — Standalone 15m",
        "tag": "Fisher",
        "file": "backtest_fisher_3lot_15min_chart.html",
        "summary": "Fisher 3-Lot scaled exit run on pure 15-min bars. No MTF gate — entries fire on 15m FT crosses alone.",
        "entry": "15m FT crosses its 20-SMA. FT(Close) and FT(HA Close) sources. V3/V4 add RSI > 50 / RSI < 50 filter.",
        "exit": "Stop = 1.5× ATR(15m). TP1/2/3 = 1×/2×/3× ATR(15m). Stop ladder: TP1 → BE, TP2 → trail to +1 ATR.",
        "position": "3 contracts. Avg stop ≈ $80/lot. Trade count moderate: 150-1,000 per session.",
        "sweep": "4 variants × 4 sessions = 16 runs.",
        "notes": "Best run: V2 FT(HA) no-RSI, US Afternoon → +135.8 pts = +$6,791 (+13.6% on $50k), 283 entries, 67.4% WR, PF 1.29. 9 of 16 runs profitable. Cleaner than 5m but still much smaller edge than standalone 60m.",
    },
    {
        "id": "fisher_3lot_60m",
        "name": "Fisher 3-Lot — Standalone 60m ★",
        "tag": "Fisher",
        "file": "backtest_fisher_3lot_60min_chart.html",
        "summary": "Fisher 3-Lot scaled exit run on pure 60-min bars. No MTF gate. The slowest entry timeframe — and the clear winner of every test so far.",
        "entry": "60m FT crosses its 20-SMA. FT(Close) and FT(HA Close) sources. V3/V4 add RSI > 50 / RSI < 50 filter.",
        "exit": "Stop = 1.5× ATR(60m). TP1/2/3 = 1×/2×/3× ATR(60m). Stop ladder: TP1 → BE, TP2 → trail to +1 ATR.",
        "position": "3 contracts. Avg stop ≈ $150/lot. Trade count low: 35-280 per session — most selective.",
        "sweep": "4 variants × 4 sessions = 16 runs.",
        "notes": "★ TOP OVERALL: V2 FT(HA) no-RSI, US Morning → +166.5 pts = +$8,326 (+16.7% on $50k), 76 entries, 69.6% WR, PF 1.92. Highest PF: V3 FT(Close)+RSI, US Morning = PF 2.12. Worst single 3-lot blowup: -$784. RSI filter lifts PF in mornings, hurts afternoons.",
    },
    {
        "id": "fisher_3lot_combined",
        "name": "Fisher 3-Lot — All Timeframes Combined",
        "tag": "Fisher",
        "file": "backtest_fisher_3lot_chart.html",
        "summary": "Master combined view of all 48 standalone runs (5m, 15m, 60m × 4 variants × 4 sessions).",
        "entry": "Same FT cross + optional RSI filter logic across all three timeframes.",
        "exit": "Same 1.5×ATR stop / 1-2-3×ATR target ladder across all timeframes.",
        "position": "3 contracts per entry.",
        "sweep": "4 variants × 3 timeframes × 4 sessions = 48 runs.",
        "notes": "Use this card to compare timeframes side-by-side in one table. Edge concentrates at 60m: 6 of top 7 runs by PF are 60m, with V2 FT(HA) no-RSI, 60m, US Morning leading at +$8,326 / PF 1.92.",
    },
    {
        "id": "fisher_3lot_mtf_15m",
        "name": "Fisher 3-Lot MTF (60m bias + 15m entries)",
        "tag": "Fisher",
        "file": "backtest_fisher_3lot_mtf_15m_chart.html",
        "summary": "Multi-timeframe Fisher 3-Lot. 60m FT vs 20-SMA sets directional bias; 15m FT crosses fire entries only when aligned with that bias.",
        "entry": "60m bias = +1 if FT[60m] > SMA20[60m], -1 if below, 0 otherwise. 15m FT crosses its 20-SMA → take the long/short only if 60m bias agrees. Optional RSI > 50 / < 50 filter on V3 & V4.",
        "exit": "Same as standalone: stop 1.5× ATR(15m), TP1/2/3 at 1/2/3× ATR(15m), BE → +1ATR ladder.",
        "position": "3 contracts entered together. Stops are tighter than the 60m version (avg ~$80/lot) because 15m ATR is smaller.",
        "sweep": "4 variants × 4 sessions = 16 runs. P&L in points and dollars.",
        "notes": "★ Best run: V2 FT(HA) no-RSI, US Morning → +89.1 pts = +$4,455 (+8.9% on $50k), 149 entries, 67.7% WR, PF 1.35. Top PF: V4 FT(HA)+RSI, US Afternoon → PF 1.50. Underperforms the standalone 60m (~half the P&L) — the MTF gate filters out too many of the strong 60m signals. RSI filter starts to help here.",
    },
    {
        "id": "fisher_3lot_mtf_5m",
        "name": "Fisher 3-Lot MTF (60m bias + 5m entries)",
        "tag": "Fisher",
        "file": "backtest_fisher_3lot_mtf_chart.html",
        "summary": "Multi-timeframe Fisher 3-Lot at fastest entry timeframe. 60m FT vs 20-SMA sets bias; 5m FT crosses fire entries only when aligned.",
        "entry": "60m bias = +1 if FT[60m] > SMA20[60m], -1 if below. 5m FT crosses its 20-SMA → take the trade only if 60m bias agrees. Optional RSI > 50 / < 50 filter on V3 & V4.",
        "exit": "Same exit framework — stop 1.5× ATR(5m), TP 1/2/3× ATR(5m), BE → +1ATR ladder. Stops here are very tight (avg ~$40/lot) because 5m ATR is small.",
        "position": "3 contracts. ~1,800 entries on All-Day variants — way more trade activity than the slower timeframes.",
        "sweep": "4 variants × 4 sessions = 16 runs. P&L in points and dollars.",
        "notes": "Best run: V1 FT(Close) no-RSI, US Afternoon → +52 pts = +$2,606 (+5.2% on $50k), 512 entries, PF 1.11 (marginal). Only 5 of 16 runs profitable. 5m noise eats the edge — far worse than 15m or standalone 60m. Useful mainly as a data point showing that finer entry timeframe ≠ better.",
    },
]

TAG_COLORS = {
    "5m": "#58a6ff",
    "15m": "#3fb950",
    "MTF": "#d29922",
    "3-Lot": "#f85149",
    "Fisher": "#e599f7",
}


def render_card(s):
    tag_color = TAG_COLORS.get(s["tag"], "#8b949e")
    return f"""
        <div class="card" id="card-{s['id']}" onclick="loadStrategy('{s['id']}', '{s['file']}')">
          <div class="card-header">
            <span class="tag" style="background:{tag_color}22;color:{tag_color};border:1px solid {tag_color}44">{s['tag']}</span>
            <h3>{s['name']}</h3>
          </div>
          <p class="summary">{s['summary']}</p>
          <div class="spec">
            <div class="row"><span class="label">Entry</span><span class="value">{s['entry']}</span></div>
            <div class="row"><span class="label">Exit</span><span class="value">{s['exit']}</span></div>
            <div class="row"><span class="label">Position</span><span class="value">{s['position']}</span></div>
            <div class="row"><span class="label">Sweep</span><span class="value">{s['sweep']}</span></div>
            <div class="row notes"><span class="label">Notes</span><span class="value">{s['notes']}</span></div>
          </div>
          <div class="view-btn">View Chart →</div>
        </div>"""


def generate():
    cards_html = "".join(render_card(s) for s in STRATEGIES)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ES Futures — Strategy Dashboard</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}

  body {{
    background: #0d1117;
    color: #e0e0e0;
    font-family: 'SF Mono', 'Fira Code', 'Cascadia Code', monospace;
    padding: 0;
  }}

  .header {{
    background: #161b22;
    border-bottom: 1px solid #30363d;
    padding: 20px 32px;
    position: sticky;
    top: 0;
    z-index: 100;
  }}

  .header h1 {{
    color: #58a6ff;
    font-size: 22px;
    margin-bottom: 4px;
  }}

  .header p {{
    color: #8b949e;
    font-size: 13px;
  }}

  .banner {{
    background: #11203a;
    border: 1px solid #1f3a6a;
    border-radius: 8px;
    margin: 20px 32px;
    padding: 18px 22px;
  }}

  .banner h2 {{
    color: #d29922;
    font-size: 15px;
    margin-bottom: 12px;
    letter-spacing: 0.3px;
  }}

  .banner table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 12.5px;
    margin-bottom: 12px;
  }}

  .banner th {{
    text-align: left;
    color: #58a6ff;
    background: #0d1117;
    border: 1px solid #21262d;
    padding: 6px 10px;
    font-weight: 600;
    text-transform: uppercase;
    font-size: 10.5px;
    letter-spacing: 0.4px;
  }}

  .banner td {{
    color: #c9d1d9;
    border: 1px solid #21262d;
    padding: 5px 10px;
  }}

  .banner tr.best td {{
    background: #1a4d1a33;
    color: #3fb950;
    font-weight: 600;
  }}

  .banner .caveat {{
    color: #d8c79a;
    font-size: 11.5px;
    line-height: 1.55;
    margin-top: 6px;
  }}

  .grid-container {{
    padding: 24px 32px;
  }}

  .grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(420px, 1fr));
    gap: 18px;
  }}

  .card {{
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 18px 20px;
    cursor: pointer;
    transition: all 0.2s ease;
    display: flex;
    flex-direction: column;
    gap: 12px;
  }}

  .card:hover {{
    border-color: #58a6ff;
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(88, 166, 255, 0.15);
  }}

  .card.active {{
    border-color: #58a6ff;
    background: #1a2332;
    box-shadow: 0 0 0 1px #58a6ff;
  }}

  .card-header {{
    display: flex;
    align-items: center;
    gap: 10px;
  }}

  .card-header h3 {{
    color: #f0f6fc;
    font-size: 15px;
    font-weight: 600;
  }}

  .tag {{
    font-size: 11px;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 12px;
    white-space: nowrap;
  }}

  .summary {{
    color: #c9d1d9;
    font-size: 13px;
    line-height: 1.5;
    font-style: italic;
  }}

  .spec {{
    border-top: 1px solid #21262d;
    padding-top: 10px;
    display: flex;
    flex-direction: column;
    gap: 6px;
  }}

  .row {{
    display: grid;
    grid-template-columns: 70px 1fr;
    gap: 10px;
    align-items: start;
    font-size: 11.5px;
    line-height: 1.5;
  }}

  .row .label {{
    color: #58a6ff;
    font-weight: 600;
    text-transform: uppercase;
    font-size: 10.5px;
    letter-spacing: 0.5px;
    padding-top: 1px;
  }}

  .row .value {{
    color: #c9d1d9;
  }}

  .row.notes .label {{
    color: #d29922;
  }}

  .row.notes .value {{
    color: #d8c79a;
  }}

  .view-btn {{
    align-self: flex-start;
    background: #21262d;
    color: #58a6ff;
    font-size: 12px;
    font-weight: 600;
    padding: 6px 14px;
    border-radius: 6px;
    border: 1px solid #30363d;
    transition: all 0.15s ease;
    margin-top: 4px;
  }}

  .card:hover .view-btn {{
    background: #58a6ff;
    color: #0d1117;
    border-color: #58a6ff;
  }}

  .viewer {{
    padding: 0 32px 32px;
    display: none;
  }}

  .viewer.visible {{
    display: block;
  }}

  .viewer-header {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 16px 0;
  }}

  .viewer-header h2 {{
    color: #58a6ff;
    font-size: 16px;
  }}

  .close-btn {{
    background: #21262d;
    color: #8b949e;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 6px 12px;
    cursor: pointer;
    font-family: inherit;
    font-size: 12px;
    transition: all 0.15s ease;
  }}

  .close-btn:hover {{
    color: #f85149;
    border-color: #f85149;
  }}

  .chart-frame {{
    width: 100%;
    height: 85vh;
    border: 1px solid #30363d;
    border-radius: 8px;
    background: #0d1117;
  }}
</style>
</head>
<body>

<div class="header">
  <h1>ES Futures Strategy Dashboard</h1>
  <p>{len(STRATEGIES)} strategy variants — HA EMA, RSI, Fisher Transform, MTF, scaled exits | Data: real ES futures, Bloomberg, 6.5mo (Nov 2025 → Jun 2026), 24h Globex | Sessions: Overnight 00:00-04:00 · US Morning 04:00-13:00 · US Afternoon 13:00-17:00 · All Day</p>
</div>

<div class="banner">
  <h2>★ Latest ES results with commission (Bloomberg, 6.5 mo, full 24h Globex, $50/point, 3 lots/entry)</h2>
  <table>
    <thead><tr><th>Strategy</th><th>Session</th><th>Entries</th><th>Gross ($)</th><th>Net @ $4.90</th><th>Net @ $3.00</th><th>Return (gross)</th><th>Win %</th><th>PF</th></tr></thead>
    <tbody>
      <tr class="best"><td>HA+RSI 3-Lot MTF (60m+5m)</td><td>All Day</td><td>1,907</td><td>+266,975</td><td>+238,942</td><td>+249,812</td><td>+534%</td><td>68.0%</td><td>1.33</td></tr>
      <tr class="best"><td>HA+RSI 3-Lot MTF (60m+5m)</td><td>US Morning (04:00-13:00)</td><td>936</td><td>+165,449</td><td>+151,690</td><td>+157,025</td><td>+331%</td><td>68.3%</td><td><b>1.38</b></td></tr>
      <tr><td>HA+RSI 3-Lot MTF (60m+15m)</td><td>All Day</td><td>828</td><td>+215,084</td><td>+202,912</td><td>+207,632</td><td>+430%</td><td>67.7%</td><td>1.36</td></tr>
      <tr><td>HA+RSI 3-Lot MTF (60m+15m)</td><td>US Morning (04:00-13:00)</td><td>531</td><td>+117,851</td><td>+110,045</td><td>+113,072</td><td>+236%</td><td>66.9%</td><td>1.30</td></tr>
      <tr><td>HA+RSI 3-Lot MTF (60m+5m)</td><td>Overnight (00:00-04:00)</td><td>459</td><td>+38,662</td><td>+31,915</td><td>+34,531</td><td>+77.3%</td><td>66.8%</td><td>1.35</td></tr>
      <tr><td>HA+RSI 3-Lot MTF (60m+15m)</td><td>US Afternoon (13:00-17:00)</td><td>192</td><td>+53,980</td><td>+51,158</td><td>+52,252</td><td>+108%</td><td>66.8%</td><td>1.30</td></tr>
      <tr><td>HA+RSI 3-Lot MTF (60m+15m)</td><td>Overnight (00:00-04:00)</td><td>216</td><td>+26,427</td><td>+23,252</td><td>+24,483</td><td>+52.9%</td><td>66.2%</td><td>1.28</td></tr>
      <tr><td>HA+RSI 3-Lot MTF (60m+5m)</td><td>US Afternoon (13:00-17:00)</td><td>326</td><td>+19,674</td><td>+14,882</td><td>+16,740</td><td>+39.3%</td><td>66.0%</td><td>1.10</td></tr>
      <tr><td>Walk-forward 60+15 US Morning</td><td>Q1-Q4 composite (04:00-13:00)</td><td>531</td><td>+117,851</td><td>+110,045</td><td>+113,072</td><td>—</td><td>67.0%</td><td>avg 1.38 (Q4 = 0.96)</td></tr>
      <tr class="best"><td>★ Daily + Weekly bias (10yr ES, yfinance)</td><td>All Day (no session)</td><td>191</td><td>+371,349</td><td>+368,541</td><td>+369,630</td><td>+742.7%</td><td>72.1%</td><td><b>4.27</b></td></tr>
      <tr class="best"><td>★ 60m + Daily bias (2.4yr ES, yfinance)</td><td>All Day (no session)</td><td>712</td><td>+209,769</td><td>+199,302</td><td>+203,361</td><td>+419.5%</td><td>66.2%</td><td><b>3.66</b></td></tr>
    </tbody>
  </table>
  <p class="caveat"><b>Slow-MTF tail risk:</b> Daily+Weekly worst single 3-lot trade = <b>-$67,474</b> (134% of $50k account — would blow up). 60m+Daily worst = -$21,382 (43% of account). Avg daily ATR stop ≈ $4,295/lot. Either lower lot count or raise account size before trading slow TFs.</p>
  <p class="caveat"><b>Annualized comparison:</b> D+W ≈ $37k/yr · 60m+D ≈ $87k/yr · intraday 60+5 All Day ≈ $494k/yr (but 4-5× the trade count and bigger drawdowns).</p>
  <p class="caveat"><b>Commission formula:</b> entries × 3 lots × $/contract round-trip. Two scenarios shown: <b>$4.90/contract</b> (retail) and <b>$3.00/contract</b> (prop). Every strategy is still net-profitable after $4.90 commission. Slow-TF commission share is &lt;1% of gross (191 entries × $14.70 = $2,808 vs $371k gross).</p>
  <p class="caveat"><b>Caveats:</b> 6.5 mo window (Nov 30 2025 → Jun 15 2026), zero slippage. Q4 walk-forward shows edge softening — went <b>negative</b> on the expanded 04:00-13:00 window (PF 0.96, -$4,794). Baseline HA+RSI without MTF gate now <b>loses</b> on real ES — the MTF gate + 3-lot ladder is doing the work.</p>
  <p class="caveat"><b>Why the numbers jumped 10×:</b> prior runs applied $50/point to SPY-scale moves (SPY ATR ≈ 1 pt). Real ES ATR ≈ 10 pts. The strategy edge (PF, WR) is similar; the dollar economics were under-stated before.</p>
</div>

<div class="grid-container">
  <div class="grid">
    {cards_html}
  </div>
</div>

<div class="viewer" id="viewer">
  <div class="viewer-header">
    <h2 id="viewer-title">—</h2>
    <button class="close-btn" onclick="closeViewer()">Close Chart</button>
  </div>
  <iframe id="chart-frame" class="chart-frame" src="about:blank"></iframe>
</div>

<script>
  let activeCard = null;

  function loadStrategy(id, file) {{
    if (activeCard) activeCard.classList.remove('active');
    const card = document.getElementById('card-' + id);
    card.classList.add('active');
    activeCard = card;

    const viewer = document.getElementById('viewer');
    const frame = document.getElementById('chart-frame');
    const title = document.getElementById('viewer-title');

    title.textContent = card.querySelector('h3').textContent;
    frame.src = file;
    viewer.classList.add('visible');
    viewer.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
  }}

  function closeViewer() {{
    const viewer = document.getElementById('viewer');
    const frame = document.getElementById('chart-frame');
    viewer.classList.remove('visible');
    frame.src = 'about:blank';
    if (activeCard) {{
      activeCard.classList.remove('active');
      activeCard = null;
    }}
    window.scrollTo({{ top: 0, behavior: 'smooth' }});
  }}
</script>

</body>
</html>"""

    output = os.path.join(os.path.dirname(__file__) or ".", "dashboard.html")
    with open(output, "w") as f:
        f.write(html)

    print(f"Dashboard saved to {output}")
    webbrowser.open("file://" + os.path.abspath(output))
    print("Opened in browser.")


if __name__ == "__main__":
    generate()
