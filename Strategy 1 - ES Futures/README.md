# Strategy 1 — ES E-Mini Futures (HA EMA + RSI + 3-Lot Scaled Exit)

Multi-timeframe backtests on the S&P 500 E-mini (ES) futures. The strategy combines a Heikin Ashi / EMA-band / RSI signal with a 3-lot scaled-exit ladder, gated by a higher-timeframe bias.

## Signal

```
LONG  : HA close > 20-EMA(High) AND RSI(14) > 55
SHORT : HA close < 20-EMA(Low)  AND RSI(14) < 45
```

A slower timeframe runs the same logic to set the directional bias (long / short / neutral). Entries on the faster timeframe only fire when they agree with that bias.

## Exit (3-lot ladder)

```
Stop : 1.5 × ATR(entry TF)
TP1  : +1 × ATR  → close lot 1, move stop on lots 2 & 3 to break-even
TP2  : +2 × ATR  → close lot 2, trail stop on lot 3 to +1 × ATR
TP3  : +3 × ATR  → close lot 3 (or trail-stopped first)
```

## Timeframe pairs tested

| Bias bar | Entry bar | Approx hold | Avg stop $/lot | Best PF |
|---|---|---|---:|---:|
| 60-min | 5-min | minutes–hours | $409 | 1.33 |
| 60-min | 15-min | hours | $744 | 1.36 |
| Daily | 60-min | hours–day | $1,102 | **3.66** |
| Weekly | Daily | days–weeks | $4,295 | **4.27** |

## Sessions (ET)

- All Day — 00:00 – 23:59
- Overnight — 00:00 – 04:00
- US Morning — 04:00 – 13:00
- US Afternoon — 13:00 – 17:00

(Daily/Weekly runs ignore sessions — each bar already spans the full day/week.)

## Headline results

6.5 months of Bloomberg intraday + 10 years of yfinance daily/weekly, $50k account, $50/point, ES point value, 3 contracts/entry, ATR-sized stops.

| Strategy | Period | Entries | Gross | Net @ $4.90 | Net @ $3.00 | WR | PF |
|---|---|---:|---:|---:|---:|---:|---:|
| HA+RSI 3-Lot MTF (60+5), All Day | 6.5 mo | 1,907 | +$266,975 | +$238,942 | +$249,812 | 68.0% | 1.33 |
| HA+RSI 3-Lot MTF (60+15), All Day | 6.5 mo | 828 | +$215,084 | +$202,912 | +$207,632 | 67.7% | 1.36 |
| **Daily + Weekly bias** | **10 yr** | 191 | **+$371,349** | **+$368,541** | **+$369,630** | **72.1%** | **4.27** |
| **60m + Daily bias** | **2.4 yr** | 712 | **+$209,769** | **+$199,302** | **+$203,361** | **66.2%** | **3.66** |

Commissions:
- Two scenarios — **$4.90/contract** round-trip (retail) and **$3.00/contract** (prop)
- Formula: entries × 3 lots × $/contract
- Slow-MTF commission share is < 1% of gross

## What's in this repo

### Source

- `config.py` — global constants (EMA20, RSI14/55/45, ATR14, $50/pt, $50k account)
- `data_feed.py` — Bloomberg parquet loader + Polygon SPY fallback + yfinance fallback
- `indicators.py` — Heikin Ashi candles, EMA bands, RSI, ATR, Fisher Transform
- `strategy.py` — Backtrader baseline reference

### Backtest scripts

| File | What it runs |
|---|---|
| `backtest.py` | Single-pass baseline 5m HA+RSI |
| `backtest_15m.py` | 15m grid sweep (10 stops × 11 targets × 4 sessions) |
| `backtest_mtf.py` / `backtest_mtf_5m.py` | 60m bias + 15m or 5m entries, single contract |
| `backtest_fisher*.py` | Fisher-Transform variants (standalone, EMA-hybrid, 3-lot, MTF) |
| `backtest_haRSI_3lot_mtf_5m.py` / `_15m.py` | ★ HA+RSI 3-lot MTF — the intraday winner family |
| `backtest_slow_mtf.py` | ★ Daily+Weekly and 60m+Daily slow-MTF runs |
| `backtest_walkforward.py` | Q1–Q4 walk-forward on the winner |
| `optimize.py`, `session_analysis.py`, `session_grid.py`, `compare.py` | Analysis helpers |
| `dashboard.py` | Generates `dashboard.html` (the master view) |
| `make_figures.py` | Generates equity-curve PNGs in `figures/` |
| `apply_commissions.py` | Post-processes logs to compute net P&L at $4.90 and $3.00 |
| `run_all_backtests.py` | Batch runner — executes every backtest, captures logs to `runs/` |
| `pull_es_data.py` / `load_bloomberg_es.py` | Data loaders (yfinance + Bloomberg `data (1).xlsx`) |

### Data

- `ES_5m_bbg.parquet`, `ES_15m_bbg.parquet`, `ES_60m_bbg.parquet` — Bloomberg ES futures, 24h Globex, Nov 2025 → Jun 2026
- `ES_1d_10y.parquet`, `ES_1wk_10y.parquet` — yfinance ES=F continuous, 10 yr
- `ES_60m_730d.parquet` — yfinance ES=F 60m, 2.4 yr
- `ES_5m_60d.parquet`, `ES_15m_60d.parquet` — yfinance 60-day intraday (limit)

### Outputs

- `dashboard.html` — main interactive leaderboard + per-strategy cards
- `figures/` — equity-curve PNGs per strategy
- `runs/` — captured stdout per script
- `*_chart.html` — per-strategy Plotly charts (smaller variants kept; the 50MB+ ones excluded)
- `commission_results.json`, `slow_mtf_results.json` — machine-readable results
- `strategies_summary.tex`, `mtf_findings.tex` — write-ups

## Run

```bash
# Drop the Bloomberg Excel into ~/Desktop/data (1).xlsx first, then:
python load_bloomberg_es.py    # cache ES bars to parquet
python pull_es_data.py         # pull yfinance daily / weekly / 60m

# Headline run:
python backtest_haRSI_3lot_mtf_15m.py
python backtest_haRSI_3lot_mtf_5m.py
python backtest_slow_mtf.py

# Or batch every backtest:
python run_all_backtests.py

# Dashboard:
python dashboard.py            # → opens dashboard.html
```

Requires `pandas`, `numpy`, `plotly`, `matplotlib`, `pyarrow`, `yfinance`.

## Caveats

- **Bloomberg intraday window is only 6.5 months** (Nov 30 2025 → Jun 15 2026). Slow-MTF (Daily/Weekly) uses 10-year yfinance data which is more representative.
- **Tail risk on slow MTF is large**: Daily + Weekly worst single 3-lot trade was −$67,474 (134% of a $50k account). Either trade fewer lots or use a larger account.
- **Q4 walk-forward weakened**: on the expanded 04:00–13:00 window Q4 went mildly negative (PF 0.96). Recent regime is softer.
- **Baseline HA+RSI without MTF gate now loses on real ES** — the MTF gate + 3-lot ladder is doing the work, the bare signal isn't enough.
- **Zero slippage** in every run. Real ES round-trip slippage typically adds another 0.25–0.5 pts of cost per contract.
