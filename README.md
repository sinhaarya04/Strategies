<div align="center">

# 📈 Strategies

### **Systematic futures strategy research — backtests, walk-forwards, and live dashboards**

[![Python](https://img.shields.io/badge/Python-backtesting-3776AB?style=for-the-badge&logo=python&logoColor=white)](#)
[![ES Futures](https://img.shields.io/badge/Market-ES%20E--mini-0d1117?style=for-the-badge)](#)
[![Data](https://img.shields.io/badge/Data-Bloomberg%20%2B%20Yahoo-orange?style=for-the-badge)](#)

*A research monorepo: every strategy gets its own folder with code, data, figures, and findings.*

</div>

---

## 🧭 What's Inside

| Strategy | Market | Idea |
|----------|--------|------|
| [**Strategy 1 — ES Futures**](Strategy%201%20-%20ES%20Futures/) | S&P 500 E-mini | Heikin Ashi / EMA-band / RSI signal + **3-lot scaled-exit ladder**, gated by higher-timeframe bias |

Plus [`index.html`](index.html) — a GitHub-dark themed dashboard that indexes all strategies and their interactive backtest charts.

## ⚙️ Strategy 1 at a Glance

```
LONG  : HA close > 20-EMA(High) AND RSI(14) > 55
SHORT : HA close < 20-EMA(Low)  AND RSI(14) < 45

Exit  : 1.5×ATR stop · TP1 +1×ATR (→BE) · TP2 +2×ATR (→trail) · TP3 +3×ATR
```

The folder includes:

- **Multi-timeframe backtests** (5m / 15m / 60m entry with daily/weekly bias) — `backtest_*.py`
- **Fisher Transform and HA-RSI signal variants** — `backtest_fisher*.py`, `backtest_haRSI_*.py`
- **Walk-forward analysis** — `backtest_walkforward.py`
- **Commission-adjusted results** — `apply_commissions.py`, `commission_results.json`
- **Session analysis & parameter grids** — `session_analysis.py`, `session_grid.py`
- **Interactive HTML charts** for every run (`*_chart.html`) and a live `dashboard.py`
- **Bloomberg + Yahoo data loaders** with cached parquet bars (`ES_*.parquet`)
- **Write-ups** in LaTeX — `mtf_findings.tex`, `strategies_summary.tex`

## 🚀 Reproduce a Backtest

```bash
cd "Strategy 1 - ES Futures"
pip install pandas numpy yfinance plotly
python backtest_mtf.py          # multi-timeframe base run
python run_all_backtests.py     # everything
```

Open the generated `*_chart.html` files in a browser for interactive trade-by-trade review.

---

<div align="center">

*Research code — not investment advice. Past backtest performance ≠ future returns.*

</div>
