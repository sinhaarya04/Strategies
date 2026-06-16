"""
ES E-mini Futures Backtest Runner
Heikin Ashi EMA Band + RSI Strategy
"""

import sys
import backtrader as bt
import backtrader.analyzers as btanalyzers
from data_feed import fetch_es_data
from strategy import HeikinAshiEMAStrategy
from config import INITIAL_CAPITAL, COMMISSION_PER_CONTRACT, MARGIN_PER_CONTRACT


def run_backtest(plot=False):
    cerebro = bt.Cerebro()

    # Fetch data
    print("=" * 55)
    print("  ES E-MINI FUTURES BACKTEST")
    print("  Strategy: Heikin Ashi EMA Band + RSI")
    print("=" * 55)

    df = fetch_es_data()
    data = bt.feeds.PandasData(dataname=df)
    cerebro.adddata(data)

    # Strategy
    cerebro.addstrategy(HeikinAshiEMAStrategy)

    # Broker settings
    cerebro.broker.setcash(INITIAL_CAPITAL)
    cerebro.broker.setcommission(
        commission=COMMISSION_PER_CONTRACT,
        margin=MARGIN_PER_CONTRACT,
        mult=50.0,  # ES = $50/point
    )

    # Analyzers
    cerebro.addanalyzer(btanalyzers.SharpeRatio, _name="sharpe", timeframe=bt.TimeFrame.Days)
    cerebro.addanalyzer(btanalyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(btanalyzers.TradeAnalyzer, _name="trades")
    cerebro.addanalyzer(btanalyzers.Returns, _name="returns")

    # Run
    print(f"\n  Capital: ${INITIAL_CAPITAL:,.0f}")
    print(f"  Commission: ${COMMISSION_PER_CONTRACT}/contract RT")
    print(f"  Margin: ${MARGIN_PER_CONTRACT}/contract")
    print(f"  Bars: {len(df)}")
    print("-" * 55)

    results = cerebro.run()
    strat = results[0]

    # Results
    final_value = cerebro.broker.getvalue()
    pnl = final_value - INITIAL_CAPITAL
    ret_pct = (pnl / INITIAL_CAPITAL) * 100

    sharpe = strat.analyzers.sharpe.get_analysis()
    dd = strat.analyzers.drawdown.get_analysis()
    trades = strat.analyzers.trades.get_analysis()

    total_trades = trades.get("total", {}).get("total", 0)
    won = trades.get("won", {}).get("total", 0)
    lost = trades.get("lost", {}).get("total", 0)
    win_rate = (won / total_trades * 100) if total_trades > 0 else 0

    avg_win = trades.get("won", {}).get("pnl", {}).get("average", 0)
    avg_loss = trades.get("lost", {}).get("pnl", {}).get("average", 0)

    print("\n" + "=" * 55)
    print("  RESULTS")
    print("=" * 55)
    print(f"  Final Value:    ${final_value:,.2f}")
    print(f"  P&L:            ${pnl:+,.2f} ({ret_pct:+.2f}%)")
    print(f"  Sharpe Ratio:   {sharpe.get('sharperatio', 'N/A')}")
    print(f"  Max Drawdown:   {dd.get('max', {}).get('drawdown', 0):.2f}%")
    print(f"  Max DD ($):     ${dd.get('max', {}).get('moneydown', 0):,.2f}")
    print("-" * 55)
    print(f"  Total Trades:   {total_trades}")
    print(f"  Won:            {won} ({win_rate:.1f}%)")
    print(f"  Lost:           {lost}")
    print(f"  Avg Win:        ${avg_win:,.2f}")
    print(f"  Avg Loss:       ${avg_loss:,.2f}")

    if avg_loss != 0:
        print(f"  Profit Factor:  {abs(avg_win * won) / abs(avg_loss * lost):.2f}" if lost > 0 else "  Profit Factor:  inf")

    print("=" * 55)

    if plot:
        cerebro.plot(style="candle", volume=False)

    return results


if __name__ == "__main__":
    do_plot = "--plot" in sys.argv
    run_backtest(plot=do_plot)
