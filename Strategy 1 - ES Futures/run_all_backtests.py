"""
Batch-runner: re-execute every backtest script against the Bloomberg ES data.

- Monkey-patches webbrowser.open() to a no-op so each script doesn't spawn a browser tab.
- Imports each backtest module and calls its main() inside a try/except.
- Captures stdout into runs/<script>.log so failures don't pollute the console.
- Prints a one-line PASS/FAIL summary at the end.
"""
import os
import sys
import time
import traceback
import importlib
import contextlib
from pathlib import Path

# 1) Suppress browser pop-ups everywhere
import webbrowser
webbrowser.open = lambda *a, **kw: True
webbrowser.open_new = lambda *a, **kw: True
webbrowser.open_new_tab = lambda *a, **kw: True

# 2) Ensure Bloomberg ES is selected
os.environ["USE_BBG_ES"] = "1"

# 3) Ensure this directory is on sys.path
HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))

LOG_DIR = HERE / "runs"
LOG_DIR.mkdir(exist_ok=True)

# Order: cheap → expensive. Skip helpers (config, indicators, data_feed, _run_*).
SCRIPTS = [
    "backtest",                            # baseline 5m HA+RSI single contract
    "backtest_15m",                        # 15m grid 10x11x4
    "backtest_mtf",                        # 60m bias + 15m entries, 1-contract grid
    "backtest_mtf_5m",                     # 60m bias + 5m entries, 1-contract grid
    "backtest_fisher",                     # Fisher standalone
    "backtest_fisher_ema",                 # Fisher + EMA hybrid
    "backtest_fisher_3lot",                # Fisher + 3-lot ladder, standalone TFs
    "backtest_fisher_3lot_mtf",            # Fisher 3-lot, 60m+5m
    "backtest_fisher_3lot_mtf_15m",        # Fisher 3-lot, 60m+15m
    "backtest_haRSI_3lot_mtf_5m",          # ★ HA+RSI 3-lot, 60m+5m (prev winner family)
    "backtest_haRSI_3lot_mtf_15m",         # ★ HA+RSI 3-lot, 60m+15m (top single-session)
    "backtest_3lot",                       # legacy 3-lot (HA+RSI 5m)
    "session_analysis",                    # best baseline split by session
    "session_grid",                        # 440-combo heatmap
    "optimize",                            # 10x11 sweep on baseline
    "backtest_walkforward",                # walk-forward on winner
]

def run_one(name):
    log_path = LOG_DIR / f"{name}.log"
    print(f"\n=== {name} ===  -> {log_path.name}", flush=True)
    start = time.time()
    try:
        with open(log_path, "w") as f:
            with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                # Force fresh import so a prior run's module state is cleared
                if name in sys.modules:
                    del sys.modules[name]
                mod = importlib.import_module(name)
                if hasattr(mod, "main"):
                    mod.main()
                # else: script ran at import (e.g. backtest_fisher.py runs at module level for some)
        dur = time.time() - start
        print(f"   PASS  ({dur:0.1f}s)", flush=True)
        return ("PASS", dur, None)
    except SystemExit:
        dur = time.time() - start
        print(f"   PASS-exit ({dur:0.1f}s)", flush=True)
        return ("PASS", dur, None)
    except Exception as e:
        dur = time.time() - start
        err = traceback.format_exc()
        # Append the traceback to the log file
        with open(log_path, "a") as f:
            f.write("\n\n=== EXCEPTION ===\n")
            f.write(err)
        print(f"   FAIL  ({dur:0.1f}s) — {e.__class__.__name__}: {e}", flush=True)
        return ("FAIL", dur, str(e))

def main():
    results = []
    t0 = time.time()
    for name in SCRIPTS:
        status, dur, err = run_one(name)
        results.append((name, status, dur, err))
    total = time.time() - t0

    print(f"\n{'=' * 70}")
    print(f"SUMMARY  (total {total/60:0.1f} min)")
    print(f"{'=' * 70}")
    width = max(len(s) for s in SCRIPTS) + 2
    for name, status, dur, err in results:
        flag = "✅" if status == "PASS" else "❌"
        msg = f"  {flag}  {name:<{width}}  {dur:>6.1f}s"
        if err:
            msg += f"   {err[:80]}"
        print(msg)

if __name__ == "__main__":
    main()
