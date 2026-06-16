"""
Apply $4.90 and $3.00 per-contract round-trip commissions to the existing
intraday backtest results.

Commission = entries x 3 lots x $/contract round-trip.
(Each entry opens 3 lots; each lot has its own round-trip exit.)

Reads the log lines from runs/*.log, parses entries and gross P&L, prints a
side-by-side leaderboard with Gross / Net@$4.90 / Net@$3.00.
"""
import re
from pathlib import Path

RUNS = Path(__file__).parent / "runs"
LOTS_PER_ENTRY = 3
COMMS = [("Net @ $4.90", 4.90), ("Net @ $3.00", 3.00)]

# Source logs and what to label them as in the leaderboard
STRATEGY_LOGS = [
    ("HA+RSI 3-Lot MTF (60m+5m)",  "backtest_haRSI_3lot_mtf_5m.log"),
    ("HA+RSI 3-Lot MTF (60m+15m)", "backtest_haRSI_3lot_mtf_15m.log"),
]

# Lines we're parsing look like (one per session):
#   "All Day                         E= 1907  P&L= +5339.49pts  $  +266,975  Ret=+533.9%  WR= 68.0%  PF=1.329  ..."
LINE_RE = re.compile(
    r"^\s+(?P<sess>.+?)\s+E=\s*(?P<entries>\d+)\s+P&L=\s*(?P<pts>[+-]?[\d.]+)pts\s+\$\s*(?P<usd>[+-]?[\d,]+)\s+Ret=\s*(?P<ret>[+-]?[\d.]+)%\s+WR=\s*(?P<wr>[\d.]+)%\s+PF=(?P<pf>[\d.]+)"
)

def parse(log_path):
    rows = []
    with open(log_path) as f:
        for line in f:
            m = LINE_RE.match(line)
            if not m:
                continue
            sess = m["sess"].strip()
            entries = int(m["entries"])
            usd = int(m["usd"].replace(",", ""))
            rows.append({
                "session": sess,
                "entries": entries,
                "gross_usd": usd,
                "wr": float(m["wr"]),
                "pf": float(m["pf"]),
                "ret": float(m["ret"]),
            })
    return rows

# Aggregate
results = []
for name, fname in STRATEGY_LOGS:
    rows = parse(RUNS / fname)
    for r in rows:
        r["strategy"] = name
        for label, c in COMMS:
            haircut = r["entries"] * LOTS_PER_ENTRY * c
            r[label] = r["gross_usd"] - haircut
            r[f"haircut_{c}"] = haircut
        results.append(r)

# Order by gross
results.sort(key=lambda r: r["gross_usd"], reverse=True)

# Print
print(f"\n{'='*120}")
print(f"COMMISSION-ADJUSTED LEADERBOARD — Bloomberg ES, 6.5 mo, 3-lot per entry")
print(f"Commission applied as: entries x 3 lots x $/contract round-trip")
print(f"{'='*120}\n")

w = max(len(r['strategy']) for r in results)
print(f"  {'Strategy':<{w}}  {'Session':<28}  {'Entries':>7}  "
      f"{'Gross':>11}  {'Haircut$4.90':>13}  {'Net@$4.90':>11}  "
      f"{'Haircut$3.00':>13}  {'Net@$3.00':>11}  {'WR':>5}  {'PF':>5}")
print("  " + "-"*(w+5+28+5+7+5+11+5+13+5+11+5+13+5+11+5+5+5+5))

for r in results:
    print(
        f"  {r['strategy']:<{w}}  {r['session']:<28}  {r['entries']:>7,}  "
        f"${r['gross_usd']:>+10,}  "
        f"${r['haircut_4.9']:>11,.0f}  ${r[COMMS[0][0]]:>+10,}  "
        f"${r['haircut_3.0']:>11,.0f}  ${r[COMMS[1][0]]:>+10,}  "
        f"{r['wr']:>5.1f}  {r['pf']:>5.2f}"
    )

# Save a simple dict for the dashboard banner to pick up
import json
out = Path(__file__).parent / "commission_results.json"
with open(out, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved {out.name}")
