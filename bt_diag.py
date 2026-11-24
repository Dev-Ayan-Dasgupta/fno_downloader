"""Where do the losses come from?  python bt_diag.py [run_id] [m] [variant] [slippage]
Reads data/backtest/<run_id>/trades_m<m>_<variant>.csv (default: latest run, m 1.5, variant A, slippage 0.02)."""
import os
import re
import sys

import numpy as np
import pandas as pd

import bt_config as B

root = os.path.join(B.data_dir(), "backtest")
runs = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
run = sys.argv[1] if len(sys.argv) > 1 else runs[-1]
m = sys.argv[2] if len(sys.argv) > 2 else "1.5"
v = sys.argv[3] if len(sys.argv) > 3 else "A"
slip = float(sys.argv[4]) if len(sys.argv) > 4 else 0.02
tr = pd.read_csv(f"{root}/{run}/trades_m{m}_{v}.csv", parse_dates=["day", "signal_time"])
pd.set_option("display.width", 200)
print(f"Run {run} | m={m} variant={v} | {len(tr)} trades | slippage assumed {slip:.0%} per side\n")


def slip_cost_unit(r):
    exits = sum(float(f) * float(p) / (1 - slip) * slip for _, p, f in re.findall(r"(\w+)@([\d.]+)x([\d.]+)", r["legs"]))
    return r["entry"] * slip + exits


risk_px = tr["risk_pct"] * tr["entry"]
tr["slip_r"] = tr.apply(slip_cost_unit, axis=1) / risk_px
tr["pre_slip_r"] = tr["r_gross"] + tr["slip_r"]
print("FRICTIONS (average per trade, in R)")
print(f"  stop distance as share of premium: {tr['risk_pct'].mean():.3f}")
print(f"  gross R after slippage: {tr['r_gross'].mean():+.3f} | slippage cost: {tr['slip_r'].mean():.3f} | "
      f"gross R BEFORE slippage: {tr['pre_slip_r'].mean():+.3f} | brokerage and taxes: {(tr['r_gross'] - tr['r_net']).mean():.3f} | net: {tr['r_net'].mean():+.3f}\n")


def grp(col, name=None):
    g = tr.groupby(col, observed=True).agg(n=("r_net", "size"), avg_r_net=("r_net", "mean"), win_rate=("r_net", lambda x: (x > 0).mean()),
                                           pre_slip_r=("pre_slip_r", "mean")).round(3)
    print(f"BY {name or col}"); print(g.to_string()); print()


grp("direction")
grp("exit_reason")
tr["hour"] = tr["signal_time"].dt.strftime("%H") + ":xx"
grp("hour", "SIGNAL HOUR")
grp(pd.cut(tr["dte"], [0, 6, 13, 40]), "DAYS TO EXPIRY")
grp(pd.cut(tr["delta"].abs(), [0.35, 0.45, 0.55, 0.65]), "ABS DELTA")
grp(pd.cut(tr["risk_pct"], [0, 0.06, 0.10, 0.15, 0.25]), "STOP DISTANCE (share of premium)")
grp(tr["day"].dt.to_period("M"), "MONTH")
print("UNDERLYING VS OPTION (did the stock reach the T2 level vs did the option reach T2?)")
print(f"  stock reached T2 level at some point: {tr['u_reach_t2'].mean():.3f} | option hit T2: {tr['hit_t2'].mean():.3f}")
print(f"  stock outcome before its stop: {tr['u_outcome'].str.extract(r'^(T\\d|SL|EOD|none)')[0].value_counts(normalize=True).round(3).to_dict()}")
print(pd.crosstab(tr["u_reach_t2"], tr["hit_t2"], rownames=["stock reached T2"], colnames=["option hit T2"]).to_string(), "\n")
s = tr.groupby("symbol")["r_net"].agg(["size", "sum"]).query("size >= 5").sort_values("sum")
print("WORST 5 / BEST 5 STOCKS (>=5 trades, total net R)"); print(pd.concat([s.head(5), s.tail(5)]).round(1).to_string())
print(f"\nAverage MAE {tr['mae_r'].mean():.2f}R | MFE {tr['mfe_r'].mean():.2f}R")