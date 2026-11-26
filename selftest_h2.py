"""Tests for hypothesis H2 (pullback entry) on a synthetic world with a scripted thrust and retest."""
import os, shutil, sys, tempfile
from datetime import date, datetime, timedelta
import numpy as np, pandas as pd

TMP = tempfile.mkdtemp(); os.environ["DATA_DIR"] = TMP
import bt_pricing as P, bt_signals as S, bt_run as R
ok_all = []
def check(name, cond, extra=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  {extra}" if extra else "")); ok_all.append(bool(cond))

def sessions(n, start=date(2025, 1, 1)):
    out, d0 = [], start
    while len(out) < n:
        if d0.weekday() < 5: out.append(d0)
        d0 += timedelta(days=1)
    return out
days = sessions(40); BRK = days[32]; EXP = date(2025, 2, 27); m = np.arange(375)

def build(day_path):
    shutil.rmtree(TMP, ignore_errors=True); os.makedirs(TMP)
    rows, nrows, orows = [], [], []
    for k, d in enumerate(days):
        ts = [datetime(d.year, d.month, d.day, 9, 15) + timedelta(minutes=int(i)) for i in m]
        p_ = day_path() if k == 32 else 1000.0 + 5 * np.sin(k / 3) + 12 * np.sin(2 * np.pi * m / 375)
        o = np.r_[p_[0], p_[:-1]]; v = 300.0 if k == 32 else 100.0
        rows += list(zip(ts, o, np.maximum(o, p_) + 0.2, np.minimum(o, p_) - 0.2, p_, np.full(375, v), [None] * 375))
        npth = 24000 + 2 * m; no = np.r_[npth[0], npth[:-1]]
        nrows += list(zip(ts, no, np.maximum(no, npth) + 1, np.minimum(no, npth) - 1, npth, [0.0] * 375, [None] * 375))
        if 8 <= k <= 32:
            for K in range(960, 1070, 10):
                for typ in ("CE", "PE"):
                    gs = f"NSE-SYN-27Feb25-{K}-{typ}"
                    pr = np.array([P.bs_price(p_[i], K, P.time_to_expiry(ts[i], EXP), 0.30, typ) for i in range(375)])
                    po = np.r_[pr[0], pr[:-1]]
                    for i in range(375):
                        orows.append((ts[i], gs, float(K), typ, po[i], max(po[i], pr[i]) + 0.05, max(min(po[i], pr[i]) - 0.05, 0.05), pr[i], 100.0, 1000.0))
    cols = ["ts", "o", "h", "l", "c", "v", "oi"]
    for sym, r_ in (("SYN", rows), ("NIFTY", nrows)):
        os.makedirs(f"{TMP}/stock_1m/{sym}", exist_ok=True); pd.DataFrame(r_, columns=cols).to_parquet(f"{TMP}/stock_1m/{sym}/a.parquet", index=False)
    os.makedirs(f"{TMP}/opt_1m/SYN", exist_ok=True)
    pd.DataFrame(orows, columns=["ts", "contract", "strike", "typ", "o", "h", "l", "c", "v", "oi"]).to_parquet(f"{TMP}/opt_1m/SYN/{EXP}.parquet", index=False)

def pullback_path():       # opening range, thrust to 1018, retest of the zone, then continuation
    return np.where(m < 15, 1000 + 2 * np.sin(m), np.where(m < 40, 1000 + (m - 15) * 0.72,
           np.where(m < 60, 1018 - (m - 40) * 0.5, 1008 + (m - 60) * 0.3)))
def failing_path():        # thrust, then the retest closes below the zone
    return np.where(m < 15, 1000 + 2 * np.sin(m), np.where(m < 40, 1000 + (m - 15) * 0.72, np.where(m < 60, 1018 - (m - 40) * 1.0, 998 + (m - 60) * 0.02)))

p = S.Params(hyp="pullback")
build(pullback_path)
ctx = S.build_ctx("SYN"); nifty5, _ = S.index_context("NIFTY")
sigs, rej = S.find_pullback_signals(ctx, nifty5, p)
bsigs, _ = S.find_signals(ctx, nifty5, S.Params())
check("H2: exactly one pullback signal (CE) on the scripted day", len(sigs) == 1 and sigs[0]["direction"] == "CE" and sigs[0]["day"] == BRK, f"{len(sigs)} signals")
sg = sigs[0]
check("H2: signal comes after the setup, within 12 bars and before 12:00", sg["T"] > sg["setup_T"] and 1 <= sg["wait_bars"] <= 12 and sg["T"].time() <= datetime(2000, 1, 1, 12, 0).time(),
      f"setup {sg['setup_T'].time()} -> signal {sg['T'].time()} ({sg['wait_bars']} bars)")
Z = max(sg["or_high"], sg["vwap"]); tol = 0.1 * sg["atr5"]
check("H2: pullback bar touched the zone and closed above it", sg["bar_l"] <= Z + tol and sg["c"] > Z, f"low {sg['bar_l']:.2f} zone {Z:.2f} close {sg['c']:.2f}")
check("H2: thrust at least 1 ATR(5m) above the zone", sg["thrust"] >= 1.0, f"thrust {sg['thrust']:.1f} ATR")
check("H2: signal is later than the baseline breakout signal on the same day", len(bsigs) >= 1 and sg["T"] > min(b["T"] for b in bsigs if b["day"] == BRK))
tr, rej2, info = R.run_backtest(["SYN"], (days[0], days[-1]), [S.Params(hyp="pullback", stop_mult=1.5, variant="A")], {"SYN": 100}, log=lambda *a: None)
t = tr[(1.5, "A")]
check("H2: full pipeline runs; one trade or a logged reason", info["signals"] == 1 and (len(t) == 1 or len(rej2) >= 1), f"trades={len(t)}")
if len(t):
    check("H2: trade accounting is sane", abs(sum(float(x.split('x')[1]) for x in t.iloc[0]["legs"].split(";")) - 1) < 1e-9 and t.iloc[0]["risk_pct"] <= 0.25)

build(failing_path)
ctx = S.build_ctx("SYN"); sigs, rej = S.find_pullback_signals(ctx, nifty5, p)
check("H2: retest that closes below the zone cancels the setup", len(sigs) == 0 and any(r["reason"] == "failed_retest" for r in rej), f"{[r['reason'] for r in rej]}")
build(pullback_path); ctx = S.build_ctx("SYN")
sigs, _ = S.find_pullback_signals(ctx, nifty5, S.Params(hyp="pullback", pb_wait=3))
check("H2: wait limit respected (3 bars is too short for this pullback)", len(sigs) == 0)
sigs, _ = S.find_pullback_signals(ctx, nifty5, S.Params(hyp="pullback", pb_thrust=50.0))
check("H2: insufficient thrust gives no signal", len(sigs) == 0)

print(f"\n{sum(ok_all)}/{len(ok_all)} checks passed"); shutil.rmtree(TMP, ignore_errors=True)
sys.exit(0 if all(ok_all) else 1)
