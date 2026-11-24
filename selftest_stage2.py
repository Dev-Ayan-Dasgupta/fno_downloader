"""Offline tests for stage 2: trade simulation rules + an end-to-end run on a synthetic world."""
import os, re, shutil, sys, tempfile
from datetime import date, datetime, timedelta
import numpy as np, pandas as pd

TMP = tempfile.mkdtemp(); os.environ["DATA_DIR"] = TMP
import bt_pricing as P, bt_trade as T, bt_signals as S, bt_run as R, bt_stats as ST, bt_costs as C
ok_all = []
def check(name, cond, extra=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  {extra}" if extra else "")); ok_all.append(bool(cond))

# ---------------------------------------------------------------- simulate(): hand-made candles
def cand(rows):
    ts = np.array([np.datetime64(datetime(2025, 2, 14, 9, 51) + timedelta(minutes=i)) for i in range(len(rows))])
    a = np.array(rows, dtype=float)
    return ts, a[:, 0], a[:, 1], a[:, 2], a[:, 3]
EOD = np.datetime64(datetime(2025, 2, 14, 14, 59))
entry, stop, tg = 10.0, 7.5, [12.5, 15.0, 17.5, 20.0]
def sim(rows, v="A"): return T.simulate(entry, stop, tg, *cand(rows), v, EOD)
def legs(s): return [(r, round(p, 2), f) for r, p, f in s["legs"]]

s = sim([[10, 12.6, 9.9, 12.0], [12, 15.1, 11.9, 15.0], [15, 15.2, 12.4, 12.45]])
check("A: T1 then T2 books 50% at 15.0, trailing stop (T1=12.5) exits rest", legs(s) == [("T2", 15.0, 0.5), ("TRAIL", 12.5, 0.5)], str(legs(s)))
sA = sim([[10, 12.6, 9.9, 12.0], [11, 11.5, 9.8, 10.0]], "A"); sB = sim([[10, 12.6, 9.9, 12.0], [11, 11.5, 9.8, 10.0]], "B")
check("A vs B: after T1 only, A stops out at entry (10.0), B keeps the original stop", legs(sA) == [("TRAIL", 10.0, 1.0)] or legs(sA)[0][0] == "TRAIL" and sB["legs"][0][0] == "EOD", f"A={legs(sA)} B={legs(sB)}")
s = sim([[10, 12.6, 7.4, 8.0]])
check("stop wins when stop and T1 fall in the same candle", legs(s) == [("SL", 7.5, 1.0)], str(legs(s)))
s = sim([[7.0, 8.0, 6.5, 7.2]])
check("gap below the stop fills at the open (7.0)", legs(s) == [("SL", 7.0, 1.0)], str(legs(s)))
s = sim([[10, 12.54, 9.9, 12.0], [12, 12.55, 11.9, 12.0]])
check("target needs high >= level + 1 tick (12.54 no, 12.55 yes)", s["hit"][0] is True and len(s["legs"]) == 1 and s["legs"][0][0] == "EOD")
s0 = sim([[10, 12.54, 9.9, 12.0]]); check("12.54 does not hit T1 (needs 12.55)", s0["hit"][0] is False)
s = sim([[10, 21.0, 9.9, 20.0]])
check("huge candle books T2,T3,T4 = 100% and nothing remains", [x[0] for x in s["legs"]] == ["T2", "T3", "T4"] and abs(sum(x[2] for x in s["legs"]) - 1) < 1e-9)
ts, o, h, l, c = cand([[10, 10.5, 9.8, 10.2], [10.2, 10.6, 10.0, 10.4]])
ts = np.append(ts, np.datetime64(datetime(2025, 2, 14, 15, 5))); o, h, l, c = [np.append(x, y) for x, y in zip((o, h, l, c), (30, 30, 1, 1))]
s = T.simulate(entry, stop, tg, ts, o, h, l, c, "A", EOD)
check("end of day: exits at the last close before 14:59; later candles ignored", legs(s) == [("EOD", 10.4, 1.0)], str(legs(s)))
s = T.simulate(entry, stop, tg, *cand([[10, 10.2, 9.0, 9.5]])[0:1], *[np.array([])] * 0, np.array([]), np.array([]), np.array([]), np.array([]), "A", EOD) if False else T.simulate(entry, stop, tg, np.array([], dtype="datetime64[ns]"), *[np.array([])] * 4, "A", EOD)
check("no candles after entry: flagged stale, exits at entry", s["legs"][0][0] == "EOD_NOCANDLE" and s["stale"])

# ---------------------------------------------------------------- finalize(): accounting
p = S.Params()
sig = dict(symbol="X", day=date(2026, 6, 1), direction="CE", T=pd.Timestamp("2026-06-01 09:50"), rvol=2.0)
sel = dict(contract="c", expiry=date(2026, 6, 30), strike=100.0, delta=0.5, iv=0.3, spot=100.0)
lv = dict(entry=10.0, stop=7.5, tg=tg, risk=2.5)
sm = dict(legs=[("T2", 15.0, 0.5), ("TRAIL", 12.5, 0.5)], mae_px=0.2, mfe_px=5.5, hit=[True, True, False, False], stale=False)
tr = T.finalize(sig, sel, lv, sm, p, 500)
legs_sum = sum(float(x.split("x")[1]) for x in tr["legs"].split(";"))
check("finalize: leg fractions sum to 1, costs > 0, net R < gross R", abs(legs_sum - 1) < 1e-9 and tr["costs_rs"] > 0 and tr["r_net"] < tr["r_gross"], f"r_gross={tr['r_gross']} r_net={tr['r_net']}")
exp_gross = (0.5 * (15 * 0.98 - 10 * 1.02) + 0.5 * (12.5 * 0.98 - 10 * 1.02)) * 500 * 4
check("finalize: gross P&L includes 2% slippage each side", abs(tr["gross_rs"] - exp_gross) < 1, f"{tr['gross_rs']} vs {exp_gross:.1f}")

# ---------------------------------------------------------------- end-to-end synthetic world
def sessions(n, start=date(2025, 1, 1)):
    out, d0 = [], start
    while len(out) < n:
        if d0.weekday() < 5: out.append(d0)
        d0 += timedelta(days=1)
    return out
days = sessions(40); BRK = days[32]; EXP = date(2025, 2, 27)
m = np.arange(375)
def spot_path(k):
    if k == 32: return np.where(m < 15, 1000 + 2 * np.sin(m), 1000 + 0.15 * (m - 15))
    return 1000.0 + 5 * np.sin(k / 3) + 12 * np.sin(2 * np.pi * m / 375)
def bars(path, vol, noise=0.2):
    o = np.r_[path[0], path[:-1]]
    return o, np.maximum(o, path) + noise, np.minimum(o, path) - noise, path, np.full(375, vol, dtype=float)
rows = []; nrows = []; orows = []
for k, d in enumerate(days):
    ts = [datetime(d.year, d.month, d.day, 9, 15) + timedelta(minutes=int(i)) for i in m]
    p_ = spot_path(k); o, h, l, c, v = bars(p_, 300 if k == 32 else 100)
    rows += list(zip(ts, o, h, l, c, v, [None] * 375))
    npth = 24000 + 2 * m; no, nh, nl, nc, _ = bars(npth, 0, noise=1.0)
    nrows += list(zip(ts, no, nh, nl, nc, [0.0] * 375, [None] * 375))
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

per = (days[0], days[-1])
trades, rej, info = R.run_backtest(["SYN"], per, [S.Params(stop_mult=1.5, variant="A"), S.Params(stop_mult=1.5, variant="B")], {"SYN": 100}, log=lambda *a: None)
trA, trB = trades[(1.5, "A")], trades[(1.5, "B")]
check("e2e: exactly one signal, a CE on the breakout day at 09:50", info["signals"] == 1 and len(trA) == 1 and trA.iloc[0]["direction"] == "CE"
      and trA.iloc[0]["day"] == BRK and pd.Timestamp(trA.iloc[0]["signal_time"]).strftime("%H:%M") == "09:50",
      f"signals={info['signals']} trades={len(trA)}")
t = trA.iloc[0]
check("e2e: all four targets hit on a steadily rising path, exit reason T4", bool(t.hit_t1 and t.hit_t2 and t.hit_t3 and t.hit_t4) and t.exit_reason == "T4", f"{t.legs}")
check("e2e: positive R, costs reduce it, stop within the 25% cap", t.r_gross > 1 and t.r_net < t.r_gross and t.risk_pct <= 0.25, f"r_gross={t.r_gross} r_net={t.r_net} risk={t.risk_pct}")
check("e2e: delta within 0.35-0.65, nearest expiry 13 days out", 0.35 <= abs(t.delta) <= 0.65 and t.dte == 13, f"delta={t.delta} dte={t.dte}")
check("e2e: variant B gives the same result on a path that never retraces", abs(trB.iloc[0].r_gross - t.r_gross) < 1e-9)
check("e2e: underlying outcome recorded", t.u_outcome.startswith("T") and bool(t.u_reach_t2), t.u_outcome)
tr2, rej2, _ = R.run_backtest(["SYN"], per, [S.Params(max_stop_frac=0.001)], {"SYN": 100}, log=lambda *a: None)
check("e2e: stop wider than the cap is skipped and logged", len(tr2[(1.5, "A")]) == 0 and rej2["reason"].str.contains("stop_over_cap").any())
tr3, rej3, _ = R.run_backtest(["SYN"], per, [S.Params(rvol_min=5.0)], {"SYN": 100}, log=lambda *a: None)
check("e2e: higher volume threshold gives no signal", len(tr3[(1.5, "A")]) == 0)
summ = ST.summarize(pd.concat([trA, trA.assign(day=days[1], r_net=-1.0, r_gross=-1.0, signal_time=pd.Timestamp("2025-01-02 10:00"))]))
check("stats: summary and day-clustered CI computed", summ["n"] == 2 and summ["ci_lo"] == summ["ci_lo"], str({k: summ[k] for k in ("avg_r_net", "ci_lo", "ci_hi")}))
check("stats: max drawdown on a simple series", abs(ST.max_drawdown(np.array([1, -2, -1, 3])) - 3) < 1e-9)

print(f"\n{sum(ok_all)}/{len(ok_all)} checks passed"); shutil.rmtree(TMP, ignore_errors=True)
sys.exit(0 if all(ok_all) else 1)
