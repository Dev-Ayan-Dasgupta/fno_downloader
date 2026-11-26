"""Offline tests: trade simulation rules, execution conventions, ATR modes, data completeness, frozen configuration."""
import os, shutil, sys, tempfile
from datetime import date, datetime, timedelta
import numpy as np, pandas as pd

TMP = tempfile.mkdtemp(); os.environ["DATA_DIR"] = TMP
import bt_data as D, bt_pricing as P, bt_trade as T, bt_signals as S, bt_run as R, bt_stats as ST
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
def sim(rows, v="A", mode="optimistic"): return T.simulate(entry, stop, tg, *cand(rows), v, EOD, intrabar_mode=mode)
def legs(s): return [(r, round(float(p), 2), f) for r, p, f in s["legs"]]

s = sim([[10, 12.6, 10.1, 12.0], [12, 15.1, 11.9, 15.0], [15, 15.2, 12.4, 12.45]])
check("optimistic A: T1 then T2 books 50% at 15.0, trailing stop (T1=12.5) exits the rest", legs(s) == [("T2", 15.0, 0.5), ("TRAIL", 12.5, 0.5)], str(legs(s)))
sA = sim([[10, 12.6, 10.1, 12.0], [11, 11.5, 9.8, 10.0]], "A"); sB = sim([[10, 12.6, 10.1, 12.0], [11, 11.5, 9.8, 10.0]], "B")
check("A vs B: after T1 only, A stops out at entry (10.0); B keeps the original stop", legs(sA) == [("TRAIL", 10.0, 1.0)] and sB["legs"][0][0] == "EOD", f"A={legs(sA)} B={legs(sB)}")
s = sim([[10, 12.6, 7.4, 8.0]]); check("stop wins when stop and T1 fall in the same candle", legs(s) == [("SL", 7.5, 1.0)], str(legs(s)))
s = sim([[7.0, 8.0, 6.5, 7.2]]); check("gap below the stop fills at the open (7.0)", legs(s) == [("SL", 7.0, 1.0)], str(legs(s)))
s = sim([[10, 12.54, 9.9, 12.0]]); check("12.54 does not hit T1 (needs 12.55)", s["hit"][0] is False)
s = sim([[10, 12.55, 10.1, 12.0]]); check("12.55 hits T1", s["hit"][0] is True)
s = sim([[10, 21.0, 9.9, 20.0]]); check("huge candle books T2,T3,T4 = 100%", [x[0] for x in s["legs"]] == ["T2", "T3", "T4"] and abs(sum(x[2] for x in s["legs"]) - 1) < 1e-9)
ts, o, h, l, c = cand([[10, 10.5, 9.8, 10.2], [10.2, 10.6, 10.0, 10.4]])
ts = np.append(ts, np.datetime64(datetime(2025, 2, 14, 15, 5))); o, h, l, c = [np.append(x, y) for x, y in zip((o, h, l, c), (30, 30, 1, 1))]
s = T.simulate(entry, stop, tg, ts, o, h, l, c, "A", EOD)
check("end of day: exits at the last close before 14:59; later candles ignored", legs(s) == [("EOD", 10.4, 1.0)], str(legs(s)))
s = T.simulate(entry, stop, tg, np.array([], dtype="datetime64[ns]"), *[np.array([])] * 4, "A", EOD)
check("no candles after entry: flagged stale, exits at entry", s["legs"][0][0] == "EOD_NOCANDLE" and s["stale"])
# conservative intrabar handling (audit item): a stop raised by a target inside a candle applies to that same candle
c1 = [[10, 12.6, 9.9, 12.0]]
check("conservative A: T1 and a dip to entry in one candle -> stopped at entry", legs(sim(c1, "A", "conservative")) == [("TRAIL", 10.0, 1.0)], str(legs(sim(c1, "A", "conservative"))))
check("optimistic A: the same candle keeps the trade open", sim(c1, "A", "optimistic")["legs"][0][0] == "EOD")
check("conservative B: T1 does not move the stop, so the trade survives", sim(c1, "B", "conservative")["legs"][0][0] == "EOD")
s = sim([[13, 15.1, 11.0, 14.0]], "A", "conservative")
check("conservative: T2 booked, then the new stop (T1=12.5) hit inside the same candle", legs(s) == [("T2", 15.0, 0.5), ("TRAIL", 12.5, 0.5)], str(legs(s)))
def sim2(rows, mode="conservative"): return T.simulate(entry, stop, [12.5, 15.0, np.nan, np.nan], *cand(rows), "A", EOD, intrabar_mode=mode, scheme="t1t2")
s = sim2([[10, 12.6, 10.1, 12.0], [12, 15.1, 11.9, 15.0]])
check("t1t2: 50% at T1, stop to entry, 50% at T2", legs(s) == [("T1", 12.5, 0.5), ("T2", 15.0, 0.5)], str(legs(s)))
s = sim2([[10, 12.6, 10.1, 12.0], [11, 11.5, 9.9, 10.0]])
check("t1t2: after T1 a return to entry exits the second half at entry", legs(s) == [("T1", 12.5, 0.5), ("TRAIL", 10.0, 0.5)], str(legs(s)))
s = sim2(c1)
check("t1t2 conservative: T1 and a dip to entry in the same candle -> second half exits at entry", legs(s) == [("T1", 12.5, 0.5), ("TRAIL", 10.0, 0.5)], str(legs(s)))
s = sim2(c1, "optimistic"); check("t1t2 optimistic: the same candle keeps the second half open", [x_[0] for x_ in s["legs"]] == ["T1", "EOD"])
s = sim2([[10, 12.0, 7.4, 8.0]]); check("t1t2: stop before T1 exits everything at the stop", legs(s) == [("SL", 7.5, 1.0)], str(legs(s)))
s = sim2([[10, 12.6, 10.1, 12.0], [12, 12.4, 11.0, 11.5]]); check("t1t2: after T1, flat to the end of the day exits the second half at the last close", legs(s) == [("T1", 12.5, 0.5), ("EOD", 11.5, 0.5)], str(legs(s)))
try:
    T.simulate(entry, stop, tg, *cand(c1), "A", EOD, intrabar_mode="bogus"); check("unknown intrabar mode rejected", False)
except ValueError:
    check("unknown intrabar mode rejected", True)

# ---------------------------------------------------------------- finalize(): accounting
p = S.Params()
sig = dict(symbol="X", day=date(2026, 6, 1), direction="CE", T=pd.Timestamp("2026-06-01 09:50"), rvol=2.0)
sel = dict(contract="c", expiry=date(2026, 6, 30), strike=100.0, delta=0.5, iv=0.3, spot=100.0)
lv = dict(entry=10.0, stop=7.5, tg=tg, risk=2.5)
sm = dict(legs=[("T2", 15.0, 0.5), ("TRAIL", 12.5, 0.5)], mae_px=0.2, mfe_px=5.5, hit=[True, True, False, False], stale=False)
tr = T.finalize(sig, sel, lv, sm, p, 500)
legs_sum = sum(float(x.split("x")[1]) for x in tr["legs"].split(";"))
check("finalize: fractions sum to 1, costs > 0, net R < gross R, modes recorded", abs(legs_sum - 1) < 1e-9 and tr["costs_rs"] > 0 and tr["r_net"] < tr["r_gross"]
      and tr["execution_mode"] == "next_open" and tr["intrabar_mode"] == "conservative")
exp_gross = (0.5 * (15 * 0.98 - 10 * 1.02) + 0.5 * (12.5 * 0.98 - 10 * 1.02)) * 500 * 4
check("finalize: gross P&L includes 2% slippage each side", abs(tr["gross_rs"] - exp_gross) < 1, f"{tr['gross_rs']} vs {exp_gross:.1f}")

# ---------------------------------------------------------------- synthetic world
def sessions(n, start=date(2025, 1, 1)):
    out, d0 = [], start
    while len(out) < n:
        if d0.weekday() < 5: out.append(d0)
        d0 += timedelta(days=1)
    return out
days = sessions(40); BRK = days[32]; EXP = date(2025, 2, 27); m = np.arange(375)
def spot_path(k):
    if k == 32: return np.where(m < 15, 1000 + 2 * np.sin(m), 1000 + 0.15 * (m - 15))
    return 1000.0 + 5 * np.sin(k / 3) + 12 * np.sin(2 * np.pi * m / 375)
rows = []; nrows = []; orows = []
for k, d in enumerate(days):
    ts = [datetime(d.year, d.month, d.day, 9, 15) + timedelta(minutes=int(i)) for i in m]
    p_ = spot_path(k); o = np.r_[p_[0], p_[:-1]]; v = 300.0 if k == 32 else 100.0
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

per = (days[0], days[-1]); LOG = lambda *a: None
trades, rej, info = R.run_backtest(["SYN"], per, [S.Params(stop_mult=1.5, variant="A"), S.Params(stop_mult=1.5, variant="B")], {"SYN": 100}, log=LOG)
trA, trB = trades[(1.5, "A")], trades[(1.5, "B")]
check("e2e: exactly one signal, a CE on the breakout day at 09:50", info["signals"] == 1 and len(trA) == 1 and trA.iloc[0]["direction"] == "CE"
      and trA.iloc[0]["day"] == BRK and pd.Timestamp(trA.iloc[0]["signal_time"]).strftime("%H:%M") == "09:50", f"signals={info['signals']} trades={len(trA)}")
t = trA.iloc[0]
check("e2e: all four targets hit on a steadily rising path, exit T4", bool(t.hit_t1 and t.hit_t2 and t.hit_t3 and t.hit_t4) and t.exit_reason == "T4", f"{t.legs}")
check("e2e: positive R, costs reduce it, stop within the 25% cap", t.r_gross > 1 and t.r_net < t.r_gross and t.risk_pct <= 0.25, f"r_gross={t.r_gross} r_net={t.r_net} risk={t.risk_pct}")
check("e2e: delta 0.35-0.65, nearest expiry 13 days out", 0.35 <= abs(t.delta) <= 0.65 and t.dte == 13, f"delta={t.delta} dte={t.dte}")
check("e2e: variant B equals A on a path that never retraces", abs(trB.iloc[0].r_gross - t.r_gross) < 1e-9)
check("e2e: underlying labels recorded (any vs before stop)", t.u_outcome.startswith("T") and bool(t.u_reach_t2_any) and bool(t.u_reach_t2_before_stop))
tr2, rej2, _ = R.run_backtest(["SYN"], per, [S.Params(max_stop_frac=0.001)], {"SYN": 100}, log=LOG)
check("e2e: stop wider than the cap is skipped and logged", len(tr2[(1.5, "A")]) == 0 and rej2["reason"].str.contains("stop_over_cap").any())
tr3, _, _ = R.run_backtest(["SYN"], per, [S.Params(rvol_min=5.0)], {"SYN": 100}, log=LOG)
check("e2e: higher volume threshold gives no signal", len(tr3[(1.5, "A")]) == 0)

trT, _, _ = R.run_backtest(["SYN"], per, [S.Params(exit_scheme="t1t2")], {"SYN": 100}, log=LOG)
tt = trT[(1.5, "A")].iloc[0]
check("e2e t1t2: legs T1 then T2, only two targets, exit T2, positive R", [q.split("@")[0] for q in tt.legs.split(";")] == ["T1", "T2"] and tt.exit_reason == "T2" and not tt.hit_t3 and not tt.hit_t4
      and pd.isna(tt.t3) and pd.isna(tt.t4) and tt.r_gross > 0 and tt.exit_scheme == "t1t2", f"{tt.legs} r_gross={tt.r_gross}")

# execution convention (audit item 1)
store = D.OptionStore("SYN")
Tsig = pd.Timestamp(t.signal_time)
rowsN = store.get(t.contract, BRK)
open_next = float(rowsN[rowsN["ts"] == Tsig + pd.Timedelta(minutes=1)]["o"].iloc[0]); close_T = float(rowsN[rowsN["ts"] == Tsig]["c"].iloc[0])
check("execution next_open: entry = option OPEN of the minute after the signal; entry_ts = T+1", abs(t.entry - open_next) < 0.051 and pd.Timestamp(t.entry_ts) == Tsig + pd.Timedelta(minutes=1) and t.execution_mode == "next_open",
      f"entry {t.entry} vs open {open_next:.2f}")
trL, _, _ = R.run_backtest(["SYN"], per, [S.Params(execution_mode="legacy_close")], {"SYN": 100}, log=LOG)
tL = trL[(1.5, "A")].iloc[0]
check("execution legacy_close: entry = option CLOSE of the signal-minute candle", abs(tL.entry - close_T) < 0.051 and tL.execution_mode == "legacy_close", f"entry {tL.entry} vs close {close_T:.2f}")

# ATR5 modes (the 'session' reset leaves early-morning signals without a stop, as seen in the external patch's results)
cs, cc = S.build_ctx("SYN", atr5_mode="session"), S.build_ctx("SYN", atr5_mode="carry")
day2 = cs.f5_by_day[days[5]]
check("ATR5 carry: defined from the first bar of a later session; session mode: undefined until the 14th bar", cc.f5_by_day[days[5]]["atr5"].iloc[0] == cc.f5_by_day[days[5]]["atr5"].iloc[0]
      and day2["atr5"].iloc[:13].isna().all() and day2["atr5"].iloc[13] == day2["atr5"].iloc[13])
trS, rejS, infoS = R.run_backtest(["SYN"], per, [S.Params(atr5_mode="session")], {"SYN": 100}, log=LOG)
check("ATR5 session: the 09:50 signal exists but cannot be traded (no_atr5)", infoS["signals"] == 1 and len(trS[(1.5, "A")]) == 0 and rejS["reason"].str.contains("no_atr5").any())

# data completeness (audit item 4)
raw = pd.read_parquet(f"{TMP}/stock_1m/SYN/a.parquet")
drop = raw["ts"] == datetime(BRK.year, BRK.month, BRK.day, 9, 47)
raw[~drop].to_parquet(f"{TMP}/stock_1m/SYN/a.parquet", index=False)
cx = S.build_ctx("SYN"); nifty5, _ = S.index_context("NIFTY")
sg_c, rj_c = S.find_signals(cx, nifty5, S.Params())
sg_o, _ = S.find_signals(cx, nifty5, S.Params(require_complete=False))
check("completeness: a 5-minute bar missing a minute is not used; the rejection is logged", all(pd.Timestamp(x["T"]).strftime("%H:%M") != "09:50" for x in sg_c if x["day"] == BRK)
      and any(str(r["reason"]).startswith("incomplete_5m_bars") for r in rj_c) and any(pd.Timestamp(x["T"]).strftime("%H:%M") == "09:50" for x in sg_o))
drop = raw["ts"].between(datetime(BRK.year, BRK.month, BRK.day, 9, 20), datetime(BRK.year, BRK.month, BRK.day, 9, 22))
raw[~drop].to_parquet(f"{TMP}/stock_1m/SYN/a.parquet", index=False)
cx = S.build_ctx("SYN"); sg_c, rj_c = S.find_signals(cx, nifty5, S.Params())
check("completeness: opening range missing 3 minutes -> day skipped and logged", all(x["day"] != BRK for x in sg_c) and any(r["reason"] == "incomplete_opening_range" for r in rj_c))
raw.to_parquet(f"{TMP}/stock_1m/SYN/a.parquet", index=False)

# frozen configuration is enforced (audit item 6)
sys.argv = ["bt_run.py", "freeze", "--m", "1.5", "--variant", "A", "--hyp", "pullback", "--slip", "0.01", "--lots", "8", "--intrabar-mode", "optimistic"]
R.main()
fz = __import__("json").load(open(f"{TMP}/backtest/frozen.json"))
check("freeze stores the full configuration", fz["slip"] == 0.01 and fz["lots"] == 8 and fz["intrabar_mode"] == "optimistic" and fz["hyp"] == "pullback" and fz["execution_mode"] == "next_open" and fz["exit_scheme"] == "t4")
sys.argv = ["bt_run.py", "--period", "val", "--slip", "0.02"]
try:
    R.main(); check("validation refuses options that conflict with the frozen configuration", False)
except SystemExit as e:
    check("validation refuses options that conflict with the frozen configuration", "conflict" in str(e))

summ = ST.summarize(pd.concat([trA, trA.assign(day=days[1], r_net=-1.0, r_gross=-1.0, signal_time=pd.Timestamp("2025-01-02 10:00"), exit_reason="SL")]))
check("stats: summary, day-clustered CI, both drawdowns and the stop measure", summ["n"] == 2 and summ["ci_lo"] == summ["ci_lo"] and "daily_aggregate_max_dd_r" in summ and "sl_then_t2_any_rate" in summ, str({k: summ[k] for k in ("avg_r_net", "ci_lo", "ci_hi")}))
check("stats: max drawdown on a simple series", abs(ST.max_drawdown(np.array([1, -2, -1, 3])) - 3) < 1e-9)

print(f"\n{sum(ok_all)}/{len(ok_all)} checks passed"); shutil.rmtree(TMP, ignore_errors=True)
sys.exit(0 if all(ok_all) else 1)
