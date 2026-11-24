"""Offline tests for the Stage-1 backtest data layer, pricing and costs (synthetic data; no network)."""
import os, shutil, sys, tempfile
from datetime import date, datetime, timedelta
import numpy as np, pandas as pd

TMP = tempfile.mkdtemp()
os.environ["DATA_DIR"] = TMP
import bt_data as D, bt_pricing as P, bt_costs as C
passed = []
def check(name, cond, extra=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  {extra}" if extra else ""))
    passed.append(bool(cond))

# ---------------- costs: reproduce a published worked example (NIFTY option round trip, Rs 63.95) -----------------
d = date(2026, 6, 1)
t = C.trade_cost([("BUY", 6500, d), ("SELL", 7150, d)])
check("costs: published example total ~63.9", abs(t["total"] - 63.95) < 0.15, f"got {t['total']:.2f}")
t2 = C.trade_cost([("BUY", 50000, d), ("SELL", 60000, d)])
check("costs: exchange charge on 50k/60k = 39.08 (Zerodha example)", abs(t2["exchange"] - 39.08) < 0.5, f"got {t2['exchange']:.2f}")
check("costs: STT 0.15% after Apr-2026, 0.10% before", abs(C.order_cost("SELL", 10000, date(2026, 5, 1))["stt"] - 15) < 1e-9
      and abs(C.order_cost("SELL", 10000, date(2025, 5, 1))["stt"] - 10) < 1e-9)
check("costs: no STT on buy, no stamp on sell", C.order_cost("BUY", 1e4, d)["stt"] == 0 and C.order_cost("SELL", 1e4, d)["stamp"] == 0)
check("costs: 4 orders = Rs 80 brokerage", abs(C.trade_cost([("BUY", 1e3, d)] + [("SELL", 1e3, d)] * 3)["brokerage"] - 80) < 1e-9)

# ---------------- pricing ----------------
S, K, T = 1000.0, 1000.0, 20 / 365
pc, pp = P.bs_price(S, K, T, 0.3, "CE"), P.bs_price(S, K, T, 0.3, "PE")
check("bs: put-call parity", abs((pc - pp) - (S - K * np.exp(-P.B.RISK_FREE * T))) < 1e-6)
iv = P.implied_vol(pc, S, K, T, "CE")
check("bs: implied vol round trip", abs(iv - 0.3) < 1e-4, f"iv={iv:.5f}")
check("bs: ATM call delta ~0.5-0.56", 0.5 < P.delta(S, K, T, 0.3, "CE") < 0.56)
check("bs: below-intrinsic price gives NaN IV", P.implied_vol(0.5, 1100, 1000, T, "CE") != P.implied_vol(0.5, 1100, 1000, T, "CE"))
up = P.reprice_at_spot(1020, pc, S, K, T, "CE")
check("bs: repricing a call at higher spot raises price", up is not None and up > pc)
check("bs: time to expiry", abs(P.time_to_expiry(datetime(2026, 11, 20, 15, 30), date(2026, 11, 23)) * 365 - 3) < 1e-6)

# ---------------- synthetic stock data ----------------
def sessions(n, start=date(2025, 1, 1)):
    out, d0 = [], start
    while len(out) < n:
        if d0.weekday() < 5: out.append(d0)
        d0 += timedelta(days=1)
    return out
days = sessions(40)
rows = []
for k, dd in enumerate(days):
    px = 100.0 * (1.0 if k < 30 else 0.5)                     # 50% price drop on session 30 = unadjusted split
    vol = 100 if k != 35 else 200
    for m in range(-15, 376):                                   # includes pre-open minutes (before 09:15)
        ts = datetime(dd.year, dd.month, dd.day, 9, 15) + timedelta(minutes=m)
        o = px if m >= 0 else np.nan
        rows.append([ts, o, o * 1.001 if m >= 0 else np.nan, o * 0.999 if m >= 0 else np.nan, o, vol, None])
st = pd.DataFrame(rows, columns=["ts", "o", "h", "l", "c", "v", "oi"])
st.loc[100, ["h", "l"]] = [90.0, 110.0]                         # one bad OHLC row (high < low)
st = pd.concat([st, st.iloc[500:503]])                          # duplicates
os.makedirs(f"{TMP}/stock_1m/TEST", exist_ok=True)
st.to_parquet(f"{TMP}/stock_1m/TEST/a.parquet", index=False)
s = D.load_stock_1m("TEST")
t_ = s["ts"].dt.time
check("loader: no rows before 09:15", (t_ >= datetime(2000, 1, 1, 9, 15).time()).all())
check("loader: no NaN prices, no duplicate timestamps", s[D.OHLC].notna().all().all() and not s["ts"].duplicated().any())
check("loader: bad OHLC row dropped", ((s["h"] < s["l"]).sum() == 0) and len(s) == 40 * 376 - 1, f"rows={len(s)}")
f = D.to_5m(s)
check("5m: 75 bars/day, each complete 5 min after start", f.groupby("day").size().eq(75).all() and (f["end"] - f["ts"]).eq(pd.Timedelta(minutes=5)).all())
v = D.add_vwap(s)
check("vwap: constant price gives vwap ~ typical price", abs(v["vwap"].iloc[10] - 100.0) < 0.5)
dt = D.daily_table(s)
dt2 = D.daily_table(s[s["day"] < days[-1]])
check("ATR: no look-ahead (last session does not change its own ATR)", abs(dt["atr14"].iloc[-1] - D.daily_table(s.assign(h=np.where(s["day"] == days[-1], s["h"] * 2, s["h"])))["atr14"].iloc[-1]) < 1e-9)
ex = D.exclusion_days(dt)
check("split gap: 5 sessions before to 20 after excluded", len(ex) == 15 and days[25] in ex and days[24] not in ex and days[39] in ex, f"n={len(ex)}")
r = D.rvol_table(s, window=20)
check("RVOL: 2x volume session gives ~2.0 at minute 14", abs(r.loc[days[35], 14] - 2.0) < 0.05 and abs(r.loc[days[30], 14] - 1.0) < 1e-9, f"{r.loc[days[35],14]:.3f}")
orr = D.opening_range(s)
check("opening range: first 15 minutes high/low", abs(orr.loc[days[0], "or_high"] - 100.1) < 1e-6 and abs(orr.loc[days[0], "or_low"] - 99.9) < 1e-6)

# ---------------- synthetic option data ----------------
exp = date(2025, 2, 27); gs = "NSE-TEST-27Feb25-100-CE"
def opt_rows(contract, dd, minutes, oi_pattern):
    out = []
    for m, oi in zip(minutes, oi_pattern):
        ts = datetime(dd.year, dd.month, dd.day, 9, 15) + timedelta(minutes=m)
        out.append([ts, contract, 100.0, "CE", 5.0, 5.1, 4.9, 5.0, 1000, oi])
    return out
rows = opt_rows(gs, days[0], [5, 6, 20, 21], [100.0, None, None, None])           # 09:20 oi, 09:21 filled, 09:35/36 too old
rows += opt_rows(gs, days[1], [0, 0], [50.0, 50.0])                               # duplicate timestamp
oo = pd.DataFrame(rows, columns=["ts", "contract", "strike", "typ", "o", "h", "l", "c", "v", "oi"])
os.makedirs(f"{TMP}/opt_1m/TEST", exist_ok=True)
oo.to_parquet(f"{TMP}/opt_1m/TEST/{exp}.parquet", index=False)
store = D.OptionStore("TEST")
g0 = store.get(gs, days[0]).set_index("ts")
check("OI fill: within 5 minutes filled, beyond 5 minutes not",
      g0["oi_f"].iloc[0] == 100 and g0["oi_f"].iloc[1] == 100 and np.isnan(g0["oi_f"].iloc[2]) and np.isnan(g0["oi_f"].iloc[3]))
check("OI fill: never across days", np.isnan(store.get(gs, days[1])["oi_f"].iloc[0]) is False or store.get(gs, days[1])["oi"].iloc[0] == 50)
check("options: duplicates removed, expiry parsed", len(store.get(gs, days[1])) == 1 and store.df["expiry"].iloc[0] == exp)
check("options: has() / missing day detected", store.has(gs, days[0]) and not store.has(gs, days[5]))

# ---------------- on-demand fetch with a fake client ----------------
class FakeClient:
    n = 0
    def const(self, name): return name
    def call(self, method, **kw):
        FakeClient.n += 1
        d0 = datetime.fromisoformat(kw["start_time"]).date(); d1 = datetime.fromisoformat(kw["end_time"]).date()
        rows, d = [], d0
        while d <= d1:
            if d.weekday() < 5:
                for m in range(0, 30):
                    rows.append([f"{d}T{9 + (15 + m) // 60:02d}:{(15 + m) % 60:02d}:00", 5, 5.1, 4.9, 5.0, 100, 10.0])
            d += timedelta(days=1)
        return {"candles": rows}
store2 = D.OptionStore("TEST", client=FakeClient())
want = [days[5], days[6], days[7]]
n1 = store2.ensure(gs, want); calls1 = FakeClient.n
check("on-demand: missing days fetched in one merged request and cached", n1 == 3 and calls1 == 1 and all(store2.has(gs, x) for x in want), f"calls={calls1}")
store2.ensure(gs, want)
check("on-demand: second ensure does not refetch", FakeClient.n == calls1)
store3 = D.OptionStore("TEST", client=FakeClient())
check("on-demand: fetched data persisted and reloaded in a new session", store3.has(gs, days[5]) and FakeClient.n == calls1)
cov = store.coverage_by_day()
check("liquidity: coverage computed and shifted (point-in-time)", len(cov) >= 1 and store.liquidity().iloc[0] != store.liquidity().iloc[0])

print(f"\n{sum(passed)}/{len(passed)} checks passed")
shutil.rmtree(TMP, ignore_errors=True)
sys.exit(0 if all(passed) else 1)
