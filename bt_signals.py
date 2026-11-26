"""Stage 2a: per-stock context and signal detection (spec section 3). Uses only completed 5-minute bars."""
from dataclasses import dataclass
from datetime import time

import numpy as np
import pandas as pd

import bt_data as D


@dataclass
class Params:
    or_minutes: int = 15          # opening range length
    rvol_min: float = 1.5         # relative volume threshold
    ext_cap: float = 0.35         # max extension beyond the range, in daily ATRs
    atr_floor: float = 0.015      # minimum daily ATR(14) as a share of price
    stop_mult: float = 1.5        # minimum stop distance in 5-minute ATRs (calibrated on Development only)
    variant: str = "A"            # trailing-stop variant A or B (spec 12.2)
    slippage: float = 0.02        # per side, share of premium
    lots: int = 4                 # smallest size that allows 50/25/25 in whole lots
    roll_days: int = 2            # use the next expiry when fewer calendar days remain
    delta_lo: float = 0.35
    delta_hi: float = 0.65
    min_premium: float = 2.0
    max_stop_frac: float = 0.25   # option stop never wider than this share of entry premium
    min_rr_t2: float = 1.5
    scan_from: str = "09:30"
    scan_to: str = "12:00"
    eod: str = "14:59"
    hyp: str = "breakout"         # "breakout" (baseline) or "pullback" (H2)
    pb_tol: float = 0.1           # H2: retest tolerance in 5-min ATRs (fixed, not tuned)
    pb_thrust: float = 1.0        # H2: required thrust above the zone in 5-min ATRs (fixed)
    pb_wait: int = 12             # H2: bars allowed between setup and pullback (fixed)
    execution_mode: str = "next_open"    # entry at the OPEN of the 1-minute candle after the signal bar; "legacy_close" for comparison
    intrabar_mode: str = "conservative"  # same-candle stop handling: "conservative" or "optimistic"
    atr5_mode: str = "carry"             # 14-bar ATR over 5-min bars: "carry" across sessions (intraday TR only) or "session" (resets daily)
    exit_scheme: str = "t4"              # "t4": 50/25/25 at T2/T3/T4 (baseline, H2); "t1t2": 50% at T1 then stop to entry, 50% at T2 (H3)
    require_complete: bool = True        # signal bars need all five 1-minute candles; opening range may miss at most one


class Ctx:
    pass


def _tm(s):
    h, m = s.split(":")
    return time(int(h), int(m))


def build_ctx(sym, data_dir=None, atr5_mode="carry"):
    m1 = D.load_stock_1m(sym, data_dir)
    if m1.empty:
        return None
    m1 = D.add_vwap(m1)
    m1["bar"] = m1["ts"].dt.floor("5min")
    f5 = D.to_5m(m1)
    sess = m1[m1["ts"].dt.time < time(15, 30)]
    f5["vwap"] = f5["ts"].map(sess.groupby("bar")["vwap"].last())
    daily = D.daily_table(m1)
    rv = D.rvol_table(m1)
    pos = {d: i for i, d in enumerate(rv.index)}
    m_end = ((f5["end"].dt.hour * 60 + f5["end"].dt.minute) - (9 * 60 + 15) - 1).to_numpy()
    di = f5["day"].map(pos).to_numpy()
    f5["rvol"] = rv.to_numpy()[di, m_end]
    prev_c = f5.groupby("day")["c"].shift(1)
    tr = pd.concat([f5["h"] - f5["l"], (f5["h"] - prev_c).abs(), (f5["l"] - prev_c).abs()], axis=1).max(axis=1)
    if atr5_mode == "session":
        f5["atr5"] = tr.groupby(f5["day"]).transform(lambda x: x.rolling(14, min_periods=14).mean())   # undefined before 10:25
    else:
        f5["atr5"] = tr.rolling(14, min_periods=14).mean()       # intraday true range only (no overnight gaps), carried across sessions
    c = Ctx()
    c.sym, c.m1, c.f5, c.daily = sym, m1, f5, daily
    c.excl = D.exclusion_days(daily)
    c.orr_cache = {}
    c.rv = rv
    c.m1c = m1.set_index("ts")["c"]
    c.f5_by_day = {d: g.reset_index(drop=True) for d, g in f5.groupby("day")}
    c.m1_by_day = {d: g for d, g in m1.groupby("day")}
    c.liq = pd.Series(dtype=float)
    return c


def index_context(sym="NIFTY", data_dir=None):
    """Nifty 5-minute close and its session reference level at each bar end. The index has no traded volume in the
    data, so the reference is the time-weighted average of the typical price (VWAP proxy); documented in the spec."""
    m1 = D.load_stock_1m(sym, data_dir)
    if m1.empty:
        return None, "missing"
    tp = (m1["h"] + m1["l"] + m1["c"]) / 3
    has_vol = float(m1["v"].fillna(0).sum()) > 0
    if has_vol:
        ref = D.add_vwap(m1)["vwap"]
    else:
        g = tp.groupby(m1["day"])
        ref = g.cumsum() / (g.cumcount() + 1)
    m1 = m1.assign(ref=ref, bar=m1["ts"].dt.floor("5min"))
    sess = m1[m1["ts"].dt.time < time(15, 30)]
    g = sess.groupby("bar")
    out = pd.DataFrame({"c": g["c"].last(), "ref": g["ref"].last()})
    out.index = out.index + pd.Timedelta(minutes=5)
    return out, ("vwap" if has_vol else "twap")


def find_signals(ctx, nifty5, p):
    """Returns (signals, rejects). A signal = first completed 5-min bar per stock/direction/day passing every rule."""
    sigs, rej = [], []
    lo, hi = _tm(p.scan_from), _tm(p.scan_to)
    for day, g in ctx.f5.groupby("day", sort=True):
        if day in ctx.excl or day not in ctx.daily.index:
            continue
        d = ctx.daily.loc[day]
        if not (d["atr14_pct"] >= p.atr_floor):
            continue
        om = ctx.m1_by_day[day]
        mi = om["ts"].dt.hour * 60 + om["ts"].dt.minute - 555
        om = om[(mi >= 0) & (mi < p.or_minutes)]
        if om.empty or (p.require_complete and len(om) < p.or_minutes - 1):
            rej.append(dict(symbol=ctx.sym, day=day, direction="-", T=pd.NaT, stage="data", reason="incomplete_opening_range"))
            continue
        or_high, or_low = om["h"].max(), om["l"].min()
        g = g[(g["end"].dt.time >= lo) & (g["end"].dt.time <= hi)
              & ((g["ts"].dt.hour * 60 + g["ts"].dt.minute - 555) >= p.or_minutes)]
        if p.require_complete:
            n_bad = int((g["n"] < 5).sum())
            if n_bad:
                rej.append(dict(symbol=ctx.sym, day=day, direction="-", T=pd.NaT, stage="data", reason=f"incomplete_5m_bars({n_bad})"))
            g = g[g["n"] == 5]
        g = g.reset_index(drop=True)
        if g.empty:
            continue
        nf = nifty5.reindex(g["end"]) if nifty5 is not None else None
        n_c = nf["c"].to_numpy() if nf is not None else np.full(len(g), np.nan)
        n_r = nf["ref"].to_numpy() if nf is not None else np.full(len(g), np.nan)
        c, vw, rv = g["c"].to_numpy(), g["vwap"].to_numpy(), g["rvol"].to_numpy()
        atr_d = d["atr14"]
        for direction in ("CE", "PE"):
            if direction == "CE":
                brk, trend, nift, ext = c > or_high, c > vw, n_c > n_r, (c - or_high) <= p.ext_cap * atr_d
            else:
                brk, trend, nift, ext = c < or_low, c < vw, n_c < n_r, (or_low - c) <= p.ext_cap * atr_d
            vol = rv >= p.rvol_min
            ok = brk & trend & vol & nift & ext
            idx = np.flatnonzero(ok)
            if idx.size:
                i = int(idx[0])
                r = g.iloc[i]
                sigs.append(dict(symbol=ctx.sym, day=day, direction=direction, T=r["end"], bar_start=r["ts"],
                                 c=float(r["c"]), bar_h=float(r["h"]), bar_l=float(r["l"]), vwap=float(r["vwap"]),
                                 rvol=float(r["rvol"]), atr5=float(r["atr5"]), atr_d=float(atr_d),
                                 or_high=float(or_high), or_low=float(or_low)))
            elif brk.any():
                i = int(np.flatnonzero(brk)[0])
                fails = [n for n, a in (("vwap", trend), ("rvol", vol), ("nifty", nift), ("extended", ext)) if not a[i]]
                rej.append(dict(symbol=ctx.sym, day=day, direction=direction, T=g.loc[i, "end"],
                                stage="trigger", reason="+".join(fails) or "none"))
    return sigs, rej


def find_pullback_signals(ctx, nifty5, p):
    """H2 (spec section 14): wait for a retest of max(OR high, VWAP) after a volume-backed breakout that showed a thrust."""
    sigs, rej = [], []
    lo, hi = _tm(p.scan_from), _tm(p.scan_to)
    for day, g in ctx.f5.groupby("day", sort=True):
        if day in ctx.excl or day not in ctx.daily.index:
            continue
        d = ctx.daily.loc[day]
        if not (d["atr14_pct"] >= p.atr_floor):
            continue
        om = ctx.m1_by_day[day]
        mi = om["ts"].dt.hour * 60 + om["ts"].dt.minute - 555
        om = om[(mi >= 0) & (mi < p.or_minutes)]
        if om.empty or (p.require_complete and len(om) < p.or_minutes - 1):
            rej.append(dict(symbol=ctx.sym, day=day, direction="-", T=pd.NaT, stage="data", reason="incomplete_opening_range"))
            continue
        or_high, or_low = om["h"].max(), om["l"].min()
        g = g[(g["end"].dt.time <= hi) & ((g["ts"].dt.hour * 60 + g["ts"].dt.minute - 555) >= p.or_minutes)]
        if p.require_complete:
            n_bad = int((g["n"] < 5).sum())
            if n_bad:
                rej.append(dict(symbol=ctx.sym, day=day, direction="-", T=pd.NaT, stage="data", reason=f"incomplete_5m_bars({n_bad})"))
            g = g[g["n"] == 5]
        g = g.reset_index(drop=True)
        if len(g) < 3:
            continue
        nf = nifty5.reindex(g["end"]) if nifty5 is not None else None
        n_c = nf["c"].to_numpy() if nf is not None else np.full(len(g), np.nan)
        n_r = nf["ref"].to_numpy() if nf is not None else np.full(len(g), np.nan)
        c, h, l = g["c"].to_numpy(), g["h"].to_numpy(), g["l"].to_numpy()
        vw, rv, a5 = g["vwap"].to_numpy(), g["rvol"].to_numpy(), g["atr5"].to_numpy()
        for direction in ("CE", "PE"):
            long_ = direction == "CE"
            if long_:
                setup = (c > or_high) & (c > vw) & (rv >= p.rvol_min) & (n_c > n_r)
            else:
                setup = (c < or_low) & (c < vw) & (rv >= p.rvol_min) & (n_c < n_r)
            idx = np.flatnonzero(setup)
            if not idx.size:
                continue
            s = int(idx[0])
            outcome = "no_pullback"
            for j in range(s + 1, min(len(g), s + 1 + p.pb_wait)):
                if not (a5[j] == a5[j] and a5[j] > 0):
                    continue
                tol = p.pb_tol * a5[j]
                if long_:
                    Z = max(or_high, vw[j]); touch = l[j] <= Z + tol; held = c[j] > Z
                    thrust = (h[s:j].max() - Z) / a5[j]; nift = n_c[j] > n_r[j]
                else:
                    Z = min(or_low, vw[j]); touch = h[j] >= Z - tol; held = c[j] < Z
                    thrust = (Z - l[s:j].min()) / a5[j]; nift = n_c[j] < n_r[j]
                if not touch:
                    continue
                if not held:
                    outcome = "failed_retest"
                    break
                if thrust >= p.pb_thrust and nift:
                    if g.loc[j, "end"].time() >= lo:
                        r = g.iloc[j]
                        sigs.append(dict(symbol=ctx.sym, day=day, direction=direction, T=r["end"], bar_start=r["ts"],
                                         c=float(r["c"]), bar_h=float(r["h"]), bar_l=float(r["l"]), vwap=float(r["vwap"]),
                                         rvol=float(rv[s]), atr5=float(a5[j]), atr_d=float(d["atr14"]),
                                         or_high=float(or_high), or_low=float(or_low),
                                         setup_T=g.loc[s, "end"], wait_bars=int(j - s), thrust=float(thrust)))
                        outcome = "signal"
                    break
            if outcome != "signal":
                rej.append(dict(symbol=ctx.sym, day=day, direction=direction, T=g.loc[s, "end"], stage="setup", reason=outcome))
    return sigs, rej
