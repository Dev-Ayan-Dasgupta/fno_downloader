"""Stage 2b: contract selection, levels, trade simulation, costs (spec sections 4-7 and 12)."""
import math
from datetime import time

import numpy as np
import pandas as pd

import bt_costs as C
import bt_data as D
import bt_pricing as P

TICK = 0.05


def _r(x):
    return round(round(x / TICK) * TICK, 2)


def _tm(s):
    h, m = s.split(":")
    return time(int(h), int(m))


def select_contract(sig, ctx, store, p):
    """Select contract using a synchronized post-signal execution snapshot.

    next_open: signal is known at T; enter at the next 1m candle open (T + 1 minute).
    legacy_close: retain the old close-at-T convention for comparison only.
    """
    day, T, typ = sig["day"], sig["T"], sig["direction"]
    liq = ctx.liq.get(day, np.nan)
    if not (liq >= 0.8):
        return None, "liquidity_gate"
    store.contracts_index()
    exp = next((e for e in store.expiries if (e - day).days >= p.roll_days), None)
    if exp is None:
        return None, "no_expiry"
    if p.execution_mode == "next_open":
        entry_ts = T + pd.Timedelta(minutes=1)
        price_ts = entry_ts
        spot_rows = ctx.m1_by_day[day]
        spot_rows = spot_rows[spot_rows["ts"] == entry_ts]
        if spot_rows.empty:
            return None, "no_next_minute_spot"
        spot, price_col = float(spot_rows["o"].iloc[0]), "o"
    elif p.execution_mode == "legacy_close":
        # The old candle is timestamped at its start T; its close is only known at T+1.
        price_ts = T
        entry_ts = T + pd.Timedelta(minutes=1)
        spot = float(ctx.m1c.get(T, np.nan))
        if not np.isfinite(spot):
            return None, "no_spot_candle"
        price_col = "c"
    else:
        return None, f"unknown_execution_mode:{p.execution_mode}"
    strikes = store.strikes.get((exp, typ), [])
    if not strikes:
        return None, "no_strikes"
    i = int(np.argmin([abs(k - spot) for k in strikes]))
    best, why = None, {}
    for j in (i - 1, i, i + 1):
        if j < 0 or j >= len(strikes):
            continue
        K = strikes[j]
        cname = store.cidx_get(exp, K, typ)
        if cname is None:
            why["no_contract"] = why.get("no_contract", 0) + 1
            continue
        if not store.has(cname, day):
            store.ensure(cname, [day])
        rows = store.get(cname, day)
        row = rows[rows["ts"] == price_ts]
        if row.empty:
            why["no_entry_candle"] = why.get("no_entry_candle", 0) + 1
            continue
        price = float(row[price_col].iloc[0])
        if not np.isfinite(price) or price < p.min_premium:
            why["min_premium"] = why.get("min_premium", 0) + 1
            continue
        Ty = P.time_to_expiry(entry_ts, exp)
        iv = P.implied_vol(price, spot, K, Ty, typ)
        if not np.isfinite(iv):
            why["iv_nan"] = why.get("iv_nan", 0) + 1
            continue
        dl = P.delta(spot, K, Ty, iv, typ)
        if not (p.delta_lo <= abs(dl) <= p.delta_hi):
            why["delta"] = why.get("delta", 0) + 1
            continue
        score = abs(abs(dl) - 0.5)
        if best is None or score < best["score"]:
            best = dict(contract=cname, expiry=exp, strike=K, price=price, iv=iv,
                        delta=dl, Ty=Ty, spot=spot, score=score, entry_ts=entry_ts,
                        price_field=price_col)
    if best is None:
        return None, "no_eligible_contract:" + ",".join(f"{k}={v}" for k, v in sorted(why.items()))
    return best, ""


def _swing(day_bars, sig_start, side):
    """Most recent confirmed 2-bar pivot (low for CE, high for PE) before the signal bar."""
    b = day_bars[day_bars["ts"] < sig_start].reset_index(drop=True)
    x = b["l"].to_numpy() if side == "CE" else b["h"].to_numpy()
    for j in range(len(b) - 2, 0, -1):
        if side == "CE" and x[j] < x[j - 1] and x[j] < x[j + 1]:
            return float(x[j])
        if side == "PE" and x[j] > x[j - 1] and x[j] > x[j + 1]:
            return float(x[j])
    return np.nan


def make_levels(sig, sel, ctx, p, stop_mult):
    """Dynamic stop (spec 12.1) and 1R-4R targets on the underlying, repriced to option premiums."""
    side, spot, atr5 = sig["direction"], sel["spot"], sig["atr5"]
    if not (atr5 == atr5 and atr5 > 0):
        return None, "no_atr5"
    swing = _swing(ctx.f5_by_day[sig["day"]], sig["bar_start"], side)
    if side == "CE":
        cands = [sig["bar_l"], sig["or_low"], sig["vwap"], swing]
        ok = [x for x in cands if x == x and spot - x >= stop_mult * atr5]
        stop_u = max(ok) if ok else spot - stop_mult * atr5
        sign = 1
    else:
        cands = [sig["bar_h"], sig["or_high"], sig["vwap"], swing]
        ok = [x for x in cands if x == x and x - spot >= stop_mult * atr5]
        stop_u = min(ok) if ok else spot + stop_mult * atr5
        sign = -1
    Ru = abs(spot - stop_u)
    if Ru <= 0:
        return None, "zero_risk"
    px = lambda L: P.bs_price(L, sel["strike"], sel["Ty"], sel["iv"], side)
    entry = sel["price"]
    stop = _r(px(stop_u))
    ks = (1, 2) if p.exit_scheme == "t1t2" else (1, 2, 3, 4)
    nan = float("nan")
    tg = [_r(px(spot + sign * k * Ru)) for k in ks] + [nan] * (4 - len(ks))
    tg_u = [spot + sign * k * Ru for k in ks] + [nan] * (4 - len(ks))
    risk = entry - stop
    if stop <= 0 or risk <= 0:
        return None, "bad_stop_price"
    if risk / entry > p.max_stop_frac:
        return None, "stop_over_cap"
    if not all(tg[i] < tg[i + 1] for i in range(len(ks) - 1)):
        return None, "targets_not_increasing"
    if (tg[1] - entry) / risk < p.min_rr_t2:
        return None, "rr_t2"
    return dict(stop_u=stop_u, Ru=Ru, tg_u=tg_u, entry=entry, stop=stop, tg=tg, risk=risk), ""


def simulate(entry, stop0, tg, ts, o, h, l, c, variant, eod_ts,
            intrabar_mode="conservative", scheme="t4"):
    """Walk 1m option candles from entry onward.

    conservative: a stop activated by a target inside a candle is assumed hit if that
    candle's low crosses it. optimistic: newly activated stops apply from next candle.
    Initial stop wins same-candle ties in both modes.
    """
    if intrabar_mode not in ("conservative", "optimistic"):
        raise ValueError(f"Unknown intrabar_mode: {intrabar_mode}")
    stop, remaining = stop0, 1.0
    legs, hit = [], [False] * 4
    min_low = max_high = entry
    last = None
    for i in range(len(ts)):
        if ts[i] > eod_ts:
            break
        last = i
        min_low, max_high = min(min_low, l[i]), max(max_high, h[i])
        if l[i] <= stop:
            legs.append(("SL" if stop <= stop0 + 1e-9 else "TRAIL", min(stop, o[i]), remaining))
            remaining = 0.0
            break
        for k in range(2 if scheme == "t1t2" else 4):
            if hit[k]:
                continue
            if h[i] < tg[k] + TICK:
                break
            hit[k] = True
            if scheme == "t1t2":
                legs.append((f"T{k + 1}", tg[k], 0.5))      # 50% at T1, then the stop moves to entry; 50% at T2
                remaining -= 0.5
                if k == 0:
                    stop = max(stop, entry)
            elif k == 0:
                if variant == "A":
                    stop = max(stop, entry)
            else:
                frac = (0.5, 0.25, 0.25)[k - 1]
                legs.append((f"T{k + 1}", tg[k], frac))
                remaining -= frac
                stop = max(stop, tg[k - 1])
            if intrabar_mode == "conservative" and remaining > 1e-9 and l[i] <= stop:
                legs.append(("TRAIL" if stop > stop0 + 1e-9 else "SL", min(stop, o[i]), remaining))
                remaining = 0.0
                break
        if remaining <= 1e-9:
            remaining = 0.0
            break
    if remaining > 1e-9:
        if last is None:
            legs.append(("EOD_NOCANDLE", entry, remaining))
        else:
            legs.append(("EOD", float(c[last]), remaining))
    return dict(legs=legs, mae_px=entry - min_low, mfe_px=max_high - entry,
                hit=hit, stale=(last is None))


def underlying_outcome(sig, lv, ctx, eod_t, entry_ts=None):
    """Separate T2 reached at any time from T2 reached before the underlying stop.

    Same-minute target/stop touches are ambiguous and are treated conservatively:
    T2 does not count as preceding the stop when both occur in one 1m candle.
    """
    entry_ts = entry_ts if entry_ts is not None else sig["T"]
    m1 = ctx.m1_by_day[sig["day"]]
    w = m1[(m1["ts"] >= entry_ts) & (m1["ts"].dt.time <= eod_t)]
    if w.empty:
        return dict(outcome="none", reach_t2_any=False, reach_t2_before_stop=False,
                    stop_hit=False)
    h, l = w["h"].to_numpy(), w["l"].to_numpy()
    long_ = sig["direction"] == "CE"
    stop_idx = np.flatnonzero(l <= lv["stop_u"]) if long_ else np.flatnonzero(h >= lv["stop_u"])
    stop_i = int(stop_idx[0]) if stop_idx.size else len(h)
    target_idx = []
    for level in lv["tg_u"]:
        idx = np.flatnonzero(h >= level) if long_ else np.flatnonzero(l <= level)
        target_idx.append(int(idx[0]) if idx.size else None)
    reach_any = target_idx[1] is not None
    reach_before = reach_any and target_idx[1] < stop_i
    best = max((k for k, idx in enumerate(target_idx, start=1)
                if idx is not None and idx < stop_i), default=0)
    label = f"T{best}" if best else ("SL" if stop_idx.size else "EOD")
    if best and stop_idx.size:
        label += "_then_SL"
    return dict(outcome=label, reach_t2_any=reach_any,
                reach_t2_before_stop=reach_before, stop_hit=bool(stop_idx.size))


def finalize(sig, sel, lv, sim, p, lot_size):
    slip = p.slippage
    entry_fill = lv["entry"] * (1 + slip)
    qty = lot_size * p.lots
    legs = [(r, px * (1 - slip), f) for r, px, f in sim["legs"]]
    gross_u = sum(f * (fill - entry_fill) for _, fill, f in legs)
    gross = gross_u * qty
    d = sig["day"]
    orders = [("BUY", entry_fill * qty, d)] + [("SELL", fill * qty * f, d) for _, fill, f in legs]
    cost = C.trade_cost(orders)["total"]
    risk_rs = lv["risk"] * qty
    return dict(symbol=sig["symbol"], day=d, direction=sig["direction"], signal_time=sig["T"],
                contract=sel["contract"], expiry=sel["expiry"], strike=sel["strike"], dte=(sel["expiry"] - d).days,
                delta=round(sel["delta"], 3), iv=round(sel["iv"], 3), spot=round(sel["spot"], 2),
                rvol=round(sig["rvol"], 2), entry=lv["entry"], stop=lv["stop"], t1=lv["tg"][0], t2=lv["tg"][1],
                t3=lv["tg"][2], t4=lv["tg"][3], risk_pct=round(lv["risk"] / lv["entry"], 3),
                legs=";".join(f"{r}@{fill:.2f}x{f:.2f}" for r, fill, f in legs),
                exit_reason=legs[-1][0], n_exits=len(legs), lot_size=lot_size, lots=p.lots,
                gross_rs=round(gross, 2), costs_rs=round(cost, 2), net_rs=round(gross - cost, 2),
                risk_rs=round(risk_rs, 2), r_gross=round(gross / risk_rs, 3), r_net=round((gross - cost) / risk_rs, 3),
                mae_r=round(sim["mae_px"] / lv["risk"], 3), mfe_r=round(sim["mfe_px"] / lv["risk"], 3),
                hit_t1=sim["hit"][0], hit_t2=sim["hit"][1], hit_t3=sim["hit"][2], hit_t4=sim["hit"][3],
                stale=sim["stale"], execution_mode=p.execution_mode, exit_scheme=p.exit_scheme,
                intrabar_mode=p.intrabar_mode, entry_ts=sel.get("entry_ts"))
