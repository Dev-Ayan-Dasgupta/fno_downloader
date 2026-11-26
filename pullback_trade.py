"""
Underlying-only simulator for Pullback Strategy v1.2.

Three independent exit policies:
  FIXED_1P5R: initial stop or 1.5R target
  FIXED_2R:   initial stop or 2R target
  SWING_TRAIL: initial stop, then a 5-completed-1m-bar swing trail

Signals are filled at the entry 1m candle open recorded by pullback_signals.py.
Intrabar ambiguity is conservative: if stop and target are both touched in one
minute, stop wins. Stop gaps fill at the worse of stop and candle open.
Targets fill at target (no optimistic price improvement).

This is an underlying-stock research simulator, not the options engine.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import time
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SimParams:
    eod: str = "14:59"
    fixed_targets_r: tuple = (1.5, 2.0)
    trail_lookback_bars: int = 5
    trail_buffer_atr: float = 0.05
    slippage_bps_per_side: float = 5.0
    brokerage_per_order_rs: float = 20.0
    other_cost_bps_per_side: float = 0.0
    # Risk-budget sizing is the default. Set to None to use fixed quantity_shares.
    risk_budget_rs: Optional[float] = 1000.0
    quantity_shares: int = 1
    max_shares: Optional[int] = None
    conservative_intrabar: bool = True


def _time(s: str) -> time:
    h, m = map(int, s.split(":"))
    return time(int(h), int(m))


def _valid_ohlc(df: pd.DataFrame) -> pd.Series:
    cols = ["o", "h", "l", "c"]
    x = df[cols].apply(pd.to_numeric, errors="coerce")
    return (
        x.notna().all(axis=1)
        & (x > 0).all(axis=1)
        & (x["h"] >= x[["o", "c", "l"]].max(axis=1))
        & (x["l"] <= x[["o", "c", "h"]].min(axis=1))
    )


def _fill_stop(stop: float, open_: float, direction: str) -> float:
    # Stop-market order: gaps are filled at the worse opening price.
    if direction == "CE":
        return min(stop, open_)
    return max(stop, open_)


def _position_quantity(risk_per_share: float, p: SimParams) -> int:
    if p.risk_budget_rs is None:
        qty = int(p.quantity_shares)
    else:
        if p.risk_budget_rs <= 0:
            raise ValueError("risk_budget_rs must be positive")
        qty = int(np.floor(float(p.risk_budget_rs) / risk_per_share))
    if p.max_shares is not None:
        qty = min(qty, int(p.max_shares))
    return max(0, qty)


def _slipped_fills(entry: float, exit_: float, direction: str, quantity: int, p: SimParams):
    slip = p.slippage_bps_per_side / 10_000.0
    other = p.other_cost_bps_per_side / 10_000.0
    if direction == "CE":
        entry_fill = entry * (1 + slip)
        exit_fill = exit_ * (1 - slip)
    else:
        # Short underlying exposure is represented by selling first, buying back later.
        entry_fill = entry * (1 - slip)
        exit_fill = exit_ * (1 + slip)
    turnover = (entry_fill + exit_fill) * quantity
    brokerage = 2.0 * p.brokerage_per_order_rs
    other_cost = turnover * other
    slippage_cost = (abs(entry_fill - entry) + abs(exit_fill - exit_)) * quantity
    return entry_fill, exit_fill, brokerage, other_cost, slippage_cost


def _one_policy(signal: dict, bars: pd.DataFrame, policy: str, p: SimParams) -> dict:
    direction = str(signal["direction"])
    sign = 1.0 if direction == "CE" else -1.0
    entry_ts = pd.Timestamp(signal["entry_ts"])
    entry = float(signal["entry_price"])
    stop0 = float(signal["initial_stop"])
    risk = float(signal["initial_risk"])
    atr = float(signal["atr5"])

    if direction not in ("CE", "PE"):
        raise ValueError(f"Unknown direction: {direction}")
    if not all(np.isfinite(x) for x in (entry, stop0, risk, atr)) or entry <= 0 or risk <= 0 or atr <= 0:
        raise ValueError("Invalid signal entry/stop/risk/ATR")
    expected_risk = entry - stop0 if direction == "CE" else stop0 - entry
    if expected_risk <= 0 or abs(expected_risk - risk) > max(0.02, risk * 0.02):
        raise ValueError("Signal initial_risk does not reconcile with entry and stop")

    day_bars = bars.copy()
    day_bars["ts"] = pd.to_datetime(day_bars["ts"])
    day_bars = day_bars.sort_values("ts")
    day_bars = day_bars[(day_bars["ts"] >= entry_ts) & (day_bars["ts"].dt.time <= _time(p.eod))]
    day_bars = day_bars[_valid_ohlc(day_bars)].reset_index(drop=True)
    if day_bars.empty or day_bars.iloc[0]["ts"] != entry_ts:
        return {
            **signal, "policy": policy, "exit_reason": "NO_ENTRY_BAR",
            "exit_ts": pd.NaT, "exit_price": np.nan, "gross_r": np.nan,
            "net_r": np.nan, "quantity_shares": np.nan, "gross_pnl_rs": np.nan,
            "net_pnl_rs": np.nan, "risk_rs": np.nan, "brokerage_rs": np.nan,
            "other_costs_rs": np.nan, "slippage_cost_rs": np.nan,
            "total_costs_including_slippage_rs": np.nan, "cost_r": np.nan,
            "net_r": np.nan, "mfe_r": np.nan, "mae_r": np.nan,
            "time_to_mfe_min": np.nan, "time_to_stop_min": np.nan,
            "time_to_target_min": np.nan, "bars_held": 0,
        }

    quantity = _position_quantity(risk, p)
    target_r = 1.5 if policy == "FIXED_1P5R" else 2.0 if policy == "FIXED_2R" else None
    if policy not in ("FIXED_1P5R", "FIXED_2R", "SWING_TRAIL"):
        raise ValueError(f"Unknown policy: {policy}")
    target = entry + sign * target_r * risk if target_r is not None else np.nan

    active_stop = stop0
    exit_reason, exit_ts, exit_price = None, None, None
    mfe_px, mae_px = 0.0, 0.0
    mfe_ts, mae_ts = None, None
    target_ts, stop_ts = None, None
    bars_held = 0
    prior_lows, prior_highs = [], []
    last_close = None

    for i, bar in day_bars.iterrows():
        ts = pd.Timestamp(bar["ts"])
        o, h, l, c = map(float, (bar.o, bar.h, bar.l, bar.c))
        bars_held += 1
        last_close = c

        fav = (h - entry) if direction == "CE" else (entry - l)
        adv = (entry - l) if direction == "CE" else (h - entry)
        if fav > mfe_px:
            mfe_px, mfe_ts = fav, ts
        if adv > mae_px:
            mae_px, mae_ts = adv, ts

        stop_touched = l <= active_stop if direction == "CE" else h >= active_stop
        target_touched = (
            (h >= target) if direction == "CE" else (l <= target)
        ) if target_r is not None else False

        if stop_touched and (target_touched or policy == "SWING_TRAIL" or not target_touched):
            # Conservative ordering: stop wins any same-minute ambiguity.
            exit_reason = "SL" if abs(active_stop - stop0) < 1e-9 else "TRAIL"
            exit_price = _fill_stop(active_stop, o, direction)
            exit_ts = ts
            stop_ts = ts
            break

        if target_r is not None and target_touched:
            exit_reason = f"TARGET_{target_r:g}R"
            exit_price = target
            exit_ts = ts
            target_ts = ts
            break

        # Update trailing stop only after this 1m bar closes; it becomes active
        # on the next 1m bar, so no look-ahead within the current candle.
        if policy == "SWING_TRAIL":
            if len(prior_lows) >= p.trail_lookback_bars:
                if direction == "CE":
                    candidate = min(prior_lows[-p.trail_lookback_bars:]) - p.trail_buffer_atr * atr
                    active_stop = max(active_stop, candidate)
                else:
                    candidate = max(prior_highs[-p.trail_lookback_bars:]) + p.trail_buffer_atr * atr
                    active_stop = min(active_stop, candidate)
        prior_lows.append(l)
        prior_highs.append(h)

    if exit_reason is None:
        last = day_bars.iloc[-1]
        exit_reason = "EOD"
        exit_ts = pd.Timestamp(last.ts)
        exit_price = float(last.c)

    entry_fill, exit_fill, brokerage, other_cost, slippage_cost = _slipped_fills(
        entry, float(exit_price), direction, quantity, p
    )
    gross_pnl_per_share = sign * (float(exit_price) - entry)
    gross_pnl_rs = gross_pnl_per_share * quantity
    estimated_costs = brokerage + other_cost
    net_pnl = sign * (exit_fill - entry_fill) * quantity - brokerage - other_cost
    risk_rs = risk * quantity
    gross_r = gross_pnl_per_share / risk
    net_r = net_pnl / risk_rs if risk_rs > 0 else np.nan

    def elapsed(ts):
        return (ts - entry_ts).total_seconds() / 60 if ts is not None else np.nan

    return {
        **signal,
        "policy": policy,
        "exit_reason": exit_reason,
        "exit_ts": exit_ts,
        "exit_price": float(exit_price),
        "entry_fill": entry_fill,
        "exit_fill": exit_fill,
        "quantity_shares": quantity,
        "gross_pnl_rs": gross_pnl_rs,
        "slippage_cost_rs": slippage_cost,
        "brokerage_rs": brokerage,
        "other_costs_rs": other_cost,
        "estimated_costs_rs": estimated_costs,
        "total_costs_including_slippage_rs": slippage_cost + estimated_costs,
        "cost_r": (slippage_cost + estimated_costs) / risk_rs if risk_rs > 0 else np.nan,
        "net_pnl_rs": net_pnl,
        "risk_rs": risk_rs,
        "gross_r": gross_r,
        "net_r": net_r,
        "mfe_r": mfe_px / risk,
        "mae_r": mae_px / risk,
        "mfe_ts": mfe_ts,
        "mae_ts": mae_ts,
        "time_to_mfe_min": elapsed(mfe_ts),
        "time_to_stop_min": elapsed(stop_ts),
        "time_to_target_min": elapsed(target_ts),
        "bars_held": bars_held,
        "target_r": target_r,
        "initial_stop": stop0,
        "final_stop": active_stop,
    }


def simulate_signal(signal: dict, m1_day: pd.DataFrame, params: Optional[SimParams] = None) -> pd.DataFrame:
    p = params or SimParams()
    rows = [
        _one_policy(signal, m1_day, "FIXED_1P5R", p),
        _one_policy(signal, m1_day, "FIXED_2R", p),
        _one_policy(signal, m1_day, "SWING_TRAIL", p),
    ]
    return pd.DataFrame(rows)


def run_simulation(signals: pd.DataFrame, contexts: Dict[str, object],
                   params: Optional[SimParams] = None) -> pd.DataFrame:
    p = params or SimParams()
    out = []
    for sig in signals.to_dict("records"):
        symbol = str(sig["symbol"])
        ctx = contexts.get(symbol)
        if ctx is None:
            continue
        day = pd.Timestamp(sig["day"]).date()
        m1_day = ctx.m1_by_day.get(day)
        if m1_day is None or m1_day.empty:
            continue
        # All policies are simulated independently from the same signal and bars.
        out.append(simulate_signal(sig, m1_day, p))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def summarize(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    rows = []
    keys = ["setup", "policy"]
    for (setup, policy), g in trades.groupby(keys, dropna=False):
        x = pd.to_numeric(g["net_r"], errors="coerce").dropna()
        wins = x[x > 0].sum()
        losses = -x[x < 0].sum()
        # Day-cluster bootstrap CI for mean net R. Reproducible seed.
        clustered = g.assign(_net=pd.to_numeric(g["net_r"], errors="coerce")).dropna(subset=["_net"])
        day_values = [
            grp["_net"].to_numpy()
            for _, grp in clustered.groupby("day", sort=False)
            if len(grp)
        ]
        ci_lo = ci_hi = np.nan
        if len(day_values) >= 2:
            rng = np.random.default_rng(20261125)
            boot_means = np.empty(5000, dtype=float)
            for b in range(len(boot_means)):
                chosen = rng.integers(0, len(day_values), size=len(day_values))
                sample = np.concatenate([day_values[j] for j in chosen])
                boot_means[b] = sample.mean()
            ci_lo, ci_hi = np.quantile(boot_means, [0.025, 0.975])
        gross = pd.to_numeric(g["gross_r"], errors="coerce").dropna()
        gross_wins = gross[gross > 0].sum()
        gross_losses = -gross[gross < 0].sum()
        rows.append({
            "setup": setup, "policy": policy, "trades": len(x),
            "trading_days": g["day"].nunique(),
            "gross_win_rate_pct": (gross > 0).mean() * 100,
            "net_win_rate_pct": (x > 0).mean() * 100,
            "avg_gross_r": gross.mean(),
            "avg_net_r": x.mean(), "total_net_r": x.sum(),
            "gross_profit_factor": gross_wins / gross_losses if gross_losses > 0 else np.inf,
            "net_profit_factor": wins / losses if losses > 0 else np.inf,
            "median_net_r": x.median(),
            "avg_quantity_shares": g["quantity_shares"].mean(),
            "avg_brokerage_rs": g["brokerage_rs"].mean(),
            "avg_slippage_cost_rs": g["slippage_cost_rs"].mean(),
            "avg_other_costs_rs": g["other_costs_rs"].mean(),
            "avg_total_cost_r": g["cost_r"].mean(),
            "mean_net_r_day_boot_ci_low": ci_lo,
            "mean_net_r_day_boot_ci_high": ci_hi,
            "avg_mfe_r": g["mfe_r"].mean(),
            "avg_mae_r": g["mae_r"].mean(),
            "avg_bars_held": g["bars_held"].mean(),
            "stop_exit_pct": g["exit_reason"].isin(["SL", "TRAIL"]).mean() * 100,
            "target_exit_pct": g["exit_reason"].astype(str).str.startswith("TARGET").mean() * 100,
            "eod_exit_pct": (g["exit_reason"] == "EOD").mean() * 100,
        })
    return pd.DataFrame(rows).sort_values(["setup", "policy"]).reset_index(drop=True)


def params_dict(params: Optional[SimParams] = None):
    return asdict(params or SimParams())
