"""
Pullback strategy signal engine, version 1.

Uses the existing ORB project's data layer:
    import bt_signals as ORB
    ctx = ORB.build_ctx(symbol, data_dir)
    nifty5, nifty_ref_type = ORB.index_context("NIFTY", data_dir)

Signal definitions:
- VWAP retest: established trend, pullback into VWAP tolerance, close back on trend side
  with a directional candle.
- Breakout retest: close beyond the prior N-bar high/low, then a later retest of that
  frozen breakout level, followed by a directional close.

Signals are evaluated only on completed 5-minute bars. Entry is the open of the
1-minute candle whose timestamp equals the 5-minute bar end (next minute open).
The engine emits underlying signals only; options selection/execution is a later stage.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import time
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PullbackParams:
    # Trend definition (5-minute bars)
    ema_fast: int = 20
    ema_slow: int = 50
    trend_slope_bars: int = 3
    require_vwap_slope: bool = True

    # Setup window
    scan_from: str = "09:35"
    scan_to: str = "14:00"
    eod: str = "14:59"

    # VWAP retest
    vwap_tolerance_atr: float = 0.15
    vwap_reclaim_buffer_atr: float = 0.02

    # Breakout-level retest
    breakout_lookback: int = 12
    breakout_buffer_atr: float = 0.05
    retest_tolerance_atr: float = 0.15
    retest_max_wait_bars: int = 12

    # Confirmation and risk
    min_body_atr: float = 0.05
    stop_buffer_atr: float = 0.10
    min_atr5_pct: float = 0.0
    # Defensive data-integrity limits. These are guards, not entry optimisations.
    max_bar_range_pct: float = 0.15
    max_entry_gap_pct: float = 0.05
    max_feature_deviation_pct: float = 0.20
    max_atr_pct: float = 0.10
    max_initial_risk_pct: float = 0.10
    one_signal_per_day_direction: bool = True

    # Optional broad-market confirmation. None disables it.
    nifty_confirmation: str = "none"  # none | same_direction

    # Output execution assumption
    execution_mode: str = "next_open"


def _clock(value: str) -> time:
    h, m = map(int, value.split(":"))
    return time(h, m)


def _is_finite(x) -> bool:
    try:
        return bool(np.isfinite(float(x)))
    except (TypeError, ValueError):
        return False


def _prepare_frame(ctx, p: PullbackParams) -> pd.DataFrame:
    f = ctx.f5.copy().sort_values(["day", "ts"]).reset_index(drop=True)
    required = {"ts", "end", "day", "o", "h", "l", "c", "vwap", "atr5", "n"}
    missing = required - set(f.columns)
    if missing:
        raise ValueError(f"ctx.f5 missing required columns: {sorted(missing)}")

    # Preserve the original context ATR for audit. The ORB context resets ATR
    # every session, which would make a 09:35 pullback scan blind for 14 bars.
    f["atr5_context"] = f["atr5"]

    # Flag impossible/implausible raw 5m bars before calculating indicators.
    # Bad bars are excluded from EMA/ATR calculations and the whole session is
    # skipped, preventing one corrupt candle from contaminating later signals.
    o, h, l, c = (pd.to_numeric(f[k], errors="coerce") for k in ("o", "h", "l", "c"))
    f["bar_range_pct"] = (h - l) / o.replace(0, np.nan)
    f["bar_data_valid"] = (
        o.gt(0) & h.ge(pd.concat([o, c, l], axis=1).max(axis=1)) &
        l.le(pd.concat([o, c, h], axis=1).min(axis=1)) &
        f["bar_range_pct"].le(p.max_bar_range_pct)
    )
    f["bar_data_valid"] = f["bar_data_valid"].fillna(False)

    # EMAs run continuously across sessions, ignoring invalid/incomplete candles.
    indicator_valid = f["bar_data_valid"] & pd.to_numeric(f["n"], errors="coerce").eq(5)
    clean_close = c.where(indicator_valid)
    f["ema_fast"] = clean_close.ewm(
        span=p.ema_fast, adjust=False, min_periods=p.ema_fast, ignore_na=True
    ).mean()
    f["ema_slow"] = clean_close.ewm(
        span=p.ema_slow, adjust=False, min_periods=p.ema_slow, ignore_na=True
    ).mean()

    # Build a pullback-specific ATR from 5m OHLC. True range uses the previous
    # close within the same session, so overnight gaps are deliberately omitted.
    # The 14-bar rolling window carries across sessions, avoiding daily warm-up.
    prev_close = c.groupby(f["day"], sort=False).shift(1)
    tr = pd.concat([
        h - l,
        (h - prev_close).abs(),
        (l - prev_close).abs()
    ], axis=1).max(axis=1)
    tr = tr.where(indicator_valid)
    # IMPORTANT: roll over the last 14 VALID observations, not the last 14
    # dataframe rows. A partial/missing final candle must not restart the ATR
    # warm-up at the beginning of every following session.
    valid_tr = tr.dropna()
    atr_valid = valid_tr.rolling(14, min_periods=14).mean()
    f["atr5"] = atr_valid.reindex(f.index)

    # VWAP slope resets each session; EMA slope uses the continuous EMA series.
    f["vwap_slope"] = f["vwap"] - f.groupby("day", sort=False)["vwap"].shift(p.trend_slope_bars)
    f["ema_fast_slope"] = f["ema_fast"] - f["ema_fast"].shift(p.trend_slope_bars)
    f["prior_high"] = f.groupby("day", sort=False)["h"].transform(
        lambda s: s.rolling(p.breakout_lookback, min_periods=p.breakout_lookback).max().shift(1)
    )
    f["prior_low"] = f.groupby("day", sort=False)["l"].transform(
        lambda s: s.rolling(p.breakout_lookback, min_periods=p.breakout_lookback).min().shift(1)
    )
    f["body"] = (f["c"] - f["o"]).abs()
    f["body_atr"] = f["body"] / f["atr5"].replace(0, np.nan)
    f["atr_pct"] = f["atr5"] / f["c"].replace(0, np.nan)
    return f


def _trend(row, direction: str, p: PullbackParams) -> bool:
    vals = [row.ema_fast, row.ema_slow, row.vwap, row.ema_fast_slope]
    if not all(_is_finite(v) for v in vals):
        return False
    if direction == "CE":
        ok = row.c > row.ema_fast > row.ema_slow and row.ema_fast_slope > 0 and row.c > row.vwap
        if p.require_vwap_slope:
            ok = ok and _is_finite(row.vwap_slope) and row.vwap_slope > 0
        return bool(ok)
    ok = row.c < row.ema_fast < row.ema_slow and row.ema_fast_slope < 0 and row.c < row.vwap
    if p.require_vwap_slope:
        ok = ok and _is_finite(row.vwap_slope) and row.vwap_slope < 0
    return bool(ok)


def _nifty_ok(nifty5, end, direction, mode):
    if mode == "none":
        return True
    if mode != "same_direction":
        raise ValueError("nifty_confirmation must be 'none' or 'same_direction'")
    if nifty5 is None or end not in nifty5.index:
        return False
    row = nifty5.loc[end]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[-1]
    if not (_is_finite(row.get("c")) and _is_finite(row.get("ref"))):
        return False
    return bool(row["c"] > row["ref"] if direction == "CE" else row["c"] < row["ref"])


def _entry_bar(ctx, end):
    """Return the exact 1m bar starting at the completed 5m bar's end."""
    m1 = ctx.m1_by_day.get(end.date(), pd.DataFrame())
    if m1.empty:
        return None
    hit = m1[m1["ts"] == end]
    if hit.empty:
        return None
    return hit.iloc[0]


def _price_integrity_reason(row, p: PullbackParams):
    """Reject internally inconsistent or implausibly scaled 5m bars/features."""
    vals = [row.o, row.h, row.l, row.c, row.atr5]
    if not all(_is_finite(v) for v in vals):
        return "missing_or_nonfinite_ohlc_or_atr"
    if not _is_finite(row.vwap):
        return "vwap_unavailable"
    o, h, l, c = map(float, (row.o, row.h, row.l, row.c))
    if min(o, h, l, c) <= 0 or h < max(o, c, l) or l > min(o, c, h):
        return "invalid_ohlc_relationship"
    if (h - l) / o > p.max_bar_range_pct:
        return "implausible_5m_range_pct"
    if float(row.atr5) / c > p.max_atr_pct:
        return "implausible_atr_pct"
    for name in ("vwap", "ema_fast", "ema_slow"):
        value = getattr(row, name, np.nan)
        if _is_finite(value) and abs(float(value) / c - 1.0) > p.max_feature_deviation_pct:
            return f"{name}_price_scale_mismatch"
    return None


def _record(ctx, row, direction, setup, level, p, nifty5, extra=None):
    end = pd.Timestamp(row.end)
    entry = _entry_bar(ctx, end)
    if entry is None:
        return None, "missing_next_1m_open"
    # The entry candle must be on the same price scale as the completed signal bar.
    entry_ohlc = [entry.get(k, np.nan) for k in ("o", "h", "l", "c")]
    if not all(_is_finite(v) and float(v) > 0 for v in entry_ohlc):
        return None, "invalid_next_1m_ohlc"
    eo, eh, el, ec = map(float, entry_ohlc)
    if eh < max(eo, ec, el) or el > min(eo, ec, eh):
        return None, "invalid_next_1m_ohlc"
    if abs(eo / float(row.c) - 1.0) > p.max_entry_gap_pct:
        return None, "entry_price_scale_or_gap_mismatch"
    if abs(float(level) / float(row.c) - 1.0) > p.max_feature_deviation_pct:
        return None, "reference_level_price_scale_mismatch"
    if not _nifty_ok(nifty5, end, direction, p.nifty_confirmation):
        return None, "nifty_confirmation_failed"

    atr = float(row.atr5)
    if not _is_finite(atr) or atr <= 0:
        return None, "invalid_atr5"

    if direction == "CE":
        stop = float(row.l) - p.stop_buffer_atr * atr
        risk = float(entry.o) - stop
    else:
        stop = float(row.h) + p.stop_buffer_atr * atr
        risk = stop - float(entry.o)
    if risk <= 0:
        return None, "non_positive_initial_risk"
    if risk / float(entry.o) > p.max_initial_risk_pct:
        return None, "initial_risk_pct_too_large"

    out = {
        "symbol": ctx.sym,
        "day": row.day,
        "direction": direction,
        "setup": setup,
        "signal_bar_start": row.ts,
        "signal_bar_end": end,
        "entry_ts": entry.ts,
        "entry_price": float(entry.o),
        "signal_open": float(row.o),
        "signal_high": float(row.h),
        "signal_low": float(row.l),
        "signal_close": float(row.c),
        "vwap": float(row.vwap),
        "ema_fast": float(row.ema_fast),
        "ema_slow": float(row.ema_slow),
        "atr5": atr,
        "atr5_context": float(row.atr5_context) if _is_finite(row.atr5_context) else np.nan,
        "atr5_method": "rolling_14_valid_5m_TR_no_overnight_gap",
        "reference_level": float(level),
        "initial_stop": stop,
        "initial_risk": risk,
        "initial_risk_pct": risk / float(entry.o),
        "entry_gap_pct": abs(float(entry.o) / float(row.c) - 1.0),
        "body_atr": float(row.body_atr) if _is_finite(row.body_atr) else np.nan,
        "vwap_slope": float(row.vwap_slope) if _is_finite(row.vwap_slope) else np.nan,
        "ema_fast_slope": float(row.ema_fast_slope) if _is_finite(row.ema_fast_slope) else np.nan,
        "execution_mode": p.execution_mode,
        "nifty_confirmation": p.nifty_confirmation,
    }
    if extra:
        out.update(extra)
    return out, None


def find_pullback_signals(ctx, nifty5=None, params: Optional[PullbackParams] = None):
    """
    Returns (signals_df, rejects_df).

    A maximum of one accepted signal per symbol/day/direction/setup is emitted
    when one_signal_per_day_direction=True. VWAP and breakout setups are kept
    separate and can both emit a signal on the same day.
    """
    p = params or PullbackParams()
    if p.execution_mode != "next_open":
        raise ValueError("Version 1 supports next_open execution only.")
    f = _prepare_frame(ctx, p)
    lo, hi = _clock(p.scan_from), _clock(p.scan_to)
    signals, rejects = [], []
    bad_days = set(f.loc[~f["bar_data_valid"], "day"].unique())

    for day, day_df in f.groupby("day", sort=True):
        if hasattr(ctx, "excl") and day in ctx.excl:
            rejects.append({"symbol": ctx.sym, "day": day, "setup": "BOTH", "reason": "excluded_session"})
            continue
        if day in bad_days:
            rejects.append({"symbol": ctx.sym, "day": day, "setup": "BOTH",
                            "reason": "session_contains_implausible_5m_bar"})
            continue

        g = day_df.reset_index(drop=True)
        accepted = set()
        # State for breakout-level retests, separately for CE and PE.
        active = {"CE": None, "PE": None}

        for i, row in g.iterrows():
            end = pd.Timestamp(row.end)
            if end.time() < lo or end.time() > hi:
                continue
            if int(row.n) != 5:
                rejects.append({"symbol": ctx.sym, "day": day, "signal_bar_end": end,
                                "setup": "BOTH", "reason": "incomplete_5m_bar"})
                continue
            if not _is_finite(row.atr5) or row.atr5 <= 0 or not _is_finite(row.atr_pct) or row.atr_pct < p.min_atr5_pct:
                rejects.append({"symbol": ctx.sym, "day": day, "signal_bar_end": end,
                                "setup": "BOTH", "reason": "atr_unavailable_or_below_floor"})
                continue
            integrity_error = _price_integrity_reason(row, p)
            if integrity_error:
                rejects.append({"symbol": ctx.sym, "day": day, "signal_bar_end": end,
                                "setup": "BOTH", "reason": integrity_error})
                continue

            for direction in ("CE", "PE"):
                if not _trend(row, direction, p):
                    continue

                # Optional same-direction NIFTY confirmation applies to both setups.
                if not _nifty_ok(nifty5, end, direction, p.nifty_confirmation):
                    rejects.append({"symbol": ctx.sym, "day": day, "signal_bar_end": end,
                                    "direction": direction, "setup": "BOTH",
                                    "reason": "nifty_confirmation_failed"})
                    continue

                # A) VWAP retest and reclaim.
                key = (day, direction, "VWAP_RETEST")
                if key not in accepted:
                    tol = p.vwap_tolerance_atr * float(row.atr5)
                    reclaim = p.vwap_reclaim_buffer_atr * float(row.atr5)
                    if direction == "CE":
                        touched = row.l <= row.vwap + tol
                        reclaimed = row.c >= row.vwap + reclaim
                        candle_ok = row.c > row.o and row.body_atr >= p.min_body_atr
                    else:
                        touched = row.h >= row.vwap - tol
                        reclaimed = row.c <= row.vwap - reclaim
                        candle_ok = row.c < row.o and row.body_atr >= p.min_body_atr

                    if touched and reclaimed and candle_ok:
                        rec, error = _record(
                            ctx, row, direction, "VWAP_RETEST", row.vwap, p, nifty5,
                            {"pullback_depth_atr": (
                                (row.vwap - row.l) / row.atr5 if direction == "CE"
                                else (row.h - row.vwap) / row.atr5
                            )}
                        )
                        if rec:
                            signals.append(rec)
                            accepted.add(key)
                        elif error:
                            rejects.append({"symbol": ctx.sym, "day": day, "signal_bar_end": end,
                                            "direction": direction, "setup": "VWAP_RETEST", "reason": error})

                # B) Breakout of prior rolling range, then a later retest.
                state = active[direction]
                if state is not None and i > state["breakout_i"]:
                    expired = (i - state["breakout_i"]) > p.retest_max_wait_bars
                    if expired:
                        active[direction] = None
                        rejects.append({"symbol": ctx.sym, "day": day, "signal_bar_end": end,
                                        "direction": direction, "setup": "BREAKOUT_RETEST",
                                        "reason": "retest_window_expired"})
                    else:
                        level = state["level"]
                        tol = p.retest_tolerance_atr * float(row.atr5)
                        if direction == "CE":
                            touched = row.l <= level + tol
                            held = row.c >= level
                            candle_ok = row.c > row.o and row.body_atr >= p.min_body_atr
                            depth = (level - row.l) / row.atr5
                        else:
                            touched = row.h >= level - tol
                            held = row.c <= level
                            candle_ok = row.c < row.o and row.body_atr >= p.min_body_atr
                            depth = (row.h - level) / row.atr5
                        if touched and held and candle_ok:
                            key = (day, direction, "BREAKOUT_RETEST")
                            if key not in accepted:
                                rec, error = _record(
                                    ctx, row, direction, "BREAKOUT_RETEST", level, p, nifty5,
                                    {"pullback_depth_atr": float(depth),
                                     "breakout_bar_end": state["breakout_end"],
                                     "breakout_close": state["breakout_close"],
                                     "bars_from_breakout": i - state["breakout_i"]}
                                )
                                if rec:
                                    signals.append(rec)
                                    accepted.add(key)
                                    active[direction] = None
                                elif error:
                                    rejects.append({"symbol": ctx.sym, "day": day, "signal_bar_end": end,
                                                    "direction": direction, "setup": "BREAKOUT_RETEST",
                                                    "reason": error})

                # A breakout is only armed after a completed close beyond the prior range.
                # It cannot be retested on the same candle.
                if active[direction] is None and _is_finite(row.prior_high) and _is_finite(row.prior_low):
                    if direction == "CE":
                        broke = row.c > row.prior_high + p.breakout_buffer_atr * row.atr5
                        level = float(row.prior_high)
                    else:
                        broke = row.c < row.prior_low - p.breakout_buffer_atr * row.atr5
                        level = float(row.prior_low)
                    if broke:
                        active[direction] = {
                            "level": level, "breakout_i": i, "breakout_end": end,
                            "breakout_close": float(row.c),
                        }

        # Record no-signal diagnostics at day/setup level only when no signal was found.
        for direction in ("CE", "PE"):
            for setup in ("VWAP_RETEST", "BREAKOUT_RETEST"):
                if (day, direction, setup) not in accepted:
                    rejects.append({"symbol": ctx.sym, "day": day, "direction": direction,
                                    "setup": setup, "reason": "no_qualifying_setup"})

    sig = pd.DataFrame(signals)
    rej = pd.DataFrame(rejects)
    if not sig.empty:
        sig = sig.sort_values(["entry_ts", "symbol", "setup", "direction"]).reset_index(drop=True)
        # Preserve both independent variants, but identify same-entry overlaps so a
        # combined portfolio can count the underlying event only once.
        overlap_cols = ["symbol", "day", "direction", "entry_ts"]
        sig["overlap_group_id"] = (
            sig[overlap_cols].astype(str).agg("|".join, axis=1)
        )
        sig["variant_overlap_count"] = sig.groupby("overlap_group_id")["setup"].transform("count")
        sig["is_cross_variant_overlap"] = sig["variant_overlap_count"] > 1
    return sig, rej


def params_dict(params: Optional[PullbackParams] = None):
    return asdict(params or PullbackParams())
