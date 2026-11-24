"""Black-Scholes helpers (European, no dividends). Approximate for Indian stock options: documented in the spec."""
import math
from datetime import datetime, time

import bt_config as B


def _ncdf(x):
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def _intrinsic(S, K, T, r, cp):
    disc = K * math.exp(-r * max(T, 0.0))
    return max(S - disc, 0.0) if cp == "CE" else max(disc - S, 0.0)


def bs_price(S, K, T, sigma, cp, r=B.RISK_FREE):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0) if cp == "CE" else max(K - S, 0.0)
    v = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / v
    d2 = d1 - v
    if cp == "CE":
        return S * _ncdf(d1) - K * math.exp(-r * T) * _ncdf(d2)
    return K * math.exp(-r * T) * _ncdf(-d2) - S * _ncdf(-d1)


def implied_vol(price, S, K, T, cp, r=B.RISK_FREE, lo=0.01, hi=5.0):
    """Bisection. NaN when the price is at/below intrinsic (no time value to invert) or T <= 0."""
    if T <= 0 or price <= _intrinsic(S, K, T, r, cp) + 1e-9:
        return float("nan")
    if bs_price(S, K, T, hi, cp, r) < price:
        return float("nan")
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if bs_price(S, K, T, mid, cp, r) < price:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def delta(S, K, T, sigma, cp, r=B.RISK_FREE):
    if T <= 0 or sigma <= 0:
        return (1.0 if S > K else 0.0) if cp == "CE" else (-1.0 if S < K else 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    return _ncdf(d1) if cp == "CE" else _ncdf(d1) - 1.0


def time_to_expiry(ts, expiry_date):
    """Years from ts to 15:30 on the expiry date (calendar-time convention)."""
    end = datetime.combine(expiry_date, time(15, 30))
    return max((end - ts).total_seconds(), 0.0) / (365.0 * 24 * 3600)


def reprice_at_spot(S_new, price_now, S_now, K, T, cp, r=B.RISK_FREE):
    """Option price if the underlying moves to S_new, holding implied vol and time constant (spec section 5)."""
    iv = implied_vol(price_now, S_now, K, T, cp, r)
    if iv != iv:                                   # NaN: fall back to intrinsic-based first-order estimate
        return None
    return bs_price(S_new, K, T, iv, cp, r)
