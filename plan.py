"""Pure planning helpers (no network): windows, active expiry, strike band."""
import bisect
from datetime import timedelta


def parse_contract(gs):
    """'NSE-BAJAJ-AUTO-30Dec25-1500-CE' -> ('30Dec25', 1500.0, 'CE')  (rsplit: names can contain '-')."""
    _, expd, strike, typ = gs.rsplit("-", 3)
    return expd, float(strike), typ


def grid_windows(start, end, epoch, size):
    w = epoch + timedelta(days=((start - epoch).days // size) * size)
    out = []
    while w <= end:
        out.append(w)
        w += timedelta(days=size)
    return out


def active_expiry(day, expiries, roll_days):
    """Nearest expiry with at least roll_days calendar days left (default ~3 trading days)."""
    for e in expiries:
        if (e - day).days >= roll_days:
            return e
    return None


def strike_band(strikes, low, high, k):
    """All listed strikes within k strikes of ATM at the day's low, and of ATM at the day's high."""
    def nearest(x):
        i = bisect.bisect_left(strikes, x)
        if i == 0:
            return 0
        if i == len(strikes):
            return len(strikes) - 1
        return i if strikes[i] - x < x - strikes[i - 1] else i - 1
    lo, hi = nearest(low), nearest(high)
    return strikes[max(0, lo - k): min(len(strikes) - 1, hi + k) + 1]


def merge_windows(days, max_span, max_gap):
    out = []
    for d in sorted(days):
        if out and (d - out[-1][1]).days <= max_gap and (d - out[-1][0]).days < max_span:
            out[-1][1] = d
        else:
            out.append([d, d])
    return [(a, b) for a, b in out]
