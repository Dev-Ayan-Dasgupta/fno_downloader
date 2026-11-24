"""Transaction-cost model for NSE equity options (spec section 12.7). Rates: public sources, Nov 2026; verify vs contract notes."""
from datetime import date

BROKERAGE = 20.0              # flat Rs per executed order, each order separately
GST = 0.18                    # on brokerage + exchange charge + SEBI fee
EXCHANGE = 0.0003553          # NSE option transaction charge + IPFT combined (0.03503% + Rs 50/crore; one rate from 1 Mar 2026), both sides
SEBI = 10 / 1e7               # Rs 10 per crore, both sides
STAMP = 0.00003               # 0.003%, buy side only
SLIPPAGE = 0.02               # base case, per side, fraction of premium


def stt_rate(d: date):
    return 0.0015 if d >= date(2026, 4, 1) else 0.0010     # sell side only


def order_cost(side, value, d):
    """Charges for ONE executed order of premium value `value` (Rs) on date d. side = 'BUY' or 'SELL'."""
    exch, sebi = value * EXCHANGE, value * SEBI
    out = {"brokerage": BROKERAGE, "exchange": exch, "sebi": sebi,
           "gst": GST * (BROKERAGE + exch + sebi),
           "stt": value * stt_rate(d) if side == "SELL" else 0.0,
           "stamp": value * STAMP if side == "BUY" else 0.0}
    out["total"] = sum(out.values())
    return out


def trade_cost(orders):
    """orders: iterable of (side, value, date). Returns summed charges."""
    tot = {}
    for side, value, d in orders:
        for k, v in order_cost(side, value, d).items():
            tot[k] = tot.get(k, 0.0) + v
    return tot


def buy_fill(price, slip=SLIPPAGE):
    return price * (1 + slip)


def sell_fill(price, slip=SLIPPAGE):
    return price * (1 - slip)
