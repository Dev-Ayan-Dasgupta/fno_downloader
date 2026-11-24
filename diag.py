import config  # loads .env

from datetime import date, timedelta
from client import Client

g = Client(rps=2)._get()

def t(label, sym, seg, d0, d1, interval):
    try:
        r = g.get_historical_candles(exchange=g.EXCHANGE_NSE, segment=seg, groww_symbol=sym,
                                     start_time=f"{d0} 09:15:00", end_time=f"{d1} 15:30:00",
                                     candle_interval=interval)
        cs = r.get("candles") or []
        print(f"{label}: n={len(cs)} first={cs[0][0] if cs else None} last={cs[-1][0] if cs else None}")
    except Exception as e:
        print(f"{label}: ERROR {type(e).__name__} code={getattr(e, 'code', None)} {str(e)[:160]}")

end = date(2026, 11, 20)
R, N = "NSE-RELIANCE", g.SEGMENT_CASH
M1, M5 = g.CANDLE_INTERVAL_MIN_1, getattr(g, "CANDLE_INTERVAL_MIN_5", None)
for d in (1, 5, 14, 28):
    t(f"1min last {d}d", R, N, end - timedelta(days=d), end, M1)
for a, b in [("2026-10-05", "2026-10-30"), ("2025-06-02", "2025-06-27"), ("2024-12-02", "2024-12-27")]:
    d0, d1 = date.fromisoformat(a), date.fromisoformat(b)
    t(f"1min {a}..{b}", R, N, d0, d1, M1)
    if M5:
        t(f"5min {a}..{b}", R, N, d0, d1, M5)
t("1min single day 2025-06-02", R, N, date(2025, 6, 2), date(2025, 6, 2), M1)
t("daily 2026-05-25..2026-11-20", R, N, date(2026, 5, 25), end, g.CANDLE_INTERVAL_DAY)