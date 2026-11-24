import config
from datetime import date, timedelta
from client import Client
g = Client(rps=2)._get()
for sym, k in [("RELIANCE", 1360), ("FEDERALBNK", 365)]:
    q = g.get_quote(exchange=g.EXCHANGE_NSE, segment=g.SEGMENT_FNO, trading_symbol=f"{sym}26NOV{k}CE")
    prev_oi = q.get("previous_open_interest")
    d, oi = date.today(), []
    for _ in range(5):
        d -= timedelta(days=1)
        if d.weekday() >= 5:
            continue
        c = g.get_historical_candles(exchange=g.EXCHANGE_NSE, segment=g.SEGMENT_FNO,
            groww_symbol=f"NSE-{sym}-23Nov26-{k}-CE", start_time=f"{d} 09:15:00",
            end_time=f"{d} 15:30:00", candle_interval=g.CANDLE_INTERVAL_MIN_1)["candles"]
        oi = [x[6] for x in c if len(x) > 6 and x[6] is not None]
        if oi:
            break
    ratio = oi[-1] / prev_oi if oi and prev_oi else None
    print(f"{sym} {k}CE: quote previous_open_interest={prev_oi} | last candle OI on {d}={oi[-1] if oi else None} | ratio={ratio}")