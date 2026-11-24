import config
from datetime import date
from client import Client
g = Client(rps=2)._get()
today = date.today()
for sym, k in [("RELIANCE", 1360), ("FEDERALBNK", 365)]:   # adjust strikes if the spot has moved
    q = g.get_quote(exchange=g.EXCHANGE_NSE, segment=g.SEGMENT_FNO, trading_symbol=f"{sym}26NOV{k}CE")
    print(f"\n{sym} {k}CE: bid={q.get('bid_price')} ask={q.get('offer_price')} volume={q.get('volume')} OI={q.get('open_interest')}")
    print("depth:", str(q.get("depth"))[:500])
    c = g.get_historical_candles(exchange=g.EXCHANGE_NSE, segment=g.SEGMENT_FNO,
        groww_symbol=f"NSE-{sym}-23Nov26-{k}-CE", start_time=f"{today} 09:15:00",
        end_time=f"{today} 15:30:00", candle_interval=g.CANDLE_INTERVAL_MIN_1)["candles"]
    v = sum(x[5] or 0 for x in c); oi = [x[6] for x in c if len(x) > 6 and x[6] is not None]
    print(f"candles: n={len(c)} sum_volume={v} last_OI={oi[-1] if oi else None} | candle volume / quote volume = {v / max(q.get('volume') or 1, 1):.0f}")