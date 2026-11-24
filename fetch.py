"""Download phases. Every task is idempotent and recorded in the manifest, so a stopped run resumes where it left off."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta

import pandas as pd

import config as C
import plan

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

COLS = ["ts", "o", "h", "l", "c", "v", "oi"]


def safe(sym):
    return sym.replace("&", "_and_").replace("/", "_")


def parse_candles(resp):
    rows = (resp or {}).get("candles") or []
    if not rows:
        return pd.DataFrame(columns=COLS)
    df = pd.DataFrame([list(r) + [None] * (7 - len(r)) for r in rows], columns=COLS)
    df["ts"] = pd.to_datetime(df["ts"])
    return df


def fetch_range(client, segment, gsym, d0, d1, interval, depth=0):
    """One API call; if the answer stops suspiciously early (possible silent truncation) split the window and retry."""
    try:
        resp = client.call("get_historical_candles", exchange=client.const("EXCHANGE_NSE"),
                           segment=client.const(segment), groww_symbol=gsym,
                           start_time=f"{d0} 09:15:00", end_time=f"{d1} 15:30:00",
                           candle_interval=client.const(interval))
    except Exception as e:  # noqa
        if "notfound" in type(e).__name__.lower() or str(getattr(e, "code", "")) == "404":
            print(f"[warn] {gsym} {d0}..{d1}: treated as empty ({type(e).__name__} {getattr(e, 'code', '')}: {str(e)[:100]})")
            return pd.DataFrame(columns=COLS)
        raise
    df = parse_candles(resp)
    span = (d1 - d0).days
    if depth < 2 and span > 3 and len(df) and (d1 - df["ts"].max().date()).days > 6 \
            and d1 < date.today() - timedelta(days=2):
        mid = d0 + timedelta(days=span // 2)
        a = fetch_range(client, segment, gsym, d0, mid, interval, depth + 1)
        b = fetch_range(client, segment, gsym, mid + timedelta(days=1), d1, interval, depth + 1)
        df = pd.concat([df, a, b]).drop_duplicates("ts").sort_values("ts")
    return df


def load_universe(client, store):
    und = store.json_get("meta", "universe.json", max_age=86400)
    if not und:
        df = client.call("get_all_instruments")
        fno = df[df["segment"] == "FNO"]
        und = sorted(set(fno["underlying_symbol"].dropna().astype(str)) - C.INDEX_UNDERLYINGS)
        store.json_put(und, "meta", "universe.json")
    return [s for s in und if "NSETEST" not in s and s != "NIFTYFPI"]


# ------------------------------------------------------------------ phase 1: underlying candles
def run_stock_phase(client, store, symbols, start, end, workers=4, log=print):
    today = date.today()
    syms = list(symbols) + [s for s in C.EXTRA_INDICES if s not in symbols]
    tasks = []
    for sym in syms:
        for ws in plan.grid_windows(start, end, C.GRID_EPOCH, C.GRID_DAYS):
            we = ws + timedelta(days=C.GRID_DAYS - 1)
            complete = we < today
            if complete and store.done(f"s1m:{sym}:{ws}"):
                continue
            tasks.append((sym, ws, min(we, end), complete))
    log(f"[stock 1m] {len(tasks)} request windows to fetch ({len(syms)} symbols)")

    def work(t):
        sym, ws, we, _ = t
        try:
            return t, fetch_range(client, "SEGMENT_CASH", f"NSE-{sym}", ws, we, "CANDLE_INTERVAL_MIN_1"), None
        except Exception as e:  # noqa
            return t, None, f"{type(e).__name__}: {str(e)[:200]}"

    n_ok = n_fail = rows = 0
    with ThreadPoolExecutor(workers) as ex:
        for i, f in enumerate(as_completed([ex.submit(work, t) for t in tasks]), 1):
            (sym, ws, we, complete), df, err = f.result()
            key = f"s1m:{sym}:{ws}"
            if err:
                store.mark(key, "s1m", "failed", 0, err)
                n_fail += 1
            else:
                if len(df):
                    store.write_df(df, "stock_1m", safe(sym), f"{ws}.parquet")
                    expected = 375 * sum(1 for k in range((we - ws).days + 1) if (ws + timedelta(days=k)).weekday() < 5)
                    ok_rows = len(df) >= 0.6 * expected
                    if complete and not ok_rows:
                        print(f"[warn] {sym} window {ws}..{we}: only {len(df)} rows (expected ~{expected}); will retry next run")
                    store.mark(key, "s1m", ("done" if ok_rows else "short") if complete else "partial", len(df))
                n_ok += 1
                rows += len(df)
            if i % 100 == 0 or i == len(tasks):
                log(f"  {i}/{len(tasks)} windows | ok {n_ok} failed {n_fail} | rows {rows:,} | requests {client.n_requests}")
    # daily candles (used only to cross-check the 1-minute data in the quality report)
        log("[stock 1d] fetching daily candles for cross-checks")

    def daily_work(sym):
        key = f"s1d:{sym}:{start}:{end}"
        if end < today and store.done(key):
            return None
        try:
            frames, d = [], start
            while d <= end:
                d2 = min(d + timedelta(days=C.DAILY_WINDOW_DAYS - 1), end)
                frames.append(fetch_range(client, "SEGMENT_CASH", f"NSE-{sym}", d, d2, "CANDLE_INTERVAL_DAY"))
                d = d2 + timedelta(days=1)
            return sym, key, pd.concat(frames).drop_duplicates("ts").sort_values("ts"), None
        except Exception as e:  # noqa
            return sym, key, None, f"{type(e).__name__}: {str(e)[:200]}"

    with ThreadPoolExecutor(workers) as ex:
        for res in ex.map(daily_work, syms):
            if res is None:
                continue
            sym, key, df, err = res
            if err:
                store.mark(key, "s1d", "failed", 0, err)
            else:
                if len(df):
                    store.write_df(df, "stock_1d", f"{safe(sym)}.parquet")
                store.mark(key, "s1d", "done" if end < today else "partial", len(df))
    return n_ok, n_fail


# ------------------------------------------------------------------ phase 2: option candles
def underlying_daily(store, sym):
    df = store.read_glob("stock_1m", safe(sym), "*.parquet")
    if df.empty:
        return pd.DataFrame()
    df = df.sort_values("ts")
    df["day"] = df["ts"].dt.date
    g = df.groupby("day")
    return pd.DataFrame({"open": g["o"].first(), "high": g["h"].max(), "low": g["l"].min(), "close": g["c"].last()})


def rank_symbols(store, symbols):
    """Most liquid underlyings first (median daily turnover) so the best data lands earliest."""
    scores = {}
    for sym in symbols:
        df = store.read_glob("stock_1m", safe(sym), "*.parquet")
        if df.empty:
            scores[sym] = 0.0
            continue
        df["day"] = df["ts"].dt.date
        scores[sym] = float((df["c"] * df["v"]).groupby(df["day"]).sum().median())
    store.json_put(scores, "meta", "turnover_rank.json")
    return sorted(symbols, key=lambda s: -scores[s])


def get_expiries(client, store, sym, y0, y1):
    out = set()
    for y in range(y0, min(y1, date.today().year) + 1):
        name = f"{safe(sym)}_{y}.json"
        cached = store.json_get("meta", "expiries", name, max_age=None if y < date.today().year else 86400)
        if cached is None:
            try:
                r = client.call("get_expiries", exchange=client.const("EXCHANGE_NSE"), underlying_symbol=sym, year=y)
                cached = r.get("expiries", [])
                store.json_put(cached, "meta", "expiries", name)
            except Exception:  # noqa  (e.g. next year not available yet)
                cached = []
        out.update(date.fromisoformat(x) for x in cached)
    return sorted(out)


def get_contracts(client, store, sym, expiry):
    name = f"{safe(sym)}_{expiry}.json"
    past = expiry < date.today() - timedelta(days=2)
    cached = store.json_get("meta", "contracts", name, max_age=None if past else 86400)
    if cached is None:
        r = client.call("get_contracts", exchange=client.const("EXCHANGE_NSE"), underlying_symbol=sym,
                        expiry_date=str(expiry))
        cached = r.get("contracts", [])
        store.json_put(cached, "meta", "contracts", name)
    out = {}
    for gs in cached:
        if not (gs.endswith("-CE") or gs.endswith("-PE")):
            continue                      # futures (-FUT) have no strike
        _, strike, typ = plan.parse_contract(gs)
        out[(strike, typ)] = gs
    return out


def run_option_phase(client, store, symbols, start, end, band=1, roll_days=4, workers=4, log=print):
    order = rank_symbols(store, symbols)
    log(f"[options] {len(order)} symbols, strike band +/-{band}, roll when < {roll_days} calendar days to expiry")
    n_groups = n_fail = rows = 0
    for i, sym in enumerate(order, 1):
        try:
            daily = underlying_daily(store, sym)
            if daily.empty:
                log(f"  ({i}/{len(order)}) {sym}: no underlying data, skipped")
                continue
            daily = daily[(daily.index >= start) & (daily.index <= end)]
            expiries = get_expiries(client, store, sym, start.year, end.year + 1)
            day_exp = {d: plan.active_expiry(d, expiries, roll_days) for d in daily.index}
            contracts = {e: get_contracts(client, store, sym, e) for e in sorted({e for e in day_exp.values() if e})}
            per_exp = {}
            for day, row in daily.iterrows():
                e = day_exp[day]
                if not e or not contracts.get(e):
                    continue
                strikes = sorted({k[0] for k in contracts[e]})
                for s in plan.strike_band(strikes, row["low"], row["high"], band):
                    for typ in ("CE", "PE"):
                        g = contracts[e].get((s, typ))
                        if g:
                            per_exp.setdefault(e, {}).setdefault(g, []).append(day)
            sym_rows = 0
            for e, gmap in sorted(per_exp.items()):
                last_day = max(max(v) for v in gmap.values())
                key = f"opt:{sym}:{e}:{last_day}"
                if store.done(key):
                    continue
                tasks = [(g, a, b) for g, days in gmap.items()
                         for a, b in plan.merge_windows(days, C.OPT_MAX_SPAN_DAYS, C.OPT_MERGE_GAP_DAYS)]

                def work(t):
                    g, a, b = t
                    try:
                        df = fetch_range(client, "SEGMENT_FNO", g, a, b, "CANDLE_INTERVAL_MIN_1")
                        if len(df):
                            _, strike, typ = plan.parse_contract(g)
                            df.insert(1, "contract", g)
                            df.insert(2, "strike", strike)
                            df.insert(3, "typ", typ)
                        return df, None
                    except Exception as ex_:  # noqa
                        return None, f"{g}: {type(ex_).__name__}: {str(ex_)[:150]}"

                with ThreadPoolExecutor(workers) as ex:
                    results = list(ex.map(work, tasks))
                errs = [r[1] for r in results if r[1]]
                if errs:
                    store.mark(key, "opt", "failed", 0, f"{len(errs)} of {len(tasks)} requests failed; first: {errs[0]}")
                    n_fail += 1
                    continue
                frames = [r[0] for r in results if r[0] is not None and len(r[0])]
                df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
                if len(df):
                    store.write_df(df.sort_values(["contract", "ts"]), "opt_1m", safe(sym), f"{e}.parquet")
                store.mark(key, "opt", "done", len(df), f"{len(tasks)} requests")
                n_groups += 1
                sym_rows += len(df)
            rows += sym_rows
            log(f"  ({i}/{len(order)}) {sym}: {sym_rows:,} option rows | total {rows:,} | requests {client.n_requests}")
        except Exception as e:  # noqa
            n_fail += 1
            store.mark(f"opt:{sym}:ERR", "opt", "failed", 0, f"{type(e).__name__}: {str(e)[:200]}")
            log(f"  ({i}/{len(order)}) {sym}: FAILED {type(e).__name__}: {str(e)[:150]}")
    return n_groups, n_fail
