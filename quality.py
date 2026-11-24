"""Quality report over everything downloaded. Writes CSVs and QUALITY.md under data/reports/."""
import os
from collections import Counter
from datetime import time as dtime

import numpy as np
import pandas as pd

from fetch import safe

SESSION_MIN = 375


def _bad_ohlc(df):
    o, h, l, c = df["o"], df["h"], df["l"], df["c"]
    return (h < np.maximum(o, c)) | (l > np.minimum(o, c)) | (h < l) | (o <= 0) | (h <= 0) | (l <= 0) | (c <= 0) | df[["o", "h", "l", "c"]].isna().any(axis=1)


def stock_quality(store, symbols):
    per_sym_days, frames = {}, {}
    for sym in symbols:
        df = store.read_glob("stock_1m", safe(sym), "*.parquet")
        if df.empty:
            continue
        df = df.drop_duplicates()
        df["day"] = df["ts"].dt.date
        frames[sym] = df
        per_sym_days[sym] = set(df["day"])
    cnt = Counter(d for s in per_sym_days.values() for d in s)
    calendar = sorted(d for d, n in cnt.items() if n >= 0.5 * max(len(per_sym_days), 1))
    rows = []
    for sym, df in frames.items():
        days = per_sym_days[sym]
        lo, hi = min(days), max(days)
        exp = [d for d in calendar if lo <= d <= hi]
        n_per_day = df.groupby("day").size()
        t = df["ts"].dt.time
        d1 = store.read_glob("stock_1d", f"{safe(sym)}.parquet")
        r = {"symbol": sym, "first_day": lo, "last_day": hi, "days": len(days),
             "missing_days_vs_calendar": len([d for d in exp if d not in days]),
             "median_candles_per_day": float(n_per_day.median()),
             "days_under_90pct_coverage": int((n_per_day < 0.9 * SESSION_MIN).sum()),
             "bad_ohlc_rows": int(_bad_ohlc(df).sum()),
             "duplicate_timestamps": int(df["ts"].duplicated().sum()),
             "off_session_rows": int(((t < dtime(9, 15)) | (t > dtime(15, 30))).sum())}
        d = df.sort_values("ts")
        ret = d.groupby("day")["c"].pct_change().abs()
        r["jumps_over_8pct_in_1min"] = int((ret > 0.08).sum())
        g = d.groupby("day").agg(o=("o", "first"), h=("h", "max"), l=("l", "min"), c=("c", "last"), v=("v", "sum"))
        gap = (g["o"] / g["c"].shift(1) - 1).abs()
        r["overnight_gaps_over_25pct"] = ";".join(str(x) for x in gap[gap > 0.25].index[:5])
        if len(d1):
            d1 = d1.copy()
            d1["day"] = pd.to_datetime(d1["ts"]).dt.date
            m = g.join(d1.set_index("day")[["h", "l", "c", "v"]], rsuffix="_d", how="inner")
            if len(m):
                r["daily_vs_1m_high_diff_over_0.5pct"] = int(((m["h"] / m["h_d"] - 1).abs() > 0.005).sum())
                r["daily_vs_1m_low_diff_over_0.5pct"] = int(((m["l"] / m["l_d"] - 1).abs() > 0.005).sum())
                r["daily_vs_1m_close_diff_over_0.5pct"] = int(((m["c"] / m["c_d"] - 1).abs() > 0.005).sum())
                ratio = m["v"] / m["v_d"].replace(0, np.nan)
                r["daily_vs_1m_volume_off_over_10pct"] = int(((ratio - 1).abs() > 0.10).sum())
        rows.append(r)
    return pd.DataFrame(rows), calendar


def option_quality(store, symbols):
    cd_frames = []
    for sym in symbols:
        df = store.read_glob("opt_1m", safe(sym), "*.parquet")
        if df.empty:
            continue
        df = df.drop_duplicates(["contract", "ts"])
        df["day"] = df["ts"].dt.date
        df["vol_pos"] = (df["v"].fillna(0) > 0)
        df["oi_present"] = df["oi"].notna()
        df["bad"] = _bad_ohlc(df)
        df["oi_neg"] = (df["oi"].fillna(0) < 0)
        df = df.sort_values(["contract", "ts"])
        prev = df.groupby(["contract", "day"])["oi"].shift(1)
        df["oi_jump"] = ((df["oi"] > 5 * prev) & (prev > 0))
        g = df.groupby(["contract", "day"]).agg(
            strike=("strike", "first"), typ=("typ", "first"), n_candles=("ts", "size"), vol_pos=("vol_pos", "mean"),
            oi_present=("oi_present", "mean"), bad_rows=("bad", "sum"), oi_neg=("oi_neg", "sum"),
            oi_jumps=("oi_jump", "sum"), volume=("v", "sum")).reset_index()
        g.insert(0, "symbol", sym)
        g["coverage"] = g["n_candles"] / SESSION_MIN
        cd_frames.append(g)
    if not cd_frames:
        return pd.DataFrame(), pd.DataFrame()
    cd = pd.concat(cd_frames, ignore_index=True)
    s = cd.groupby("symbol").agg(contract_days=("coverage", "size"), median_coverage=("coverage", "median"),
                                 share_days_over_80pct=("coverage", lambda x: float((x >= 0.8).mean())),
                                 median_oi_present=("oi_present", "median"), median_volume=("volume", "median"),
                                 bad_rows=("bad_rows", "sum"), oi_negative=("oi_neg", "sum"),
                                 oi_jumps_over_5x=("oi_jumps", "sum")).reset_index()
    s["liquidity_tier"] = np.where(s["median_coverage"] >= 0.8, "A", np.where(s["median_coverage"] >= 0.5, "B", "C"))
    return cd, s.sort_values("median_coverage", ascending=False)


def build_report(store, symbols):
    out = store.path("reports")
    os.makedirs(out, exist_ok=True)
    sq, calendar = stock_quality(store, symbols)
    cd, osum = option_quality(store, symbols)
    sq.to_csv(os.path.join(out, "quality_stock.csv"), index=False)
    cd.to_csv(os.path.join(out, "quality_option_contract_days.csv"), index=False)
    osum.to_csv(os.path.join(out, "quality_option_symbol_summary.csv"), index=False)
    L = ["# Data quality report", "",
         f"Stocks with 1-minute data: {len(sq)} | trading days in calendar: {len(calendar)}",
         f"Symbols with option data: {len(osum)} | contract-days: {len(cd):,}", ""]
    if len(osum):
        t = osum["liquidity_tier"].value_counts().to_dict()
        L += ["## Option liquidity tiers (by median share of minutes with a candle)",
              f"A (>=80%): {t.get('A', 0)} | B (50-80%): {t.get('B', 0)} | C (<50%): {t.get('C', 0)}",
              "Tier C stocks have gappy option prices: exclude them from backtests, or require a liquidity filter.", "",
              "Best 10: " + ", ".join(osum.head(10)["symbol"]), "Worst 10: " + ", ".join(osum.tail(10)["symbol"]), ""]
    if len(sq):
        L += ["## Stock-level issues (counts across all stocks)"]
        for col in [c for c in sq.columns if c not in ("symbol", "first_day", "last_day", "days", "median_candles_per_day",
                                                       "overnight_gaps_over_25pct")]:
            bad = sq[sq[col] > 0]
            if len(bad):
                L.append(f"- {col}: {len(bad)} stocks affected (worst: {', '.join(bad.sort_values(col, ascending=False).head(3)['symbol'])})")
        gaps = sq[sq["overnight_gaps_over_25pct"].fillna("") != ""]
        if len(gaps):
            L.append(f"- overnight gaps over 25% (possible unadjusted split/bonus): " +
                     "; ".join(f"{r.symbol} {r.overnight_gaps_over_25pct}" for r in gaps.head(10).itertuples()))
        L.append("")
    L += ["## Known limitations (not detectable from the data)",
          "- Survivorship bias: the universe is today's F&O list. Stocks removed from F&O or delisted are missing.",
          "- Option candles seem to exist only for minutes with trades; missing minutes are 'no trade', not 'price unchanged'.",
          "- No historical bid/ask: the spread filter can only be tested live, not in backtests.",
          "- Greeks are not provided historically; compute them from price + IV model if needed.",
          "- Historical lot sizes are not provided; rupee P&L should use the lot size valid at the time. R-multiples are unaffected.",
          "- OI values missing in some candles: forward-fill within a day only, never across days."]
    with open(os.path.join(out, "QUALITY.md"), "w") as f:
        f.write("\n".join(L))
    return "\n".join(L)
