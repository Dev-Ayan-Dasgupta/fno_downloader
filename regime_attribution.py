#!/usr/bin/env python3
"""
Causal market-regime attribution for ORB validation trades.

Inputs:
  - CE trade CSV and PE trade CSV from the same frozen validation configuration
  - DATA_DIR/stock_1m/NIFTY/*.parquet
  - DATA_DIR/stock_1m/<SYMBOL>/*.parquet

Outputs:
  - enriched_trades.csv
  - regime_summary.csv
  - feature_coverage.csv

No strategy parameters are changed. All features are point-in-time:
  * Intraday VWAP and returns use 1m bars strictly before entry_ts.
  * 15m/60m features use only bars whose end <= entry_ts.
  * Daily trend uses the previous completed session only.
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd

import bt_data as D

def _resample_session(df1m, minutes):
    x = df1m.copy()
    x["minute"] = x["ts"].dt.hour * 60 + x["ts"].dt.minute - (9 * 60 + 15)
    x = x[(x["minute"] >= 0) & (x["minute"] < 375)]
    x["bucket"] = (x["minute"] // minutes).astype(int)
    g = x.groupby(["day", "bucket"], sort=True)
    out = g.agg(
        ts=("ts", "first"), o=("o", "first"), h=("h", "max"),
        l=("l", "min"), c=("c", "last"), v=("v", "sum"), n=("ts", "size")
    ).reset_index()
    out["end"] = out["ts"] + pd.to_timedelta(minutes, unit="m")
    # Discard incomplete final buckets; all other buckets must have their full minute count.
    out = out[out["n"] == minutes].copy()
    return out.sort_values("end").reset_index(drop=True)

def _rsi(s, period=14):
    delta = s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    ag = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    al = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = ag / al.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

def _add_indicators(bars):
    # Carry indicator state across sessions; resetting each day would make a 50-bar
    # EMA impossible on 15m/60m data and would discard legitimate prior-session context.
    z = bars.sort_values("end").copy()
    z["ema20"] = z["c"].ewm(span=20, adjust=False, min_periods=20).mean()
    z["ema50"] = z["c"].ewm(span=50, adjust=False, min_periods=50).mean()
    z["rsi14"] = _rsi(z["c"])
    z["trend"] = np.select(
        [(z["c"] > z["ema20"]) & (z["ema20"] > z["ema50"]),
         (z["c"] < z["ema20"]) & (z["ema20"] < z["ema50"])],
        ["bull", "bear"], default="mixed"
    )
    return z

def _daily_features(nifty1m):
    g = nifty1m.groupby("day")
    d = pd.DataFrame({"close": g["c"].last()}).sort_index()
    d["ema20"] = d["close"].ewm(span=20, adjust=False, min_periods=20).mean()
    # Shift so session D receives only information through D-1.
    d["daily_trend_prev"] = np.where(d["close"].shift(1) > d["ema20"].shift(1), "above_ema20",
                              np.where(d["close"].shift(1) < d["ema20"].shift(1), "below_ema20", "unknown"))
    d["prev_close"] = d["close"].shift(1)
    return d

def _last_completed(bars, ts):
    q = bars[bars["end"] <= ts]
    return None if q.empty else q.iloc[-1]

def _minute_snapshot(df, ts):
    # At an entry timestamp at the open of a minute, that minute's OHLCV is not yet known.
    hist = df[df["ts"] < ts]
    if hist.empty:
        return {}
    row = hist.iloc[-1]
    vol = hist["v"].fillna(0)
    tp = (hist["h"] + hist["l"] + hist["c"]) / 3
    vwap = (tp * vol).sum() / vol.sum() if vol.sum() > 0 else np.nan
    out = {"last_1m_close": float(row["c"]), "vwap": float(vwap)}
    out["above_vwap"] = "above" if row["c"] > vwap else ("below" if row["c"] < vwap else "at")
    for mins in (15, 30, 60):
        cutoff = ts - pd.Timedelta(minutes=int(mins), unit="min")
        old = hist[hist["ts"] <= cutoff]
        out[f"ret_{mins}m"] = float(row["c"] / old.iloc[-1]["c"] - 1) if not old.empty and old.iloc[-1]["c"] else np.nan
    return out

def _opening_range(df, day):
    x = df[(df["day"] == day) & (df["ts"].dt.hour * 60 + df["ts"].dt.minute < 9*60+30)]
    if x.empty:
        return np.nan
    return float(x["h"].max() - x["l"].min())

def _daily_atr(df1m):
    return D.daily_table(df1m)["atr14"].to_dict()

def _atr_prev(nifty1m):
    d = D.daily_table(nifty1m)
    return d["atr14"].to_dict()

def build_features(trades, data_dir):
    nifty = D.load_stock_1m("NIFTY", data_dir)
    if nifty.empty:
        raise RuntimeError(f"No NIFTY 1m data found under {data_dir}/stock_1m/NIFTY")
    nifty = nifty.sort_values("ts").reset_index(drop=True)
    n15 = _add_indicators(_resample_session(nifty, 15))
    n60 = _add_indicators(_resample_session(nifty, 60))
    ndaily = _daily_features(nifty)
    nifty_atr_prev = _atr_prev(nifty)
    nifty_or = {day: _opening_range(nifty, day) for day in nifty["day"].unique()}

    stock_cache = {}
    stock_bars = {}
    rows = []
    for _, tr in trades.iterrows():
        sym, day = str(tr["symbol"]), pd.to_datetime(tr["day"]).date()
        if sym not in stock_cache:
            stock_cache[sym] = D.load_stock_1m(sym, data_dir)
            if not stock_cache[sym].empty:
                stock_cache[sym] = stock_cache[sym].sort_values("ts").reset_index(drop=True)
                stock_bars[sym] = {
                    "15": _add_indicators(_resample_session(stock_cache[sym], 15)),
                    "60": _add_indicators(_resample_session(stock_cache[sym], 60)),
                    "atr_prev": _daily_atr(stock_cache[sym])
                }
            else:
                stock_bars[sym] = None
        ts = pd.to_datetime(tr["entry_ts"])
        n15r, n60r = _last_completed(n15, ts), _last_completed(n60, ts)
        s15r = _last_completed(stock_bars[sym]["15"], ts) if stock_bars[sym] else None
        s60r = _last_completed(stock_bars[sym]["60"], ts) if stock_bars[sym] else None
        nh = nifty[(nifty["day"] == day) & (nifty["ts"] < ts)]
        sh = stock_cache[sym]
        sh = sh[(sh["day"] == day) & (sh["ts"] < ts)] if not sh.empty else sh
        nm = _minute_snapshot(nh, ts) if not nh.empty else {}
        sm = _minute_snapshot(sh, ts) if not sh.empty else {}
        nr15 = _last_completed(n15, ts)
        nr60 = _last_completed(n60, ts)
        sr15 = _last_completed(stock_bars[sym]["15"], ts) if stock_bars[sym] else None
        sr60 = _last_completed(stock_bars[sym]["60"], ts) if stock_bars[sym] else None

        # Relative strength is stock return minus NIFTY return over the same lookback.
        f = {}
        for mins in (15, 30, 60):
            a, b = sm.get(f"ret_{mins}m", np.nan), nm.get(f"ret_{mins}m", np.nan)
            f[f"rs_{mins}m"] = a - b if pd.notna(a) and pd.notna(b) else np.nan

        f.update({
            "nifty_15m_trend": nr15["trend"] if nr15 is not None else "unavailable",
            "nifty_15m_rsi": nr15["rsi14"] if nr15 is not None else np.nan,
            "nifty_15m_close_ema20_pct": (nr15["c"] / nr15["ema20"] - 1) if nr15 is not None and pd.notna(nr15["ema20"]) else np.nan,
            "nifty_1h_trend": nr60["trend"] if nr60 is not None else "unavailable",
            "nifty_1h_rsi": nr60["rsi14"] if nr60 is not None else np.nan,
            "nifty_1h_close_ema20_pct": (nr60["c"] / nr60["ema20"] - 1) if nr60 is not None and pd.notna(nr60["ema20"]) else np.nan,
            "nifty_vwap_state": nm.get("above_vwap", "unavailable"),
            "stock_15m_trend": sr15["trend"] if sr15 is not None else "unavailable",
            "stock_1h_trend": sr60["trend"] if sr60 is not None else "unavailable",
            "stock_vwap_state": sm.get("above_vwap", "unavailable"),
            "nifty_ret_15m": nm.get("ret_15m", np.nan),
            "nifty_ret_30m": nm.get("ret_30m", np.nan),
            "nifty_ret_60m": nm.get("ret_60m", np.nan),
            "stock_ret_15m": sm.get("ret_15m", np.nan),
            "stock_ret_30m": sm.get("ret_30m", np.nan),
            "stock_ret_60m": sm.get("ret_60m", np.nan),
            "daily_trend_prev": ndaily.loc[day, "daily_trend_prev"] if day in ndaily.index else "unknown",
            "nifty_prev_atr14": nifty_atr_prev.get(day, np.nan),
            "stock_opening_range": _opening_range(stock_cache[sym], day) if not stock_cache[sym].empty else np.nan,
        })
        stock_atr = stock_bars[sym]["atr_prev"].get(day, np.nan) if stock_bars[sym] else np.nan
        nifty_atr = nifty_atr_prev.get(day, np.nan)
        f["stock_prev_atr14"] = stock_atr
        f["nifty_opening_range"] = nifty_or.get(day, np.nan)
        f["stock_or_width_atr"] = f["stock_opening_range"] / stock_atr if pd.notna(stock_atr) and stock_atr > 0 else np.nan
        f["nifty_or_width_atr"] = f["nifty_opening_range"] / nifty_atr if pd.notna(nifty_atr) and nifty_atr > 0 else np.nan
        rows.append(f)

    feat = pd.DataFrame(rows, index=trades.index)
    return pd.concat([trades.reset_index(drop=True), feat.reset_index(drop=True)], axis=1)

def _bucket(s, bins, labels):
    return pd.cut(pd.to_numeric(s, errors="coerce"), bins=bins, labels=labels, include_lowest=True).astype("object").fillna("unknown")

def summarize_regimes(df):
    specs = {
        "nifty_15m_trend": None, "nifty_1h_trend": None,
        "nifty_vwap_state": None, "stock_15m_trend": None,
        "stock_1h_trend": None, "stock_vwap_state": None,
        "daily_trend_prev": None,
    }
    x = df.copy()
    x["entry_hour"] = pd.to_datetime(x["entry_ts"]).dt.hour
    x["entry_time_bucket"] = pd.cut(pd.to_datetime(x["entry_ts"]).dt.hour*60 + pd.to_datetime(x["entry_ts"]).dt.minute,
        bins=[0, 10*60+30, 11*60+30, 12*60+30, 16*60], labels=["before_10:30","10:30-11:30","11:30-12:30","after_12:30"])
    x["nifty_15m_rsi_bucket"] = _bucket(x["nifty_15m_rsi"], [0,30,45,55,70,100], ["<=30","30-45","45-55","55-70",">70"])
    x["nifty_1h_rsi_bucket"] = _bucket(x["nifty_1h_rsi"], [0,30,45,55,70,100], ["<=30","30-45","45-55","55-70",">70"])
    x["rs_30m_bucket"] = _bucket(x["rs_30m"], [-np.inf,-.01,-.003,.003,.01,np.inf], ["strong_underperformer","underperformer","neutral","outperformer","strong_outperformer"])
    x["stock_or_atr_bucket"] = _bucket(x["stock_or_width_atr"], [-np.inf,.05,.10,.20,.35,np.inf], ["<.05",".05-.10",".10-.20",".20-.35",">.35"])
    x["nifty_or_atr_bucket"] = _bucket(x["nifty_or_width_atr"], [-np.inf,.05,.10,.20,.35,np.inf], ["<.05",".05-.10",".10-.20",".20-.35",">.35"])
    x["nifty_15m_distance_ema20_bucket"] = _bucket(x["nifty_15m_close_ema20_pct"], [-np.inf,-.005,-.001,.001,.005,np.inf], ["far_below","below","near","above","far_above"])
    dims = list(specs) + ["entry_time_bucket","nifty_15m_rsi_bucket","nifty_1h_rsi_bucket","rs_30m_bucket","stock_or_atr_bucket","nifty_or_atr_bucket","nifty_15m_distance_ema20_bucket"]
    out=[]
    for dim in dims:
        for (direction, regime), g in x.groupby(["direction", dim], observed=True, dropna=False):
            r=pd.to_numeric(g["r_net"],errors="coerce").dropna()
            win=r[r>0].sum(); loss=-r[r<0].sum()
            out.append({
                "feature":dim, "regime":str(regime), "direction":direction, "n":len(r),
                "days":g.loc[r.index,"day"].nunique() if len(r) else 0,
                "avg_net_r":r.mean() if len(r) else np.nan,
                "total_net_r":r.sum() if len(r) else np.nan,
                "win_rate":(r>0).mean() if len(r) else np.nan,
                "profit_factor":win/loss if loss>0 else np.nan,
                "avg_mfe_r":pd.to_numeric(g["mfe_r"],errors="coerce").mean(),
                "avg_mae_r":pd.to_numeric(g["mae_r"],errors="coerce").mean(),
            })
    return pd.DataFrame(out).sort_values(["feature","direction","n"],ascending=[True,True,False])

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--ce", required=True, help="CE-only trades CSV")
    ap.add_argument("--pe", required=True, help="PE-only trades CSV")
    ap.add_argument("--data-dir", default=os.getenv("DATA_DIR","data"))
    ap.add_argument("--out", default="data/backtest/regime_attribution")
    a=ap.parse_args()
    ce=pd.read_csv(a.ce); pe=pd.read_csv(a.pe)
    trades=pd.concat([ce,pe],ignore_index=True)
    if set(trades["direction"].dropna().unique()) - {"CE","PE"}:
        raise SystemExit("Unexpected direction values in trade files")
    if not {"entry_ts","symbol","day","r_net","mfe_r","mae_r"}.issubset(trades.columns):
        raise SystemExit("Trade CSV missing required columns")
    if trades["execution_mode"].nunique()!=1 or trades["intrabar_mode"].nunique()!=1:
        raise SystemExit("CE/PE files have different execution settings")
    os.makedirs(a.out,exist_ok=True)
    enriched=build_features(trades,a.data_dir)
    enriched.to_csv(os.path.join(a.out,"enriched_trades.csv"),index=False)
    summary=summarize_regimes(enriched)
    summary.to_csv(os.path.join(a.out,"regime_summary.csv"),index=False)
    cov=pd.DataFrame({"feature":enriched.columns,"non_null":enriched.notna().sum().values,
                      "coverage_pct":(enriched.notna().mean().values*100).round(1)})
    cov.to_csv(os.path.join(a.out,"feature_coverage.csv"),index=False)
    print(f"Trades: {len(enriched)} | CE: {(enriched.direction=='CE').sum()} | PE: {(enriched.direction=='PE').sum()}")
    print(f"Date range: {enriched.day.min()} to {enriched.day.max()}")
    print(f"Saved outputs to: {a.out}")
    print("\nTop regime rows with >= 8 trades:")
    print(summary[summary.n>=8].sort_values("avg_net_r",ascending=False).head(25).to_string(index=False))

if __name__=="__main__":
    main()
