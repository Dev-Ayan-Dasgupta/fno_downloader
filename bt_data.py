"""
Backtest data layer: loads what downloader.py saved and applies every cleaning rule from the spec.
  - drops bars outside 09:15-15:30, NaN/non-positive prices, inconsistent OHLC rows, duplicate timestamps
  - never uses information from the future (ATR, RVOL baseline and liquidity are shifted by one session)
  - OI: forward-filled inside the day for at most 5 minutes, never across days
  - excludes sessions near probable unadjusted splits/bonuses
  - fetches missing option contract-days on demand (needed for carried positions) and caches them
Put this file in the same folder as downloader.py (it imports fetch.py / plan.py from there).
"""
import glob
import json
import os
from datetime import date, datetime, time, timedelta

import numpy as np
import pandas as pd

import bt_config as B

OHLC = ["o", "h", "l", "c"]


def _safe(sym):
    return sym.replace("&", "_and_").replace("/", "_")


def _read(pattern):
    files = sorted(glob.glob(pattern))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True) if files else pd.DataFrame()


def _clean_bars(df):
    """Common cleaning for stock and option 1-minute bars."""
    if df.empty:
        return df
    df = df.copy()
    df["ts"] = pd.to_datetime(df["ts"])
    t = df["ts"].dt.time
    df = df[(t >= time(9, 15)) & (t <= time(15, 30))]
    df = df.dropna(subset=OHLC)
    df = df[(df[OHLC] > 0).all(axis=1)]
    ok = (df["h"] >= df[["o", "c", "l"]].max(axis=1)) & (df["l"] <= df[["o", "c", "h"]].min(axis=1))
    df = df[ok]
    df["day"] = df["ts"].dt.date
    return df


# --------------------------------------------------------------------------- underlying
def load_stock_1m(sym, data_dir=None):
    root = data_dir or B.data_dir()
    df = _clean_bars(_read(f"{root}/stock_1m/{_safe(sym)}/*.parquet"))
    if df.empty:
        return df
    return df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)


def to_5m(df1m):
    """5-minute bars aligned to 09:15. `ts` is the bar START; `end` is when the bar is complete (use only completed bars)."""
    df1m = df1m[df1m["ts"].dt.time < time(15, 30)]          # the lone 15:30 print would form a 76th stray bar
    g = df1m.assign(bar=df1m["ts"].dt.floor("5min")).groupby("bar")
    out = g.agg(o=("o", "first"), h=("h", "max"), l=("l", "min"), c=("c", "last"), v=("v", "sum"), n=("ts", "size"))
    out["end"] = out.index + pd.Timedelta(minutes=5)
    out = out.reset_index().rename(columns={"bar": "ts"})
    out["day"] = out["ts"].dt.date
    return out


def add_vwap(df):
    """Session VWAP (typical price weighted by volume), reset each day. Works on 1-minute or 5-minute bars."""
    d = df.copy()
    tpv = (d["h"] + d["l"] + d["c"]) / 3 * d["v"].fillna(0)
    g = d.assign(_pv=tpv, _v=d["v"].fillna(0)).groupby("day")
    d["vwap"] = g["_pv"].cumsum() / g["_v"].cumsum().replace(0, np.nan)
    return d


def daily_table(df1m):
    """Daily OHLCV plus prior-day levels. ATR(14) for day D uses days BEFORE D only."""
    g = df1m.groupby("day")
    d = pd.DataFrame({"open": g["o"].first(), "high": g["h"].max(), "low": g["l"].min(),
                      "close": g["c"].last(), "volume": g["v"].sum()})
    pc = d["close"].shift(1)
    tr = np.maximum(d["high"] - d["low"], np.maximum((d["high"] - pc).abs(), (d["low"] - pc).abs()))
    d["atr14"] = tr.rolling(14, min_periods=14).mean().shift(1)
    d["atr14_pct"] = d["atr14"] / pc
    d["prev_close"], d["prev_high"], d["prev_low"] = pc, d["high"].shift(1), d["low"].shift(1)
    d["gap"] = (d["open"] / pc - 1).abs()
    return d


def exclusion_days(daily, gap=None, before=None, after=None):
    """Sessions from `before` sessions ahead of an overnight gap above `gap` until `after` sessions behind it
    (probable unadjusted split/bonus: ATR(14) and the 20-session volume baseline stay contaminated that long)."""
    gap = B.SPLIT_GAP if gap is None else gap
    before = B.SPLIT_BEFORE if before is None else before
    after = B.SPLIT_AFTER if after is None else after
    idx, bad = list(daily.index), set()
    for i, g in enumerate(daily["gap"].to_numpy()):
        if g == g and g > gap:
            bad.update(idx[max(0, i - before): i + after + 1])
    return bad


def minute_index(ts):
    """Minutes since 09:15 (0..375)."""
    return ts.hour * 60 + ts.minute - (9 * 60 + 15)


def rvol_table(df1m, window=20):
    """Relative volume: cumulative volume through each session minute / mean of the same quantity over the previous
    `window` sessions. Rows = day, columns = minute index. Point-in-time (baseline excludes the day itself)."""
    d = df1m.assign(m=df1m["ts"].dt.hour * 60 + df1m["ts"].dt.minute - (9 * 60 + 15))
    cum = d.pivot_table(index="day", columns="m", values="v", aggfunc="sum").reindex(columns=range(0, 376))
    cum = cum.fillna(0).cumsum(axis=1)
    base = cum.rolling(window, min_periods=max(window // 2, 1)).mean().shift(1)
    return (cum / base).replace([np.inf, -np.inf], np.nan)


def opening_range(df1m, minutes=15):
    m = df1m["ts"].dt.hour * 60 + df1m["ts"].dt.minute - (9 * 60 + 15)
    w = df1m[m < minutes]
    g = w.groupby("day")
    return pd.DataFrame({"or_high": g["h"].max(), "or_low": g["l"].min()})


# --------------------------------------------------------------------------- options
def parse_contract(gs):
    """'NSE-BAJAJ-AUTO-30Dec25-1500-CE' -> (expiry date, strike, 'CE')"""
    _, e, k, t = gs.rsplit("-", 3)
    return datetime.strptime(e, "%d%b%y").date(), float(k), t


def _ffill_oi(df):
    """Forward-fill OI within (contract, day) for at most OI_FFILL_MIN minutes."""
    df = df.sort_values(["contract", "ts"]).copy()
    df["oi_ts"] = df["ts"].where(df["oi"].notna())
    g = df.groupby(["contract", "day"])
    f_oi, f_ts = g["oi"].ffill(), g["oi_ts"].ffill()
    age = (df["ts"] - f_ts).dt.total_seconds() / 60
    df["oi_f"] = f_oi.where(age <= B.OI_FFILL_MIN)
    return df.drop(columns=["oi_ts"])


def _clean_options(df):
    if df.empty:
        return df
    df = _clean_bars(df)
    if df.empty:
        return df
    df["oi"] = pd.to_numeric(df["oi"], errors="coerce")
    df = df.drop_duplicates(["contract", "ts"])
    df = _ffill_oi(df)
    exp = {c: parse_contract(c)[0] for c in df["contract"].unique()}
    df["expiry"] = df["contract"].map(exp)
    return df.reset_index(drop=True)


def load_option_1m(sym, data_dir=None):
    root = data_dir or B.data_dir()
    raw = pd.concat([_read(f"{root}/opt_1m/{_safe(sym)}/*.parquet"),
                     _read(f"{root}/opt_extra/{_safe(sym)}/*.parquet")], ignore_index=True)
    return _clean_options(raw)


class OptionStore:
    """Option candles for one underlying with fast (contract, day) access and on-demand fetching of missing days.
    Units: volume is in SHARES; OI is lots x lot size / 100 (historical) - use OI only as percentage changes."""

    def __init__(self, sym, client=None, data_dir=None):
        self.sym, self.client = sym, client
        self.root = data_dir or B.data_dir()
        self.df = load_option_1m(sym, self.root)
        self._idx = {k: v for k, v in self.df.groupby(["contract", "day"]).indices.items()} if len(self.df) else {}
        self._extra = {}                                            # (contract, day) -> DataFrame fetched this session
        self._tried_path = f"{self.root}/opt_extra/{_safe(sym)}/tried.json"
        self._tried = set(json.load(open(self._tried_path))) if os.path.exists(self._tried_path) else set()

    def contracts_index(self):
        """(expiry, strike, 'CE'/'PE') -> contract name; plus sorted expiries and strikes per (expiry, type)."""
        if not hasattr(self, "_cidx"):
            m = {}
            for c in (self.df["contract"].unique() if len(self.df) else []):
                e, k, t = parse_contract(c)
                m[(e, k, t)] = c
            self._cidx = m
            self.expiries = sorted({e for e, _, _ in m})
            st = {}
            for (e, k, t) in m:
                st.setdefault((e, t), set()).add(k)
            self.strikes = {key: sorted(v) for key, v in st.items()}
        return self._cidx

    def cidx_get(self, expiry, strike, typ):
        return self.contracts_index().get((expiry, strike, typ))

    def has(self, contract, day):
        return (contract, day) in self._idx or (contract, day) in self._extra

    def get(self, contract, day):
        if (contract, day) in self._idx:
            return self.df.iloc[self._idx[(contract, day)]].sort_values("ts")
        return self._extra.get((contract, day), pd.DataFrame(columns=self.df.columns))

    def ensure(self, contract, days):
        """Make sure candles exist for these contract-days; fetch the missing ones. Returns number of days requested."""
        miss = sorted(d for d in set(days) if not self.has(contract, d) and f"{contract}|{d}" not in self._tried)
        if not miss or self.client is None:
            return 0
        import plan
        from fetch import fetch_range
        frames = []
        for a, b in plan.merge_windows(miss, 28, 5):
            df = fetch_range(self.client, "SEGMENT_FNO", contract, a, b, "CANDLE_INTERVAL_MIN_1")
            if len(df):
                _, strike, typ = parse_contract(contract)
                df.insert(1, "contract", contract)
                df.insert(2, "strike", strike)
                df.insert(3, "typ", typ)
                frames.append(df)
        if frames:
            new = _clean_options(pd.concat(frames, ignore_index=True))
            if len(new):
                os.makedirs(os.path.dirname(self._tried_path), exist_ok=True)
                raw = pd.concat(frames, ignore_index=True)
                raw.to_parquet(f"{self.root}/opt_extra/{_safe(self.sym)}/{_safe(contract)}_{miss[0]}_{miss[-1]}.parquet", index=False)
                for (c, d), g in new.groupby(["contract", "day"]):
                    self._extra[(c, d)] = g.sort_values("ts")
        self._tried.update(f"{contract}|{d}" for d in miss)
        os.makedirs(os.path.dirname(self._tried_path), exist_ok=True)
        json.dump(sorted(self._tried), open(self._tried_path, "w"))
        return len(miss)

    def coverage_by_day(self):
        """Median candle coverage (share of the 375 session minutes with a candle) across that day's contracts."""
        if self.df.empty:
            return pd.Series(dtype=float)
        cd = self.df.groupby(["day", "contract"]).size() / B.SESSION_MINUTES
        return cd.groupby(level="day").median()

    def liquidity(self):
        """Point-in-time stock-level option liquidity: median of the previous LIQ_WINDOW sessions' coverage (shifted)."""
        c = self.coverage_by_day()
        return c.rolling(B.LIQ_WINDOW, min_periods=max(B.LIQ_WINDOW // 2, 1)).median().shift(1)
