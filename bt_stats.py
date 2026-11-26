"""Results summary: day-clustered uncertainty and clearly labelled trade-level diagnostics."""
import numpy as np
import pandas as pd


def cluster_ci(tr, col="r_net", n_boot=2000, seed=0):
    """95% CI for mean trade R, resampling whole trading days."""
    g = tr.groupby("day")[col].agg(["sum", "count"])
    sums, counts = g["sum"].to_numpy(), g["count"].to_numpy()
    if len(sums) < 2:
        return float("nan"), float("nan")
    idx = np.random.default_rng(seed).integers(0, len(sums), size=(n_boot, len(sums)))
    means = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    return tuple(np.percentile(means, [2.5, 97.5]))


def max_drawdown(r):
    """Drawdown of a sequential cumulative R series; not a capital-weighted portfolio DD."""
    c = np.r_[0.0, np.cumsum(np.asarray(r, dtype=float))]
    return float((np.maximum.accumulate(c) - c).max())


def summarize(tr):
    if tr is None or len(tr) == 0:
        return {"n": 0}
    tr = tr.sort_values(["signal_time", "symbol"], kind="stable").copy()
    r = tr["r_net"].to_numpy(dtype=float)
    wins, losses = r[r > 0].sum(), -r[r < 0].sum()
    lo, hi = cluster_ci(tr)
    daily = tr.groupby("day")["r_net"].sum().sort_index()
    month = tr.groupby(pd.to_datetime(tr["day"]).dt.to_period("M"))["r_net"].sum()
    by_stock = tr.groupby("symbol")["r_net"].sum()
    positive_stock = by_stock.clip(lower=0)
    positive_total = float(positive_stock.sum())
    if "u_reach_t2_before_stop" in tr:
        before_stop = tr[tr["u_reach_t2_before_stop"].astype(bool)]
        early_stop = float(before_stop["exit_reason"].eq("SL").mean()) if len(before_stop) else float("nan")
    else:
        early_stop = float("nan")
    tot = float(r.sum())
    out = dict(
        n=len(tr), days=int(tr["day"].nunique()), stocks=int(tr["symbol"].nunique()),
        win_rate=round(float((r > 0).mean()), 3),
        avg_r_gross=round(float(tr["r_gross"].mean()), 3),
        avg_r_net=round(float(r.mean()), 3),
        ci_lo=round(float(lo), 3), ci_hi=round(float(hi), 3),
        total_r_net=round(tot, 1),
        profit_factor=round(float(wins / losses), 2) if losses > 0 else float("inf"),
        trade_sequence_max_dd_r=round(max_drawdown(r), 1),
        daily_aggregate_max_dd_r=round(max_drawdown(daily.to_numpy()), 1),
        months_positive=round(float((month > 0).mean()), 2),
        top_stock_positive_r_share=round(float(positive_stock.max() / positive_total), 3)
            if positive_total > 0 else float("nan"),
        cost_in_r=round(float((tr["costs_rs"] / tr["risk_rs"]).mean()), 3),
        early_stop_rate=round(early_stop, 3) if np.isfinite(early_stop) else float("nan"),
        hit_t1=round(float(tr["hit_t1"].mean()), 3),
        hit_t2=round(float(tr["hit_t2"].mean()), 3),
        hit_t3=round(float(tr["hit_t3"].mean()), 3),
        hit_t4=round(float(tr["hit_t4"].mean()), 3),
        avg_mae_r=round(float(tr["mae_r"].mean()), 2),
        avg_mfe_r=round(float(tr["mfe_r"].mean()), 2),
    )
    if "u_reach_t2_any" in tr:
        sl = tr[tr["exit_reason"].eq("SL")]
        out["sl_then_t2_any_rate"] = round(float(sl["u_reach_t2_any"].mean()), 3) if len(sl) else float("nan")
        out["underlying_t2_any_rate"] = round(float(tr["u_reach_t2_any"].mean()), 3)
        out["underlying_t2_before_stop_rate"] = round(float(tr["u_reach_t2_before_stop"].mean()), 3)
    return out


def pass_check(s):
    if s.get("n", 0) == 0:
        return {}
    return {
        "avg_R_net > 0 with CI low > 0": bool(s["ci_lo"] > 0),
        "profit factor >= 1.2": bool(s["profit_factor"] >= 1.2),
        "positive in >= 60% of months": bool(s["months_positive"] >= 0.6),
        "no stock > 10% of positive R": bool(s.get("top_stock_positive_r_share", float("nan")) <= 0.10)
            if np.isfinite(s.get("top_stock_positive_r_share", float("nan"))) else False,
    }
