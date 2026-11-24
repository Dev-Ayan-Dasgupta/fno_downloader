"""Results summary: R-multiples, day-clustered confidence interval, drawdown, concentration, pass-criteria checks."""
import numpy as np
import pandas as pd


def cluster_ci(tr, col="r_net", n_boot=2000, seed=0):
    """95% interval for the average R, resampling whole TRADING DAYS (signals on one day share one market move)."""
    g = tr.groupby("day")[col].agg(["sum", "count"])
    s, n = g["sum"].to_numpy(), g["count"].to_numpy()
    if len(s) < 2:
        return float("nan"), float("nan")
    idx = np.random.default_rng(seed).integers(0, len(s), size=(n_boot, len(s)))
    means = s[idx].sum(axis=1) / n[idx].sum(axis=1)
    return tuple(np.percentile(means, [2.5, 97.5]))


def max_drawdown(r):
    c = np.r_[0.0, np.cumsum(r)]
    return float((np.maximum.accumulate(c) - c).max())


def summarize(tr):
    if tr is None or len(tr) == 0:
        return {"n": 0}
    tr = tr.sort_values("signal_time")
    r = tr["r_net"].to_numpy()
    wins, losses = r[r > 0].sum(), -r[r < 0].sum()
    lo, hi = cluster_ci(tr)
    month = tr.groupby(pd.to_datetime(tr["day"]).dt.to_period("M"))["r_net"].sum()
    by_stock = tr.groupby("symbol")["r_net"].sum()
    tot = float(r.sum())
    reach = tr[tr["u_reach_t2"]]
    out = dict(n=len(tr), days=int(tr["day"].nunique()), stocks=int(tr["symbol"].nunique()),
               win_rate=round(float((r > 0).mean()), 3), avg_r_gross=round(float(tr["r_gross"].mean()), 3),
               avg_r_net=round(float(r.mean()), 3), ci_lo=round(float(lo), 3), ci_hi=round(float(hi), 3),
               total_r_net=round(tot, 1), profit_factor=round(float(wins / losses), 2) if losses > 0 else float("inf"),
               max_dd_r=round(max_drawdown(r), 1), months_positive=round(float((month > 0).mean()), 2),
               top_stock_share=round(float(by_stock.max() / tot), 3) if tot > 0 else float("nan"),
               cost_in_r=round(float((tr["costs_rs"] / tr["risk_rs"]).mean()), 3),
               early_stop_rate=round(float(reach["exit_reason"].eq("SL").mean()), 3) if len(reach) else float("nan"),
               hit_t1=round(float(tr["hit_t1"].mean()), 3), hit_t2=round(float(tr["hit_t2"].mean()), 3),
               hit_t3=round(float(tr["hit_t3"].mean()), 3), hit_t4=round(float(tr["hit_t4"].mean()), 3),
               avg_mae_r=round(float(tr["mae_r"].mean()), 2), avg_mfe_r=round(float(tr["mfe_r"].mean()), 2))
    return out


def pass_check(s):
    """Spec section 8 criteria that can be read from one run (control and sensitivity runs are checked separately)."""
    if s.get("n", 0) == 0:
        return {}
    return {"avg_R_net > 0 with CI low > 0": bool(s["ci_lo"] > 0),
            "profit factor >= 1.2": bool(s["profit_factor"] >= 1.2),
            "positive in >= 60% of months": bool(s["months_positive"] >= 0.6),
            "no stock > 10% of total R": bool(s["top_stock_share"] <= 0.10) if s["top_stock_share"] == s["top_stock_share"] else False}
