#!/usr/bin/env python3
"""Diagnostic comparison of pre-specified PE regime hypotheses.

Groups: A=all PE; B=PE with rs_30m < -0.003; C=B plus mixed NIFTY 1h trend.
Uses day-cluster bootstrap CIs; this is diagnostic, not a final Test-set decision.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def profit_factor(r: pd.Series) -> float:
    pos = float(r[r > 0].sum())
    neg = float(-r[r < 0].sum())
    return pos / neg if neg > 0 else (float('inf') if pos > 0 else float('nan'))


def max_drawdown(r: pd.Series) -> float:
    eq = r.fillna(0).cumsum()
    peak = eq.cummax().clip(lower=0)
    return float((eq - peak).min()) if len(eq) else float('nan')


def cluster_bootstrap_ci(df: pd.DataFrame, value_col: str, day_col: str,
                         reps: int = 5000, seed: int = 20261124) -> tuple[float, float]:
    # Resample whole trading days, preserving within-day trade dependence.
    daily = df.groupby(day_col, dropna=True)[value_col].agg(['sum', 'count'])
    if len(daily) < 2:
        return (float('nan'), float('nan'))
    sums = daily['sum'].to_numpy(float)
    counts = daily['count'].to_numpy(float)
    rng = np.random.default_rng(seed)
    n = len(daily)
    means = np.empty(reps)
    for i in range(reps):
        ix = rng.integers(0, n, n)
        means[i] = sums[ix].sum() / counts[ix].sum()
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def summarize(name: str, df: pd.DataFrame, reps: int) -> dict:
    r = pd.to_numeric(df['r_net'], errors='coerce').dropna()
    work = df.loc[r.index].copy()
    if 'entry_ts' in work.columns:
        work = work.sort_values('entry_ts', kind='stable')
        r = work['r_net']
    ci_lo, ci_hi = cluster_bootstrap_ci(work, 'r_net', 'day', reps=reps,
                                         seed=20261124 + len(work))
    days = work['day'].nunique()
    return {
        'group': name, 'trades': len(work), 'trading_days': days,
        'total_R': r.sum(), 'avg_R_per_trade': r.mean(),
        'day_cluster_CI_low': ci_lo, 'day_cluster_CI_high': ci_hi,
        'win_rate': (r > 0).mean(), 'profit_factor': profit_factor(r),
        'max_trade_sequence_DD_R': max_drawdown(r),
        'median_MFE_R': pd.to_numeric(work.get('mfe_r'), errors='coerce').median(),
        'median_MAE_R': pd.to_numeric(work.get('mae_r'), errors='coerce').median(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default='enriched_trades(1).csv')
    ap.add_argument('--outdir', default='hypothesis_test_output')
    ap.add_argument('--bootstrap-reps', type=int, default=5000)
    args = ap.parse_args()
    src = Path(args.input)
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(src)
    required = {'direction','rs_30m','nifty_1h_trend','r_net','day'}
    missing = required - set(df.columns)
    if missing: raise SystemExit(f'Missing required columns: {sorted(missing)}')
    df['r_net'] = pd.to_numeric(df['r_net'], errors='coerce')
    df['rs_30m'] = pd.to_numeric(df['rs_30m'], errors='coerce')
    df['day'] = pd.to_datetime(df['day'], errors='coerce').dt.strftime('%Y-%m-%d')
    pe = df[df['direction'].astype(str).str.upper().eq('PE')].copy()
    b = pe[pe['rs_30m'] < -0.003].copy()
    c = b[b['nifty_1h_trend'].astype(str).str.lower().eq('mixed')].copy()
    groups = [('A_all_PE', pe), ('B_PE_rs30m_below_minus_0.003', b),
              ('C_PE_rs_weak_and_NIFTY_1h_mixed', c)]
    summary = pd.DataFrame([summarize(n, g, args.bootstrap_reps) for n,g in groups])
    summary.to_csv(out/'hypothesis_group_summary.csv', index=False)

    # Month-level stability: report every month, including small samples.
    monthly_rows=[]
    for name,g in groups:
        for month,mg in g.groupby(pd.to_datetime(g['day']).dt.to_period('M').astype(str)):
            r=mg['r_net'].dropna()
            monthly_rows.append({'group':name,'month':month,'trades':len(r),
                'trading_days':mg.loc[r.index,'day'].nunique(), 'total_R':r.sum(),
                'avg_R':r.mean(), 'win_rate':(r>0).mean(), 'profit_factor':profit_factor(r)})
    pd.DataFrame(monthly_rows).to_csv(out/'hypothesis_monthly_summary.csv',index=False)

    # Exact inclusion diagnostics and coverage checks.
    diagnostics = {
        'input_rows':len(df), 'PE_rows':len(pe), 'PE_rs_filter_rows':len(b),
        'PE_rs_and_mixed_rows':len(c),
        'PE_missing_rs30m':int(pe['rs_30m'].isna().sum()),
        'PE_missing_nifty_1h_trend':int(pe['nifty_1h_trend'].isna().sum()),
        'PE_trend_values':';'.join(sorted(pe['nifty_1h_trend'].dropna().astype(str).unique())),
        'threshold_rs_30m':-0.003,
        'note':'Nested exploratory comparisons on validation data; not adjusted for subgroup search or multiple testing.'
    }
    pd.DataFrame([diagnostics]).to_csv(out/'hypothesis_diagnostics.csv',index=False)
    print(summary.to_string(index=False, float_format=lambda x:f'{x:.4f}'))
    print(f'\nWrote outputs to: {out.resolve()}')

if __name__ == '__main__': main()
