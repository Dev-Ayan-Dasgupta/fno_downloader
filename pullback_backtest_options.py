from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
from pullback_options import OptionSimParams, run_options_backtest


def main():
    ap=argparse.ArgumentParser(description='Options execution backtest for pullback signals')
    ap.add_argument('--signals', required=True)
    ap.add_argument('--contracts', required=True, help='CSV with signal_key and selected historical option contract')
    ap.add_argument('--underlying-1m', required=True, help='CSV: symbol,ts,o,h,l,c')
    ap.add_argument('--options-1m', required=True, help='CSV: option_symbol,ts,o,h,l,c')
    ap.add_argument('--out', default='data/backtest/pullback_options')
    ap.add_argument('--lots', type=int, default=None)
    ap.add_argument('--target-r', type=float, default=2.0)
    ap.add_argument('--eod', default='14:59')
    ap.add_argument('--slippage-grid', default='0,1,2,3,5', help='Comma-separated bps per side; scenarios, not measured fills')
    a=ap.parse_args(); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    signals=pd.read_csv(a.signals); contracts=pd.read_csv(a.contracts)
    underlying=pd.read_csv(a.underlying_1m); options=pd.read_csv(a.options_1m)
    grid=[float(x.strip()) for x in a.slippage_grid.split(',') if x.strip()]
    all_trades=[]
    for bps in grid:
        p=OptionSimParams(target_r=a.target_r,eod=a.eod,lots_override=a.lots,slippage_bps_per_side=bps)
        t=run_options_backtest(signals,contracts,underlying,options,p)
        t['slippage_bps_per_side']=bps
        all_trades.append(t)
    trades=pd.concat(all_trades,ignore_index=True) if all_trades else pd.DataFrame()
    trades.to_csv(out/'options_trades.csv',index=False)
    valid=trades[trades.net_pnl_rs.notna()]
    if not valid.empty:
        summary=valid.groupby(['slippage_bps_per_side','setup','direction'],as_index=False).agg(
            trades=('net_pnl_rs','size'),gross_pnl_rs=('gross_pnl_rs','sum'),
            net_pnl_rs=('net_pnl_rs','sum'),avg_net_pnl_rs=('net_pnl_rs','mean'),
            avg_groww_charges_rs=('groww_charges_rs','mean'),avg_slippage_rs=('slippage_rs','mean'),
            avg_premium_outlay_rs=('premium_outlay_rs','mean'),avg_lots=('lots','mean'),
            net_win_rate=('net_pnl_rs',lambda x:(x>0).mean()))
        summary.to_csv(out/'options_summary.csv',index=False)
    pd.Series(trades.exit_reason.value_counts()).rename_axis('exit_reason').reset_index(name='count').to_csv(out/'options_diagnostics.csv',index=False)
    (out/'sim_params.json').write_text(json.dumps({'target_r':a.target_r,'eod':a.eod,'lots_override':a.lots,
        'slippage_grid_bps_per_side':grid,'cost_model':'Groww NSE equity options published schedule; verify contract notes'},indent=2))
    print(f'Saved {len(trades)} scenario-trade rows to {out}')

if __name__=='__main__': main()
