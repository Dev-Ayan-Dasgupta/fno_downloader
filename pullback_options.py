"""Options execution simulator for Pullback Strategy v1.5.

Signals and stop/target triggers are defined on the underlying. A selected long
CE/PE contract is bought at the exact signal entry minute and sold on the first
available option bar after the underlying trigger minute. This deliberately
avoids pretending that an underlying price level is an option-premium stop.

Inputs:
  signals: output of pullback_signals.py
  contracts: signal_key, option_symbol, option_type, strike, expiry, lot_size, lots
  underlying_1m: symbol, ts, o, h, l, c
  options_1m: option_symbol, ts, o, h, l, c

All timestamps are minute-bar start timestamps. Contract selection is external
and must be based only on information available at entry (e.g. historical chain
and a pre-registered delta/ATM rule). No current-chain substitution is allowed.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class OptionSimParams:
    target_r: float = 2.0
    eod: str = "14:59"
    lots_override: Optional[int] = None
    slippage_bps_per_side: float = 2.0
    brokerage_per_order_rs: float = 20.0
    stt_sell_rate: float = 0.0015       # 0.15% of option premium turnover, from 1-Apr-2026
    stamp_buy_rate: float = 0.00003     # 0.003% of premium turnover
    exchange_rate: float = 0.0003503   # Groww/NSE options rate: 0.03503% of premium turnover
    sebi_rate: float = 0.000001        # 0.0001%
    ipft_rate: float = 0.000005        # 0.0005% NSE options
    gst_rate: float = 0.18


def make_signal_key(row) -> str:
    return f"{row['symbol']}|{row['day']}|{row['direction']}|{pd.Timestamp(row['entry_ts']).strftime('%Y-%m-%d %H:%M:%S')}"


def groww_option_charges(buy_turnover: float, sell_turnover: float, p: OptionSimParams) -> dict:
    brokerage = 2.0 * p.brokerage_per_order_rs
    stt = sell_turnover * p.stt_sell_rate
    stamp = buy_turnover * p.stamp_buy_rate
    exchange = (buy_turnover + sell_turnover) * p.exchange_rate
    sebi = (buy_turnover + sell_turnover) * p.sebi_rate
    ipft = (buy_turnover + sell_turnover) * p.ipft_rate
    gst = p.gst_rate * (brokerage + exchange + sebi + ipft)
    total = brokerage + stt + stamp + exchange + sebi + ipft + gst
    return dict(brokerage_rs=brokerage, stt_rs=stt, stamp_duty_rs=stamp,
                exchange_charges_rs=exchange, sebi_charges_rs=sebi,
                ipft_charges_rs=ipft, gst_rs=gst, groww_charges_rs=total)


def _valid(df):
    x = df[['o','h','l','c']].apply(pd.to_numeric, errors='coerce')
    return (x.notna().all(axis=1) & (x > 0).all(axis=1)
            & (x.h >= x[['o','c','l']].max(axis=1))
            & (x.l <= x[['o','c','h']].min(axis=1)))


def _empty(signal, contract, reason):
    return {**signal, **contract, 'exit_policy': f"UNDERLYING_{'1.5R' if False else 'TARGET'}",
            'exit_reason': reason, 'entry_option_price': np.nan, 'exit_option_price': np.nan,
            'quantity_units': np.nan, 'lots': np.nan, 'buy_turnover_rs': np.nan,
            'sell_turnover_rs': np.nan, 'gross_pnl_rs': np.nan, 'slippage_rs': np.nan,
            'net_pnl_rs': np.nan, 'gross_return_pct_on_premium': np.nan}


def simulate_option_trade(signal: dict, contract: dict, underlying_1m: pd.DataFrame,
                          options_1m: pd.DataFrame, params: Optional[OptionSimParams]=None) -> dict:
    p = params or OptionSimParams()
    required = ['option_symbol','option_type','strike','expiry','lot_size','lots']
    if any(k not in contract or pd.isna(contract[k]) for k in required):
        raise ValueError(f"Contract row must contain {required}")
    if contract['option_type'] != signal['direction']:
        raise ValueError('option_type must match signal direction (CE/PE)')
    lot_size, lots = int(contract['lot_size']), int(p.lots_override or contract['lots'])
    if lot_size <= 0 or lots <= 0:
        raise ValueError('lot_size and lots must be positive integers')
    qty = lot_size * lots
    entry_ts = pd.Timestamp(signal['entry_ts'])
    day = str(signal['day'])
    und = underlying_1m.copy(); und['ts'] = pd.to_datetime(und.ts)
    und = und[(und.symbol == signal['symbol']) & (und.ts >= entry_ts)
              & (und.ts.dt.strftime('%Y-%m-%d') == day)].sort_values('ts')
    und = und[_valid(und)].reset_index(drop=True)
    opt = options_1m.copy(); opt['ts'] = pd.to_datetime(opt.ts)
    opt = opt[(opt.option_symbol == contract['option_symbol']) & (opt.ts >= entry_ts)
              & (opt.ts.dt.strftime('%Y-%m-%d') == day)].sort_values('ts')
    opt = opt[_valid(opt)].drop_duplicates('ts', keep='last').set_index('ts')
    if und.empty or entry_ts not in opt.index or und.iloc[0].ts != entry_ts:
        return _empty(signal, contract, 'MISSING_ENTRY_BAR')
    entry_px = float(opt.loc[entry_ts, 'o'])
    if not np.isfinite(entry_px) or entry_px <= 0:
        return _empty(signal, contract, 'INVALID_OPTION_ENTRY')

    direction = signal['direction']
    sign = 1 if direction == 'CE' else -1
    spot_entry, risk = float(signal['entry_price']), float(signal['initial_risk'])
    stop = float(signal['initial_stop'])
    target = spot_entry + sign * p.target_r * risk
    trigger_ts, reason = None, None
    last_und_ts = None
    for b in und.itertuples():
        if b.ts.strftime('%H:%M') > p.eod: break
        last_und_ts = b.ts
        stop_hit = b.l <= stop if direction == 'CE' else b.h >= stop
        target_hit = b.h >= target if direction == 'CE' else b.l <= target
        if stop_hit: trigger_ts, reason = b.ts, 'UNDERLYING_STOP'; break
        if target_hit: trigger_ts, reason = b.ts, f'UNDERLYING_TARGET_{p.target_r:g}R'; break
    if trigger_ts is not None:
        # Exit on the next minute's option open after the underlying trigger bar.
        later = opt[opt.index > trigger_ts]
        if later.empty:
            return _empty(signal, contract, 'MISSING_OPTION_EXIT_AFTER_TRIGGER')
        exit_ts = later.index[0]
        exit_px = float(later.iloc[0].o)
    else:
        # EOD liquidation at the last available option close no later than EOD.
        eligible = opt[opt.index.strftime('%H:%M') <= p.eod]
        if eligible.empty:
            return _empty(signal, contract, 'MISSING_EOD_OPTION_BAR')
        exit_ts = eligible.index[-1]
        exit_px = float(eligible.iloc[-1].c)
        reason = 'EOD'
    slip = p.slippage_bps_per_side / 10000.0
    buy_fill, sell_fill = entry_px * (1 + slip), exit_px * (1 - slip)
    buy_turnover, sell_turnover = buy_fill * qty, sell_fill * qty
    costs = groww_option_charges(buy_turnover, sell_turnover, p)
    slippage_rs = ((buy_fill-entry_px) + (exit_px-sell_fill)) * qty
    gross = (exit_px-entry_px) * qty
    net = gross - slippage_rs - costs['groww_charges_rs']
    premium_outlay = entry_px * qty
    return {**signal, **contract, 'exit_policy': f'UNDERLYING_{p.target_r:g}R',
            'exit_reason': reason, 'entry_option_ts': entry_ts, 'exit_option_ts': exit_ts,
            'entry_option_price': entry_px, 'exit_option_price': exit_px,
            'entry_fill': buy_fill, 'exit_fill': sell_fill, 'lot_size': lot_size,
            'lots': lots, 'quantity_units': qty, 'premium_outlay_rs': premium_outlay,
            'buy_turnover_rs': buy_turnover, 'sell_turnover_rs': sell_turnover,
            'gross_pnl_rs': gross, 'slippage_rs': slippage_rs,
            **costs, 'total_costs_rs': slippage_rs + costs['groww_charges_rs'],
            'net_pnl_rs': net,
            'gross_return_pct_on_premium': gross/premium_outlay*100 if premium_outlay else np.nan,
            'net_return_pct_on_premium': net/premium_outlay*100 if premium_outlay else np.nan,
            'underlying_stop': stop, 'underlying_target': target,
            'underlying_trigger_ts': trigger_ts}


def run_options_backtest(signals: pd.DataFrame, contracts: pd.DataFrame,
                         underlying_1m: pd.DataFrame, options_1m: pd.DataFrame,
                         params: Optional[OptionSimParams]=None) -> pd.DataFrame:
    p = params or OptionSimParams()
    s = signals.copy(); s['signal_key'] = s.apply(make_signal_key, axis=1)
    c = contracts.copy()
    contract_cols = ['signal_key','option_symbol','option_type','strike','expiry','lot_size','lots']
    missing = [x for x in contract_cols if x not in c.columns]
    if missing:
        raise ValueError(f'contracts.csv missing columns: {missing}')
    c = c[contract_cols]
    if c.signal_key.duplicated().any():
        raise ValueError('contracts.csv must have exactly one selected contract per signal_key')
    merged = s.merge(c, on='signal_key', how='left', validate='one_to_one', indicator=True)
    rows=[]
    for r in merged.to_dict('records'):
        if r['_merge'] != 'both':
            rows.append(_empty(r, {}, 'NO_SELECTED_CONTRACT')); continue
        rows.append(simulate_option_trade(r, r, underlying_1m, options_1m, p))
    return pd.DataFrame(rows)
