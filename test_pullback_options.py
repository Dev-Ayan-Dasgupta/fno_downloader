import unittest
import pandas as pd
from pullback_options import OptionSimParams, groww_option_charges, simulate_option_trade, run_options_backtest, make_signal_key

class OptionsTests(unittest.TestCase):
    def test_groww_charges_use_premium_turnover(self):
        p=OptionSimParams()
        x=groww_option_charges(36000, 42000, p)
        self.assertAlmostEqual(x['brokerage_rs'],40)
        self.assertAlmostEqual(x['stt_rs'],63)
        self.assertAlmostEqual(x['stamp_duty_rs'],1.08)
        self.assertAlmostEqual(x['exchange_charges_rs'],27.3234)
        self.assertGreater(x['groww_charges_rs'], 130)
        self.assertLess(x['groww_charges_rs'], 150)

    def test_underlying_target_exits_option_on_next_minute_open(self):
        sig=dict(symbol='ABC',day='2026-01-05',direction='CE',setup='BREAKOUT_RETEST',
                 entry_ts='2026-01-05 09:20:00',entry_price=100,initial_stop=98,
                 initial_risk=2,atr5=1)
        con=dict(option_symbol='ABC26JAN100CE',option_type='CE',strike=100,
                 expiry='2026-01-29',lot_size=50,lots=2)
        und=pd.DataFrame([
            ['ABC','2026-01-05 09:20:00',100,101,99,100],
            ['ABC','2026-01-05 09:21:00',100,105,99,104],
            ['ABC','2026-01-05 09:22:00',104,105,103,104]],columns=['symbol','ts','o','h','l','c'])
        opt=pd.DataFrame([
            ['ABC26JAN100CE','2026-01-05 09:20:00',60,61,59,60],
            ['ABC26JAN100CE','2026-01-05 09:21:00',60,62,59,61],
            ['ABC26JAN100CE','2026-01-05 09:22:00',62,65,61,64]],columns=['option_symbol','ts','o','h','l','c'])
        r=simulate_option_trade(sig,con,und,opt,OptionSimParams(target_r=2,slippage_bps_per_side=0))
        self.assertEqual(r['exit_reason'],'UNDERLYING_TARGET_2R')
        self.assertEqual(r['exit_option_ts'],pd.Timestamp('2026-01-05 09:22:00'))
        self.assertEqual(r['entry_option_price'],60)
        self.assertEqual(r['exit_option_price'],62)
        self.assertEqual(r['quantity_units'],100)
        self.assertEqual(r['premium_outlay_rs'],6000)

    def test_underlying_stop_has_priority_if_both_hit(self):
        sig=dict(symbol='ABC',day='2026-01-05',direction='CE',setup='X',entry_ts='2026-01-05 09:20:00',
                 entry_price=100,initial_stop=98,initial_risk=2,atr5=1)
        con=dict(option_symbol='X',option_type='CE',strike=100,expiry='2026-01-29',lot_size=1,lots=1)
        und=pd.DataFrame([['ABC','2026-01-05 09:20:00',100,105,97,100],
                          ['ABC','2026-01-05 09:21:00',100,101,99,100]],columns=['symbol','ts','o','h','l','c'])
        opt=pd.DataFrame([['X','2026-01-05 09:20:00',10,11,9,10],
                          ['X','2026-01-05 09:21:00',10,10,9,9]],columns=['option_symbol','ts','o','h','l','c'])
        r=simulate_option_trade(sig,con,und,opt,OptionSimParams(slippage_bps_per_side=0))
        self.assertEqual(r['exit_reason'],'UNDERLYING_STOP')


    def test_batch_runner_preserves_signal_columns_with_template_metadata(self):
        sig=dict(symbol='ABC',day='2026-01-05',direction='CE',setup='BREAKOUT_RETEST',
                 entry_ts='2026-01-05 09:20:00',entry_price=100,initial_stop=98,
                 initial_risk=2,atr5=1)
        sigdf=pd.DataFrame([sig]); sigdf['signal_key']=sigdf.apply(make_signal_key,axis=1)
        con=pd.DataFrame([dict(signal_key=sigdf.iloc[0].signal_key,symbol='ABC',day='2026-01-05',
            direction='CE',setup='BREAKOUT_RETEST',entry_ts='2026-01-05 09:20:00',entry_price=100,
            option_symbol='X',option_type='CE',strike=100,expiry='2026-01-29',lot_size=1,lots=1)])
        und=pd.DataFrame([['ABC','2026-01-05 09:20:00',100,101,99,100],
                          ['ABC','2026-01-05 09:21:00',100,101,99,100]],columns=['symbol','ts','o','h','l','c'])
        opt=pd.DataFrame([['X','2026-01-05 09:20:00',10,11,9,10],
                          ['X','2026-01-05 09:21:00',10,11,9,10]],columns=['option_symbol','ts','o','h','l','c'])
        out=run_options_backtest(sigdf,con,und,opt,OptionSimParams(slippage_bps_per_side=0))
        self.assertEqual(len(out),1)
        self.assertEqual(out.iloc[0].symbol,'ABC')
        self.assertEqual(out.iloc[0].option_symbol,'X')

if __name__=='__main__': unittest.main()
