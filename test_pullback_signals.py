import unittest
from types import SimpleNamespace
import pandas as pd
import numpy as np
from pullback_signals import PullbackParams, _price_integrity_reason, _prepare_frame


class PriceIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.p = PullbackParams()
        self.good = SimpleNamespace(
            o=100.0, h=101.0, l=99.0, c=100.5,
            vwap=99.8, atr5=1.2, ema_fast=100.1, ema_slow=99.9
        )

    def test_normal_bar_passes(self):
        self.assertIsNone(_price_integrity_reason(self.good, self.p))

    def test_bad_intrabar_range_rejected(self):
        row = SimpleNamespace(**vars(self.good))
        row.h = 150.0
        self.assertEqual(_price_integrity_reason(row, self.p), "implausible_5m_range_pct")

    def test_vwap_scale_mismatch_rejected(self):
        row = SimpleNamespace(**vars(self.good))
        row.vwap = 3500.0
        self.assertEqual(_price_integrity_reason(row, self.p), "vwap_price_scale_mismatch")

    def test_atr_scale_mismatch_rejected(self):
        row = SimpleNamespace(**vars(self.good))
        row.atr5 = 15.0
        self.assertEqual(_price_integrity_reason(row, self.p), "implausible_atr_pct")

    def test_pullback_atr_warmup_carries_across_sessions(self):
        rows = []
        start = pd.Timestamp("2025-01-02 09:15")
        for i in range(16):
            ts = start + pd.Timedelta(minutes=5 * i)
            rows.append(dict(
                ts=ts, end=ts + pd.Timedelta(minutes=5), day=ts.date(),
                o=100 + i * 0.1, h=101 + i * 0.1, l=99 + i * 0.1,
                c=100.5 + i * 0.1, vwap=100 + i * 0.1,
                atr5=np.nan, n=5
            ))
        ts = pd.Timestamp("2025-01-03 09:15")
        rows.append(dict(
            ts=ts, end=ts + pd.Timedelta(minutes=5), day=ts.date(),
            o=102, h=103, l=101, c=102.5, vwap=102,
            atr5=np.nan, n=5
        ))
        ctx = SimpleNamespace(f5=pd.DataFrame(rows))
        f = _prepare_frame(ctx, self.p)
        self.assertTrue(np.isfinite(f.iloc[-1].atr5))
        self.assertTrue(np.isnan(f.iloc[0].atr5))


    def test_atr_rolling_ignores_invalid_rows_between_sessions(self):
        rows = []
        start = pd.Timestamp("2025-01-02 09:15")
        for i in range(15):
            ts = start + pd.Timedelta(minutes=5 * i)
            rows.append(dict(
                ts=ts, end=ts + pd.Timedelta(minutes=5), day=ts.date(),
                o=100.0, h=101.0, l=99.0, c=100.0,
                vwap=100.0, atr5=np.nan, n=5
            ))
        # Partial closing candle is an invalid observation.
        ts = pd.Timestamp("2025-01-02 15:25")
        rows.append(dict(
            ts=ts, end=ts + pd.Timedelta(minutes=5), day=ts.date(),
            o=100.0, h=101.0, l=99.0, c=100.0,
            vwap=100.0, atr5=np.nan, n=3
        ))
        # 15 valid TR observations already exist. ATR should be available
        # immediately on the next session's first valid candle.
        ts = pd.Timestamp("2025-01-03 09:15")
        rows.append(dict(
            ts=ts, end=ts + pd.Timedelta(minutes=5), day=ts.date(),
            o=100.0, h=101.0, l=99.0, c=100.0,
            vwap=100.0, atr5=np.nan, n=5
        ))
        ctx = SimpleNamespace(f5=pd.DataFrame(rows))
        f = _prepare_frame(ctx, self.p)
        next_day_first = f[f["day"] == pd.Timestamp("2025-01-03").date()].iloc[0]
        self.assertTrue(np.isfinite(next_day_first.atr5))


if __name__ == "__main__":
    unittest.main()
