import unittest
import pandas as pd
from pullback_trade import SimParams, simulate_signal


class UnderlyingSimulatorTests(unittest.TestCase):
    def setUp(self):
        self.signal = {
            "symbol": "TEST", "day": "2025-01-02", "direction": "CE",
            "setup": "VWAP_RETEST", "entry_ts": "2025-01-02 09:30:00",
            "entry_price": 100.0, "initial_stop": 99.0,
            "initial_risk": 1.0, "atr5": 1.0,
        }

    def bars(self, rows):
        return pd.DataFrame(rows, columns=["ts", "o", "h", "l", "c"])

    def test_fixed_target_hit(self):
        b = self.bars([
            ("2025-01-02 09:30", 100, 100.5, 99.8, 100.2),
            ("2025-01-02 09:31", 100.2, 101.6, 100.1, 101.4),
        ])
        r = simulate_signal(self.signal, b, SimParams(brokerage_per_order_rs=0))
        x = r[r.policy == "FIXED_1P5R"].iloc[0]
        self.assertEqual(x.exit_reason, "TARGET_1.5R")
        self.assertAlmostEqual(x.gross_r, 1.5)

    def test_stop_wins_same_bar_target_ambiguity(self):
        b = self.bars([
            ("2025-01-02 09:30", 100, 101.7, 98.9, 100.2),
        ])
        r = simulate_signal(self.signal, b, SimParams(brokerage_per_order_rs=0))
        x = r[r.policy == "FIXED_1P5R"].iloc[0]
        self.assertEqual(x.exit_reason, "SL")

    def test_short_direction(self):
        s = dict(self.signal, direction="PE", initial_stop=101.0)
        b = self.bars([
            ("2025-01-02 09:30", 100, 100.2, 99.8, 100.0),
            ("2025-01-02 09:31", 100, 100.1, 98.4, 98.6),
        ])
        r = simulate_signal(s, b, SimParams(brokerage_per_order_rs=0))
        x = r[r.policy == "FIXED_1P5R"].iloc[0]
        self.assertEqual(x.exit_reason, "TARGET_1.5R")
        self.assertAlmostEqual(x.gross_r, 1.5)

    def test_entry_bar_must_exist(self):
        b = self.bars([("2025-01-02 09:31", 100, 100.2, 99.8, 100)])
        r = simulate_signal(self.signal, b)
        self.assertTrue((r.exit_reason == "NO_ENTRY_BAR").all())


    def test_brokerage_not_multiplied_by_share_quantity(self):
        b = self.bars([
            ("2025-01-02 09:30", 100, 100.2, 99.8, 100),
            ("2025-01-02 09:31", 100, 100.2, 99.8, 100),
        ])
        p1 = SimParams(brokerage_per_order_rs=20, risk_budget_rs=None, quantity_shares=1)
        p10 = SimParams(brokerage_per_order_rs=20, risk_budget_rs=None, quantity_shares=10)
        a = simulate_signal(self.signal, b, p1)
        z = simulate_signal(self.signal, b, p10)
        self.assertAlmostEqual(a.iloc[0].estimated_costs_rs, 40.0)
        self.assertAlmostEqual(z.iloc[0].estimated_costs_rs, 40.0)

    def test_risk_budget_sizes_position_and_costs_are_separated(self):
        b = self.bars([
            ("2025-01-02 09:30", 100, 100.2, 99.8, 100),
            ("2025-01-02 09:31", 100, 100.2, 99.8, 100),
        ])
        p = SimParams(risk_budget_rs=1000, slippage_bps_per_side=5,
                      brokerage_per_order_rs=20, other_cost_bps_per_side=1)
        r = simulate_signal(self.signal, b, p)
        x = r[r.policy == "FIXED_1P5R"].iloc[0]
        self.assertEqual(x.quantity_shares, 1000)
        self.assertAlmostEqual(x.risk_rs, 1000.0)
        self.assertAlmostEqual(x.brokerage_rs, 40.0)
        self.assertGreater(x.slippage_cost_rs, 0)
        self.assertGreater(x.total_costs_including_slippage_rs, x.brokerage_rs)
        self.assertAlmostEqual(x.net_pnl_rs, x.gross_pnl_rs - x.total_costs_including_slippage_rs)

    def test_risk_budget_floors_quantity(self):
        b = self.bars([
            ("2025-01-02 09:30", 100, 100.2, 99.8, 100),
            ("2025-01-02 09:31", 100, 100.2, 99.8, 100),
        ])
        p = SimParams(risk_budget_rs=1050, max_shares=500)
        r = simulate_signal(self.signal, b, p)
        self.assertTrue((r.quantity_shares == 500).all())


if __name__ == "__main__":
    unittest.main()
