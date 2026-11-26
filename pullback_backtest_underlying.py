"""Run underlying-only pullback trade simulation.

Example:
  python pullback_backtest_underlying.py --data-dir data \
      --signals data/backtest/pullback_smoke/pullback_signals.csv \
      --symbols RELIANCE SBIN INFY
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import bt_signals as ORB
from pullback_trade import SimParams, run_simulation, summarize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--signals", required=True)
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--out", default="data/backtest/pullback_underlying")
    ap.add_argument("--slippage-bps", type=float, default=5.0)
    ap.add_argument("--brokerage-per-order", type=float, default=20.0)
    ap.add_argument("--other-cost-bps", type=float, default=0.0)
    ap.add_argument("--quantity", type=int, default=1,
                    help="Fixed shares only when --fixed-quantity is set")
    ap.add_argument("--risk-budget", type=float, default=1000.0,
                    help="Illustrative rupee risk budget per trade (default: 1000)")
    ap.add_argument("--fixed-quantity", action="store_true",
                    help="Disable risk-budget sizing and use --quantity shares")
    ap.add_argument("--max-shares", type=int, default=None)
    args = ap.parse_args()

    sig = pd.read_csv(args.signals)
    if sig.empty:
        raise SystemExit("Signal file is empty.")
    sig["entry_ts"] = pd.to_datetime(sig["entry_ts"])
    sig["day"] = pd.to_datetime(sig["day"]).dt.date
    sig["signal_bar_start"] = pd.to_datetime(sig["signal_bar_start"])
    sig["signal_bar_end"] = pd.to_datetime(sig["signal_bar_end"])
    symbols = args.symbols or sorted(sig["symbol"].unique())

    p = SimParams(
        slippage_bps_per_side=args.slippage_bps,
        brokerage_per_order_rs=args.brokerage_per_order,
        other_cost_bps_per_side=args.other_cost_bps,
        risk_budget_rs=None if args.fixed_quantity else args.risk_budget,
        quantity_shares=args.quantity,
        max_shares=args.max_shares,
    )
    contexts = {}
    for symbol in symbols:
        print(f"Loading {symbol}...")
        ctx = ORB.build_ctx(symbol, args.data_dir)
        if ctx is not None:
            contexts[symbol] = ctx

    sig = sig[sig["symbol"].isin(contexts)]
    trades = run_simulation(sig, contexts, p)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    trades.to_csv(out / "underlying_trades.csv", index=False)
    summary = summarize(trades)
    summary.to_csv(out / "underlying_summary.csv", index=False)

    if not trades.empty:
        daily = trades.groupby(["setup", "policy", "day"], as_index=False).agg(
            trades=("net_r", "size"), gross_r=("gross_r", "sum"),
            net_r=("net_r", "sum"), avg_gross_r=("gross_r", "mean"),
            avg_net_r=("net_r", "mean"), net_pnl_rs=("net_pnl_rs", "sum"),
            gross_pnl_rs=("gross_pnl_rs", "sum"),
            costs_rs=("total_costs_including_slippage_rs", "sum"),
        )
        daily.to_csv(out / "underlying_daily.csv", index=False)
        # Chronological splits inherited from the prior ORB research. These are
        # descriptive only; prior test period has already been examined.
        d = pd.to_datetime(trades["day"])
        trades["period"] = "outside_defined_windows"
        trades.loc[d.between("2024-12-09", "2025-06-27"), "period"] = "development"
        trades.loc[d.between("2025-07-02", "2026-03-20"), "period"] = "validation"
        trades.loc[d.between("2026-04-01", "2026-11-20"), "period"] = "test_previously_examined"
        period = trades.groupby(["period", "setup", "policy"], as_index=False).agg(
            trades=("net_r", "size"), avg_gross_r=("gross_r", "mean"),
            total_gross_r=("gross_r", "sum"), avg_net_r=("net_r", "mean"),
            total_net_r=("net_r", "sum"), gross_win_rate=("gross_r", lambda x: (x > 0).mean()),
            net_win_rate=("net_r", lambda x: (x > 0).mean()),
            gross_pnl_rs=("gross_pnl_rs", "sum"), net_pnl_rs=("net_pnl_rs", "sum"),
            total_costs_rs=("total_costs_including_slippage_rs", "sum"),
        )
        period.to_csv(out / "underlying_period_summary.csv", index=False)

    with open(out / "sim_params.json", "w") as f:
        json.dump(p.__dict__, f, indent=2)
    print("\\nSUMMARY")
    print(summary.to_string(index=False) if not summary.empty else "No trades simulated.")
    print(f"\\nSaved results to {out}")


if __name__ == "__main__":
    main()
