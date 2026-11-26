"""Run a small real-data pullback signal smoke test.

Example:
    python pullback_smoke_test.py --data-dir data --symbols RELIANCE SBIN INFY
"""
import argparse
from pathlib import Path

import pandas as pd
import bt_signals as ORB
from pullback_signals import PullbackParams, find_pullback_signals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--symbols", nargs="*", default=["RELIANCE", "SBIN", "INFY"])
    ap.add_argument("--out", default="data/backtest/pullback_smoke")
    args = ap.parse_args()

    data_root = Path(args.data_dir)
    stock_root = data_root / "stock_1m"
    if not stock_root.exists():
        raise SystemExit(
            f"Cannot find {stock_root.resolve()}. Run this inside your project "
            "where the downloaded stock_1m parquet data is stored."
        )

    nifty5, nifty_ref_type = ORB.index_context("NIFTY", args.data_dir)
    print(f"NIFTY reference: {nifty_ref_type}")

    params = PullbackParams()
    all_signals, all_rejects = [], []

    for symbol in args.symbols:
        print(f"Loading {symbol}...")
        ctx = ORB.build_ctx(symbol, args.data_dir)
        if ctx is None:
            print(f"  SKIP: no 1-minute stock data for {symbol}")
            continue

        signals, rejects = find_pullback_signals(ctx, nifty5, params)
        if not signals.empty:
            all_signals.append(signals)
        if not rejects.empty:
            all_rejects.append(rejects)

        print(
            f"  bars5={len(ctx.f5)} signals={len(signals)} "
            f"VWAP={int((signals.setup == 'VWAP_RETEST').sum()) if len(signals) else 0} "
            f"breakout={int((signals.setup == 'BREAKOUT_RETEST').sum()) if len(signals) else 0}"
        )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sig = pd.concat(all_signals, ignore_index=True) if all_signals else pd.DataFrame()
    rej = pd.concat(all_rejects, ignore_index=True) if all_rejects else pd.DataFrame()
    sig.to_csv(out / "pullback_signals.csv", index=False)
    rej.to_csv(out / "pullback_rejects.csv", index=False)
    if len(rej):
        rej.groupby(["setup", "reason"], dropna=False).size().rename("count").reset_index().sort_values(
            "count", ascending=False
        ).to_csv(out / "pullback_reject_summary.csv", index=False)
    if len(sig):
        overlaps = sig[sig["is_cross_variant_overlap"]].copy()
        overlaps.to_csv(out / "pullback_overlaps.csv", index=False)

    print("\\nSaved:")
    print(out / "pullback_signals.csv")
    print(out / "pullback_rejects.csv")
    print(out / "pullback_reject_summary.csv")
    print(out / "pullback_overlaps.csv")
    if len(sig):
        print("\\nSignals by setup/direction:")
        print(sig.groupby(["setup", "direction"]).size().to_string())
        print("\\nSample signals:")
        print(sig[["symbol", "day", "direction", "setup", "signal_bar_end",
                   "entry_ts", "entry_price", "initial_stop"]].head(20).to_string(index=False))
    else:
        print("\\nNo qualifying signals in this sample.")
    if len(rej):
        print("\\nMost common rejection reasons:")
        print(rej.reason.value_counts().head(15).to_string())


if __name__ == "__main__":
    main()
