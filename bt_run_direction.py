"""
Backtest runner (spec v0.2). Examples:
  python bt_run.py --period dev --top 40 --m 1.5 --variant A     # one run on the Development period
  python bt_run.py --period dev --top 40 --grid                   # m in {1.0,1.5,2.0,2.5} x variants {A,B}
  python bt_run.py freeze --m 1.5 --variant A                     # freeze parameters after Development
  python bt_run.py --period val --top 40                          # Validation: only with frozen parameters
Every run is appended to data/backtest/runs.jsonl (the variant count matters for multiple testing).
"""
import argparse
import json
import os
import time as _time
from dataclasses import asdict
from datetime import date, datetime

import numpy as np
import pandas as pd

import bt_config as B
import bt_data as D
import bt_signals as S
import bt_stats as ST
import bt_trade as T

PERIODS = {"dev": ("2024-11-24", "2025-06-30"), "val": ("2025-07-01", "2026-03-31"), "test": ("2026-04-01", "2026-11-20")}


def _root():
    r = os.path.join(B.data_dir(), "backtest")
    os.makedirs(r, exist_ok=True)
    return r


def load_lots():
    p = os.path.join(B.data_dir(), "meta", "lot_sizes.json")
    if not os.path.exists(p):
        raise SystemExit("Missing data/meta/lot_sizes.json - run:  python bt_lots.py")
    return json.load(open(p))


def pick_symbols(args):
    if args.symbols:
        return [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    rank = json.load(open(os.path.join(B.data_dir(), "meta", "turnover_rank.json")))
    order = [s for s, _ in sorted(rank.items(), key=lambda kv: -kv[1]) if "NSETEST" not in s and s not in
             ("NIFTY", "BANKNIFTY", "NIFTYFPI")]
    return order if args.all else order[: args.top]


def run_backtest(symbols, period, params_list, lots_map, client=None, log=print):
    """Returns ({combo_key: trades DataFrame}, rejects DataFrame, info dict)."""
    d0, d1 = (x if isinstance(x, date) else date.fromisoformat(x) for x in period)
    nifty5, kind = S.index_context("NIFTY")
    log(f"Nifty reference: {kind}")
    p0 = params_list[0]
    keys = [(p.stop_mult, p.variant) for p in params_list]
    trades = {k: [] for k in keys}
    rejects, info = [], {"signals": 0, "no_contract": 0}
    eod_t = S._tm(p0.eod)
    for n, sym in enumerate(symbols, 1):
        t0 = _time.time()
        if sym not in lots_map:
            log(f"  ({n}/{len(symbols)}) {sym}: no lot size, skipped")
            continue
        ctx = S.build_ctx(sym)
        if ctx is None:
            continue
        sigs, rej = S.find_signals(ctx, nifty5, p0)
        sigs = [s for s in sigs if d0 <= s["day"] <= d1]
        # Directional validation is a post-signal filter: all underlying signal rules remain unchanged.
        # Keep rejects untouched so CE/PE runs retain the same diagnostic rejection population.
        if getattr(run_backtest, "direction_filter", "BOTH") != "BOTH":
            sigs = [s for s in sigs if s["direction"] == run_backtest.direction_filter]
        rejects += [r for r in rej if d0 <= r["day"] <= d1]
        info["signals"] += len(sigs)
        if not sigs:
            log(f"  ({n}/{len(symbols)}) {sym}: 0 signals")
            continue
        store = D.OptionStore(sym, client=client)
        ctx.liq = store.liquidity()
        made = 0
        for sg in sigs:
            sel, why = T.select_contract(sg, ctx, store, p0)
            if sel is None:
                rejects.append(dict(symbol=sym, day=sg["day"], direction=sg["direction"], T=sg["T"], stage="contract", reason=why))
                info["no_contract"] += 1
                continue
            rows = store.get(sel["contract"], sg["day"])
            if p0.execution_mode == "next_open":
                rows = rows[rows["ts"] >= sel["entry_ts"]]
            else:
                rows = rows[rows["ts"] > sg["T"]]
            ts = rows["ts"].to_numpy()
            arr = [rows[c].to_numpy() for c in ("o", "h", "l", "c")]
            eod_ts = np.datetime64(pd.Timestamp.combine(sg["day"], eod_t))
            for p in params_list:
                lv, why = T.make_levels(sg, sel, ctx, p, p.stop_mult)
                if lv is None:
                    rejects.append(dict(symbol=sym, day=sg["day"], direction=sg["direction"], T=sg["T"], stage="levels",
                                        reason=f"{why} (m={p.stop_mult})"))
                    continue
                sim = T.simulate(lv["entry"], lv["stop"], lv["tg"], ts, *arr,
                                 p.variant, eod_ts, p.intrabar_mode)
                tr = T.finalize(sg, sel, lv, sim, p, lots_map[sym])
                u = T.underlying_outcome(sg, lv, ctx, eod_t)
                tr["u_outcome"] = u["outcome"]
                tr["u_reach_t2"] = u["reach_t2_any"]
                tr["u_reach_t2_before_stop"] = u["reach_t2_before_stop"]
                tr["u_stop_hit"] = u["stop_hit"]
                tr["stop_mult"], tr["variant"] = p.stop_mult, p.variant
                trades[(p.stop_mult, p.variant)].append(tr)
                made += 1
        log(f"  ({n}/{len(symbols)}) {sym}: {len(sigs)} signals, {made} trades ({_time.time() - t0:.0f}s)")
    return {k: pd.DataFrame(v) for k, v in trades.items()}, pd.DataFrame(rejects), info


def log_run(meta):
    with open(os.path.join(_root(), "runs.jsonl"), "a") as f:
        f.write(json.dumps(meta, default=str) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="run", choices=["run", "freeze"])
    ap.add_argument("--period", default="dev", choices=list(PERIODS))
    ap.add_argument("--direction", default="BOTH", choices=["BOTH", "CE", "PE"],
                    help="Filter detected signals by option direction; does not alter signal rules")
    ap.add_argument("--symbols"); ap.add_argument("--top", type=int, default=40); ap.add_argument("--all", action="store_true")
    ap.add_argument("--m", type=float); ap.add_argument("--variant", choices=["A", "B"])
    ap.add_argument("--slip", type=float); ap.add_argument("--lots", type=int)
    ap.add_argument("--execution-mode", choices=["next_open", "legacy_close"])
    ap.add_argument("--intrabar-mode", choices=["conservative", "optimistic"])
    ap.add_argument("--grid", action="store_true"); ap.add_argument("--fetch-missing", action="store_true")
    ap.add_argument("--confirm-final", action="store_true")
    a = ap.parse_args()
    frozen_path = os.path.join(_root(), "frozen.json")

    if a.cmd == "freeze":
        defaults = S.Params()
        frozen_params = asdict(S.Params(
            stop_mult=a.m if a.m is not None else defaults.stop_mult,
            variant=a.variant if a.variant is not None else defaults.variant,
            slippage=a.slip if a.slip is not None else defaults.slippage,
            lots=a.lots if a.lots is not None else defaults.lots,
            execution_mode=a.execution_mode if a.execution_mode is not None else defaults.execution_mode,
            intrabar_mode=a.intrabar_mode if a.intrabar_mode is not None else defaults.intrabar_mode,
        ))
        if frozen_params["lots"] % 4:
            raise SystemExit("lots must be a multiple of 4 for the 50/25/25 exit schedule")
        payload = {"schema_version": 2, "params": frozen_params,
                   "frozen_at": datetime.now().isoformat()}
        with open(frozen_path, "w") as f:
            json.dump(payload, f, indent=2)
        print("Frozen:", json.dumps(payload, indent=2))
        return

    defaults = S.Params()
    if a.period in ("val", "test"):
        if not os.path.exists(frozen_path):
            raise SystemExit("Validation/test need frozen parameters: run bt_run.py freeze after Development.")
        with open(frozen_path) as f:
            fz = json.load(f)
        frozen = fz.get("params", fz)
        if a.grid:
            raise SystemExit("No grid search outside the Development period")
        if a.period == "test" and not a.confirm_final:
            raise SystemExit("The final test runs once, only after validation passes: add --confirm-final")
        for arg_name, key in (("m", "stop_mult"), ("variant", "variant"), ("slip", "slippage"),
                              ("lots", "lots"), ("execution_mode", "execution_mode"),
                              ("intrabar_mode", "intrabar_mode")):
            supplied = getattr(a, arg_name)
            if supplied is not None and key in frozen and supplied != frozen[key]:
                raise SystemExit(f"--{arg_name.replace('_', '-')} conflicts with frozen value {frozen[key]!r}")
        base_params = S.Params(**{k: v for k, v in frozen.items() if k in S.Params.__dataclass_fields__})
    else:
        base_params = S.Params(
            stop_mult=a.m if a.m is not None else defaults.stop_mult,
            variant=a.variant if a.variant is not None else defaults.variant,
            slippage=a.slip if a.slip is not None else defaults.slippage,
            lots=a.lots if a.lots is not None else defaults.lots,
            execution_mode=a.execution_mode if a.execution_mode is not None else defaults.execution_mode,
            intrabar_mode=a.intrabar_mode if a.intrabar_mode is not None else defaults.intrabar_mode,
        )
    if base_params.lots % 4:
        raise SystemExit("--lots must be a multiple of 4 (50/25/25 booking needs whole lots)")
    combos = [(m, v) for m in (1.0, 1.5, 2.0, 2.5) for v in ("A", "B")] if a.grid else [(base_params.stop_mult, base_params.variant)]
    params_list = [S.Params(**{**asdict(base_params), "stop_mult": m, "variant": v}) for m, v in combos]
    symbols = pick_symbols(a)
    client = None
    if a.fetch_missing:
        from client import Client
        client = Client(rps=4.0)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(_root(), run_id)
    os.makedirs(out, exist_ok=True)
    print(f"Run {run_id} | period {a.period} {PERIODS[a.period]} | direction {a.direction} | {len(symbols)} symbols | combos {combos}")
    print(f"Execution: {base_params.execution_mode} | intrabar: {base_params.intrabar_mode}")
    run_backtest.direction_filter = a.direction
    trades, rej, info = run_backtest(symbols, PERIODS[a.period], params_list, load_lots(), client)
    rej.to_csv(os.path.join(out, "rejects.csv"), index=False)
    print(f"\nSignals {info['signals']} | no contract/data {info['no_contract']} | rejects logged {len(rej)}")
    if len(rej):
        print(rej.groupby(["stage", rej["reason"].str.split(":").str[0].str.split(" ").str[0]]).size().sort_values(ascending=False).head(12).to_string())
    rows = []
    for (m, v), tr in trades.items():
        tr.to_csv(os.path.join(out, f"trades_m{m}_{v}.csv"), index=False)
        summary = ST.summarize(tr)
        rows.append({"m": m, "variant": v, **summary})
        log_run({"run_id": run_id, "period": a.period, "params": asdict(S.Params(**{**asdict(base_params), "stop_mult": m, "variant": v})),
                 "symbols": len(symbols), "direction": a.direction, "summary": summary})
    res = pd.DataFrame(rows)
    show = [c for c in ["m", "variant", "n", "win_rate", "avg_r_gross", "avg_r_net", "ci_lo", "ci_hi",
                        "profit_factor", "trade_sequence_max_dd_r", "daily_aggregate_max_dd_r",
                        "early_stop_rate", "hit_t2", "cost_in_r", "top_stock_positive_r_share",
                        "months_positive"] if c in res.columns]
    print("\n" + res[show].to_string(index=False))
    res.to_csv(os.path.join(out, "summary.csv"), index=False)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
