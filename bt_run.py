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
        ctx = S.build_ctx(sym, atr5_mode=p0.atr5_mode)
        if ctx is None:
            continue
        sigs, rej = (S.find_pullback_signals if p0.hyp == "pullback" else S.find_signals)(ctx, nifty5, p0)
        sigs = [s for s in sigs if d0 <= s["day"] <= d1]
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
                sim = T.simulate(lv["entry"], lv["stop"], lv["tg"], ts, *arr, p.variant, eod_ts, intrabar_mode=p.intrabar_mode, scheme=p.exit_scheme)
                tr = T.finalize(sg, sel, lv, sim, p, lots_map[sym])
                uo = T.underlying_outcome(sg, lv, ctx, eod_t, sel["entry_ts"])
                tr["u_outcome"], tr["u_reach_t2_any"] = uo["outcome"], uo["reach_t2_any"]
                tr["u_reach_t2_before_stop"], tr["u_stop_hit"] = uo["reach_t2_before_stop"], uo["stop_hit"]
                tr["stop_mult"], tr["variant"] = p.stop_mult, p.variant
                trades[(p.stop_mult, p.variant)].append(tr)
                made += 1
        log(f"  ({n}/{len(symbols)}) {sym}: {len(sigs)} signals, {made} trades ({_time.time() - t0:.0f}s)")
    return {k: pd.DataFrame(v) for k, v in trades.items()}, pd.DataFrame(rejects), info


def log_run(meta):
    with open(os.path.join(_root(), "runs.jsonl"), "a") as f:
        f.write(json.dumps(meta, default=str) + "\n")


CONFIG_KEYS = ("stop_mult", "variant", "hyp", "slip", "lots", "execution_mode", "intrabar_mode", "atr5_mode", "exit_scheme")
CONFIG_DEFAULTS = dict(stop_mult=1.5, variant="A", hyp="breakout", slip=0.02, lots=4,
                       execution_mode="next_open", intrabar_mode="conservative", atr5_mode="carry", exit_scheme="t4")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="run", choices=["run", "freeze"])
    ap.add_argument("--period", default="dev", choices=list(PERIODS))
    ap.add_argument("--symbols"); ap.add_argument("--top", type=int, default=40); ap.add_argument("--all", action="store_true")
    ap.add_argument("--m", type=float, dest="stop_mult"); ap.add_argument("--variant", choices=["A", "B"])
    ap.add_argument("--slip", type=float); ap.add_argument("--lots", type=int)
    ap.add_argument("--hyp", choices=["breakout", "pullback"])
    ap.add_argument("--execution-mode", dest="execution_mode", choices=["next_open", "legacy_close"])
    ap.add_argument("--intrabar-mode", dest="intrabar_mode", choices=["conservative", "optimistic"])
    ap.add_argument("--atr5-mode", dest="atr5_mode", choices=["carry", "session"])
    ap.add_argument("--exit-scheme", dest="exit_scheme", choices=["t4", "t1t2"])
    ap.add_argument("--grid", action="store_true"); ap.add_argument("--fetch-missing", action="store_true")
    ap.add_argument("--confirm-final", action="store_true")
    a = ap.parse_args()
    cli = {k: getattr(a, k) for k in CONFIG_KEYS if getattr(a, k) is not None}
    cfg = {**CONFIG_DEFAULTS, **cli}
    frozen_path = os.path.join(_root(), "frozen.json")
    if a.cmd == "freeze":
        json.dump({**cfg, "frozen_at": datetime.now().isoformat()}, open(frozen_path, "w"))
        print("Frozen:", open(frozen_path).read())
        return
    if a.period in ("val", "test"):
        if not os.path.exists(frozen_path):
            raise SystemExit("Validation/test need frozen parameters: run  python bt_run.py freeze --m ... --variant ... --hyp ...")
        fz = json.load(open(frozen_path))
        conflicts = {k: (v, fz.get(k)) for k, v in cli.items() if k in fz and fz[k] != v}
        if conflicts:
            raise SystemExit(f"These options conflict with the frozen configuration (requested, frozen): {conflicts}")
        if a.grid:
            raise SystemExit("No grid search outside the Development period")
        if a.period == "test" and not a.confirm_final:
            raise SystemExit("The final test runs once, only after validation passes: add --confirm-final")
        cfg = {**CONFIG_DEFAULTS, **{k: fz[k] for k in CONFIG_KEYS if k in fz}}
    if cfg["lots"] % 4:
        raise SystemExit("--lots must be a multiple of 4 (50/25/25 booking needs whole lots)")
    grid_m = (1.0, 1.5, 2.0) if cfg["hyp"] == "pullback" else (1.0, 1.5, 2.0, 2.5)
    variants = ("A",) if cfg["exit_scheme"] == "t1t2" else ("A", "B")       # the trailing variants do not apply to t1t2
    combos = [(m, v) for m in grid_m for v in variants] if a.grid else [(cfg["stop_mult"], cfg["variant"])]
    mk = lambda m, v: S.Params(stop_mult=m, variant=v, slippage=cfg["slip"], lots=cfg["lots"], hyp=cfg["hyp"],
                               execution_mode=cfg["execution_mode"], intrabar_mode=cfg["intrabar_mode"], atr5_mode=cfg["atr5_mode"],
                               exit_scheme=cfg["exit_scheme"])
    params_list = [mk(m, v) for m, v in combos]
    symbols = pick_symbols(a)
    client = None
    if a.fetch_missing:
        from client import Client
        client = Client(rps=4.0)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"{cfg['hyp']}_{cfg['exit_scheme']}_{cfg['execution_mode']}_{cfg['intrabar_mode']}"
    out = os.path.join(_root(), f"{run_id}_{tag}")
    os.makedirs(out, exist_ok=True)
    print(f"Run {run_id} | {tag} | atr5={cfg['atr5_mode']} slip={cfg['slip']} lots={cfg['lots']} | period {a.period} {PERIODS[a.period]} | "
          f"{len(symbols)} symbols | combos {combos}")
    trades, rej, info = run_backtest(symbols, PERIODS[a.period], params_list, load_lots(), client)
    rej.to_csv(os.path.join(out, "rejects.csv"), index=False)
    print(f"\nSignals {info['signals']} | no contract/data {info['no_contract']} | rejects logged {len(rej)}")
    if len(rej):
        print(rej.groupby(["stage", rej["reason"].str.split(":").str[0].str.split(" ").str[0].str.split("(").str[0]]).size().sort_values(ascending=False).head(12).to_string())
    rows = []
    for (m, v), tr in trades.items():
        tr.to_csv(os.path.join(out, f"trades_m{m}_{v}.csv"), index=False)
        s = ST.summarize(tr)
        rows.append({"m": m, "variant": v, **s})
        log_run({"run_id": run_id, "period": a.period, "params": asdict(mk(m, v)), "symbols": len(symbols), "summary": s})
    res = pd.DataFrame(rows)
    show = [c for c in ["m", "variant", "n", "win_rate", "avg_r_gross", "avg_r_net", "ci_lo", "ci_hi", "profit_factor",
                        "trade_sequence_max_dd_r", "daily_aggregate_max_dd_r", "sl_then_t2_any_rate", "hit_t2", "cost_in_r",
                        "top_stock_positive_r_share", "months_positive"] if c in res.columns]
    print("\n" + res[show].to_string(index=False))
    res.to_csv(os.path.join(out, "summary.csv"), index=False)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
