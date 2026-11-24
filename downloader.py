#!/usr/bin/env python3
"""
Groww historical data downloader (stocks + near-ATM option contracts, 1-minute, with OI) with a quality report.

  python downloader.py estimate                       # measure on 2 stocks, project time/size for all (do this first)
  python downloader.py run --months 24                # everything: stocks then options, all F&O stocks
  python downloader.py run --phase stock --symbols RELIANCE,TCS
  python downloader.py quality                        # writes data/reports/QUALITY.md + CSVs
  python downloader.py status

Safe to stop (Ctrl+C) and rerun: finished work is skipped.
"""
import argparse
import os
import shutil
import time
from datetime import date, timedelta

import config as C
from client import Client
from fetch import load_universe, run_option_phase, run_stock_phase
from quality import build_report
from store import Store


def resolve_symbols(client, store, arg):
    if arg.upper() == "ALL":
        return load_universe(client, store)
    return [s.strip().upper() for s in arg.split(",") if s.strip()]


def dates(a):
    end = date.fromisoformat(a.end) if getattr(a, "end", None) else date.today()
    start = date.fromisoformat(a.start) if getattr(a, "start", None) else end - timedelta(days=int(a.months * 30.4))
    return start, end


def do_run(a, client=None, store=None):
    client = client or Client(rps=a.rps)
    store = store or Store(C.DATA_DIR)
    start, end = dates(a)
    syms = resolve_symbols(client, store, a.symbols)
    print(f"{len(syms)} symbols | {start} -> {end} | phase={a.phase} | rps={client.rps} workers={a.workers}")
    t0 = time.time()
    if a.phase in ("stock", "all"):
        run_stock_phase(client, store, syms, start, end, a.workers)
    if a.phase in ("option", "all"):
        run_option_phase(client, store, syms, start, end, a.band, a.roll_days, a.workers)
    print(f"\nFinished in {(time.time() - t0) / 60:.1f} min | requests {client.n_requests} | retries {client.n_retries} "
          f"| disk {store.disk_mb():.0f} MB")
    for k, v in store.failures():
        print("FAILED:", k, "|", v)
    print("Run again to retry anything that failed. Then: python downloader.py quality")


def do_estimate(a, client=None, make_store=None):
    root = os.path.join(C.DATA_DIR, "_estimate")
    shutil.rmtree(root, ignore_errors=True)
    store = make_store(root) if make_store else Store(root)
    client = client or Client(rps=a.rps)
    syms = [s.strip().upper() for s in a.symbols.split(",") if s.strip()]
    start, end = dates(a)
    t0 = time.time()
    run_stock_phase(client, store, syms, start, end, a.workers, log=lambda *x: None)
    run_option_phase(client, store, syms, start, end, a.band, a.roll_days, a.workers, log=lambda *x: None)
    secs, req, mb = time.time() - t0, client.n_requests, store.disk_mb()
    n_all = len(load_universe(client, store)) if not a.no_universe else 220
    scale = (n_all / len(syms)) * (a.target_months / a.months)
    obs_rps = req / secs if secs else 0
    print(f"\nSample: {len(syms)} stocks x {a.months} months -> {req} requests in {secs:.0f}s ({obs_rps:.1f}/s), {mb:.0f} MB | retries {client.n_retries} | final rps {client.rps}")
    print(f"Projection for {n_all} stocks x {a.target_months} months:")
    print(f"  requests ~ {req * scale:,.0f}  |  time ~ {req * scale / max(obs_rps, 0.1) / 3600:.1f} h at the observed rate"
          f"  |  disk ~ {mb * scale / 1000:.1f} GB")
    print("Caveat: liquid sample stocks overstate the size of thin ones; treat this as an upper-ish bound.")
    shutil.rmtree(root, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "estimate"):
        p = sub.add_parser(name)
        p.add_argument("--symbols", default="ALL" if name == "run" else "RELIANCE,FEDERALBNK")
        p.add_argument("--start")
        p.add_argument("--end")
        p.add_argument("--months", type=float, default=24 if name == "run" else 2)
        p.add_argument("--band", type=int, default=1, help="strikes either side of ATM")
        p.add_argument("--roll-days", type=int, default=4, help="switch to next expiry when fewer calendar days remain")
        p.add_argument("--rps", type=float, default=4.0, help="requests/sec (Groww live limits: 10/s, 300/min)")
        p.add_argument("--workers", type=int, default=4)
        if name == "run":
            p.add_argument("--phase", choices=["stock", "option", "all"], default="all")
        else:
            p.add_argument("--target-months", type=float, default=24)
            p.add_argument("--no-universe", action="store_true")
    sub.add_parser("quality")
    sub.add_parser("status")
    a = ap.parse_args()
    if a.cmd == "run":
        do_run(a)
    elif a.cmd == "estimate":
        do_estimate(a)
    else:
        store = Store(C.DATA_DIR)
        if a.cmd == "status":
            for k, s, n, r in store.stats():
                print(f"{k:6} {s:8} tasks={n:7} rows={r or 0:,}")
            print(f"disk {store.disk_mb():.0f} MB")
            for k, v in store.failures():
                print("FAILED:", k, "|", v)
        else:
            syms = sorted(d for d in os.listdir(store.path("stock_1m"))) if os.path.exists(store.path("stock_1m")) else []
            print(build_report(store, syms))


if __name__ == "__main__":
    main()
