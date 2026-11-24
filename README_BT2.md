# Backtest stage 2: signals + trade simulator (spec v0.2)

Unzip INTO your `fno_downloader` folder and overwrite (bt_data.py, bt_config.py and selftest_bt.py are updated:
split/bonus exclusion is now 5 sessions before to 20 after a >25% gap, because ATR(14) and the 20-session
volume baseline stay contaminated that long).

    python selftest_bt.py            # 28 checks (stage 1)
    python selftest_stage2.py        # 21 checks (stage 2)
    python bt_lots.py                # once: saves current lot sizes (needs your Groww login)
    python bt_run.py --period dev --top 40 --m 1.5 --variant A     # one Development-period run, 40 most liquid stocks
    python bt_run.py --period dev --top 40 --grid                  # m in {1.0,1.5,2.0,2.5} x trailing variants {A,B}

Rules implemented as in the spec: opening-range breakout + VWAP + RVOL>=1.5 + Nifty (time-weighted reference because the
index has no volume) + extension cap, entry at the first option candle after the signal bar (no candle = no trade), delta
0.35-0.65 closest to 0.50, dynamic stop (nearest of bar extreme / range / VWAP / swing at least m x 5-min ATR away,
never wider than 25% of premium), targets 1R-4R repriced with Black-Scholes, 50/25/25 booking at T2/T3/T4 in whole lots
(4 lots), no time stop, everything closed at 14:59 (no carry-over), costs and 2% slippage on.
Validation/test runs are blocked until you run `python bt_run.py freeze --m ... --variant ...`.
