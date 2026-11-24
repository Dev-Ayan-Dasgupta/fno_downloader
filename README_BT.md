# Backtest stage 1: data layer, pricing, costs

Drop these files into your `fno_downloader` folder (they import `fetch.py` / `plan.py` / `client.py` from there):
`bt_config.py  bt_data.py  bt_pricing.py  bt_costs.py  bt_check.py  selftest_bt.py`

    python selftest_bt.py            # 28 offline checks, no network or data needed
    python bt_check.py RELIANCE      # runs the loader on YOUR downloaded data and prints a sanity summary

What the data layer guarantees (spec v0.2): only 09:15-15:30 bars, no NaN/duplicate/inconsistent rows, ATR/RVOL/liquidity
computed from PRIOR sessions only, OI forward-filled inside the day for at most 5 minutes, sessions within 5 days of a
>25% overnight gap flagged for exclusion, and on-demand fetching + caching of option contract-days (for carried trades)
in data/opt_extra. Units: option volume = shares; historical OI = lots x lot size / 100 (use OI only as % changes).
Cost model: spec section 12.7 (reproduces a published worked example to within Rs 0.1).
