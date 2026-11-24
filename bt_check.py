"""Run the loader on your REAL data and print a sanity summary:  python bt_check.py RELIANCE"""
import sys
import pandas as pd
import bt_data as D

sym = sys.argv[1] if len(sys.argv) > 1 else "RELIANCE"
s = D.load_stock_1m(sym)
print(f"{sym}: {len(s):,} clean 1-min stock rows, {s['day'].nunique()} sessions, {s['day'].min()} -> {s['day'].max()}")
d = D.daily_table(s)
print("excluded sessions (split/bonus gaps):", len(D.exclusion_days(d)), "| gap days >25%:", int((d['gap'] > 0.25).sum()))
print("daily ATR% (latest 3):", d["atr14_pct"].dropna().tail(3).round(4).to_list())
f = D.to_5m(s)
print("5-min bars:", len(f), "| median bars/day:", int(f.groupby('day').size().median()))
r = D.rvol_table(s)
print("RVOL at 09:30 (minute 14), median:", round(float(r[14].median()), 2), "| latest:", round(float(r[14].dropna().iloc[-1]), 2))
print("opening range (latest):", D.opening_range(s).tail(1).round(2).to_dict("records"))
o = D.OptionStore(sym)
if o.df.empty:
    print("no option data found")
else:
    print(f"options: {len(o.df):,} rows, {o.df['contract'].nunique()} contracts, expiries {o.df['expiry'].min()} -> {o.df['expiry'].max()}")
    print("OI present raw vs after 5-min fill:", round(float(o.df['oi'].notna().mean()), 2), "->", round(float(o.df['oi_f'].notna().mean()), 2))
    liq = o.liquidity().dropna()
    print("option liquidity (point-in-time median coverage): latest", round(float(liq.iloc[-1]), 2),
          "| share of sessions passing 80%:", round(float((liq >= 0.8).mean()), 2))
