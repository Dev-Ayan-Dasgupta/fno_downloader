"""Save current lot sizes (nearest expiry) to data/meta/lot_sizes.json.  Run once:  python bt_lots.py
Historical lot sizes are not available from Groww, so rupee costs for older dates are approximate."""
import json
import os
from datetime import date

import config  # noqa: F401  (loads .env)
from client import Client
import bt_config as B

df = Client(rps=2).call("get_all_instruments")
fno = df[df["segment"] == "FNO"].copy()
fno["expiry_date"] = fno["expiry_date"].astype(str)
fno = fno[fno["expiry_date"] >= date.today().isoformat()].sort_values("expiry_date")
lots = {}
for und, g in fno.groupby("underlying_symbol"):
    lots[str(und)] = int(g["lot_size"].iloc[0])
os.makedirs(os.path.join(B.data_dir(), "meta"), exist_ok=True)
json.dump(lots, open(os.path.join(B.data_dir(), "meta", "lot_sizes.json"), "w"))
print(f"Saved {len(lots)} lot sizes. Examples:", {k: lots[k] for k in list(lots)[:5]})
