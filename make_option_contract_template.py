"""Create a contract-mapping template from pullback signals; fill using historical chain data."""
import argparse
import pandas as pd
from pullback_options import make_signal_key

p=argparse.ArgumentParser()
p.add_argument('--signals',required=True)
p.add_argument('--out',default='option_contracts_template.csv')
a=p.parse_args()
s=pd.read_csv(a.signals)
s['signal_key']=s.apply(make_signal_key,axis=1)
out=s[['signal_key','symbol','day','direction','setup','entry_ts','entry_price']].copy()
for c in ['option_symbol','option_type','strike','expiry','lot_size','lots']:
    out[c]=''
out.to_csv(a.out,index=False)
print(f'Wrote {len(out)} rows to {a.out}; populate contract fields from historical option-chain data.')
