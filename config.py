import os
from datetime import date
from dotenv import load_dotenv

load_dotenv()
DATA_DIR = os.getenv("DATA_DIR", "data")
INDEX_UNDERLYINGS = {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "NIFTYNXT50", "SENSEX", "BANKEX"}
EXTRA_INDICES = ["NIFTY", "BANKNIFTY"]      # downloaded with the stock phase (for market-regime context)
GRID_EPOCH = date(2020, 1, 6)               # stock 1-min requests use fixed 28-day blocks from here
GRID_DAYS = 28                              # Groww allows max 30 days per 1-min request
DAILY_WINDOW_DAYS = 170                     # Groww allows max 180 days per daily request
OPT_MAX_SPAN_DAYS = 28
OPT_MERGE_GAP_DAYS = 5                      # bridge gaps up to this many calendar days when merging a contract's windows
