"""Backtest constants (spec v0.2). Change only with a dated entry in the spec doc."""
import os

RISK_FREE = 0.065            # annual, for Black-Scholes (no dividends modelled: approximation)
SPLIT_GAP = 0.25             # overnight gap that marks a probable unadjusted split/bonus
SPLIT_BEFORE = 5             # sessions excluded before such a gap
SPLIT_AFTER = 20             # ... and after it (ATR 14 / volume baseline 20 stay contaminated that long)
LIQ_WINDOW = 20              # sessions for the point-in-time option-liquidity measure
LIQ_MIN = 0.80               # minimum median candle coverage
OI_FFILL_MIN = 5             # forward-fill OI inside the day for at most this many minutes
SESSION_MINUTES = 375


def data_dir():
    return os.getenv("DATA_DIR", "data")
