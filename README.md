# Groww historical data downloader + quality report

Downloads 1-minute candles (with volume and OI for options) into local Parquet files, so backtests never hit the API.

## What it downloads

1. **Underlying**: 1-minute candles for every F&O stock (+ NIFTY, BANKNIFTY) and daily candles (for cross-checking).
2. **Options**: 1-minute candles for the contracts your strategy could actually pick: on each day, the ACTIVE expiry
   (nearest expiry with >= 4 calendar days left, i.e. we roll before expiry week) and every strike within `--band`
   (default 1) strikes of ATM across that day's low-to-high range, both CE and PE.
   Stocks are processed most-liquid-first, so the best data lands earliest. Everything is resumable.

## Setup

    python3 -m venv venv && source venv/bin/activate
    pip install -r requirements.txt
    cp .env.example .env     # fill in the same Groww TOTP values as the scanner

## Use (in this order)

    python downloader.py estimate        # ~5-10 min: measures real speed/size on 2 stocks, projects the full run
    caffeinate -i python downloader.py run --months 24 --workers 8 --rps 4      # Mac: keeps the laptop awake; Ctrl+C is safe, rerun resumes
    python downloader.py status          # progress + any failures
    python downloader.py quality         # writes data/reports/QUALITY.md and CSVs

Useful options: `--symbols RELIANCE,TCS`, `--phase stock|option`, `--start 2022-01-01`, `--band 2`, `--rps 4`.
Extending history later (e.g. `--start 2021-01-01`) only fetches what is missing.

## Notes

- Groww limits: 1-minute requests max 30 days (we use 28), data from 2020. Live-data limits are 10 req/s and 300/min;
  default is 4 req/s and the client halves its speed automatically if Groww pushes back.
- Login tokens expire daily at 6 AM; the client re-logs in by itself (TOTP).
- Known limitations are listed at the bottom of QUALITY.md (survivorship bias, no historical bid/ask or Greeks, ...).
- Written from Groww's public docs and tested offline against a simulated API; the first real run may reveal quirks.
  `status` shows failures with the first error message.
