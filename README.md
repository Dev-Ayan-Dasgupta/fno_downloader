# Pullback Strategy v1 — Signal Engine

This is a first, deliberately simple baseline for the new pullback strategy. It is
not yet a complete options backtester and must not be used for live trading.

## Set-up

Place `pullback_signals.py` in the same project directory as the existing
`bt_data.py` and `bt_signals.py`.

The signal engine reuses the existing data context:

```python
import bt_signals as ORB
from pullback_signals import PullbackParams, find_pullback_signals

ctx = ORB.build_ctx("RELIANCE", data_dir="data")
nifty5, nifty_reference = ORB.index_context("NIFTY", data_dir="data")

params = PullbackParams()
signals, rejects = find_pullback_signals(ctx, nifty5, params)

print(signals.head())
print(rejects["reason"].value_counts())
```

## Variant definitions

### A. VWAP retest

A completed 5-minute bar must satisfy:

- Bullish: close > EMA20 > EMA50, EMA20 rising, close above VWAP, and (by default)
  VWAP rising over the configured slope lookback.
- Pullback low reaches VWAP plus the allowed ATR tolerance.
- Bar closes back above VWAP plus the reclaim buffer.
- Bar is bullish and has the minimum body size.
- Entry is the open of the next 1-minute bar.

Bearish conditions are mirrored.

### B. Breakout-level retest

- The breakout level is the previous rolling 12-bar high/low, calculated with shift(1).
- A completed 5-minute candle must close beyond that level plus the breakout buffer.
- Only a later candle can retest the frozen level.
- The retest must reach the level within the ATR tolerance, close back on the breakout side,
  and form a directional confirmation candle.
- The retest expires after 12 completed 5-minute bars.
- Entry is the open of the next 1-minute bar.

## Baseline parameters

See `PullbackParams` in the source. The baseline uses:
- EMA20/EMA50 trend structure.
- VWAP and EMA20 slope confirmation.
- 0.15 ATR retest tolerance.
- 0.02 ATR VWAP reclaim buffer.
- 12-bar breakout lookback.
- 0.05 ATR breakout buffer.
- 12-bar maximum retest wait.
- 0.05 ATR minimum confirmation candle body.
- 0.10 ATR stop buffer beyond the confirmation candle extreme.

These are initial engineering defaults, not empirically optimised values.

## Important implementation notes

- Signals use completed 5-minute bars only.
- `entry_ts` is the timestamp of the next 1-minute candle; `entry_price` is that candle's open.
- Missing next-minute bars are rejected rather than filled with a later price.
- NIFTY confirmation defaults to `none`. Set it to `same_direction` only as a separate experiment.
- VWAP comes from the existing data layer. If stock volume is absent or zero, VWAP may be NaN and no VWAP-based setup will qualify.
- The engine emits underlying signals only. It does not yet select options, simulate exits, or calculate P&L.
- A signal is emitted once per symbol/day/direction/setup by default. VWAP and breakout setups are separate and can both signal on the same day.

## Next validation steps

1. Run this engine over a small symbol/date sample and inspect signals manually.
2. Add underlying trade simulation and compute MFE/MAE in initial-risk units.
3. Check the signal timestamps and entry opens against raw 1-minute candles.
4. Run a full chronological backtest before connecting options.


## v1.1 data-integrity safeguards

The signal engine now rejects:
- Invalid OHLC relationships.
- 5-minute high-low range above 15% of bar open.
- ATR above 10% of signal close.
- VWAP/EMA scale deviations above 20% from signal close.
- Next 1-minute entry OHLC with invalid structure.
- Next-minute entry open more than 5% away from the completed signal close.
- Initial underlying risk above 10% of entry price.
- Breakout reference levels more than 20% away from signal close.

These are defensive data-quality limits, not optimised trading filters. If your market-data vendor has legitimate instruments or corporate actions that breach them, inspect the records before changing the limits.

Signals also include:
- `initial_risk_pct`
- `entry_gap_pct`
- `overlap_group_id`
- `variant_overlap_count`
- `is_cross_variant_overlap`

Overlapping VWAP and breakout signals are retained for independent variant analysis. For a combined portfolio, use one event per `overlap_group_id` or define a pre-registered priority rule; do not count both as independent trades.

The smoke-test runner additionally writes:
- `pullback_reject_summary.csv`
- `pullback_overlaps.csv`

Run unit checks with:
```bash
python -m unittest -v test_pullback_signals.py
```


## Underlying trade simulator (v1)

New files:
- `pullback_trade.py`
- `pullback_backtest_underlying.py`
- `test_pullback_trade.py`

Run unit tests:
```bash
python -m unittest -v test_pullback_signals.py test_pullback_trade.py
```

Run the underlying simulator:
```bash
python pullback_backtest_underlying.py --data-dir data \
  --signals data/backtest/pullback_smoke/pullback_signals.csv \
  --symbols RELIANCE SBIN INFY
```

Outputs:
- `underlying_trades.csv`: one row per signal per exit policy (three policies).
- `underlying_summary.csv`: setup/policy performance, day-cluster bootstrap CI.
- `underlying_daily.csv`: daily net R series.
- `underlying_period_summary.csv`: descriptive Development/Validation/Test split.
- `sim_params.json`: execution/cost assumptions.

Exit policies:
1. Fixed 1.5R target.
2. Fixed 2R target.
3. Five-bar completed-1m swing trailing stop (initial stop remains active until trail can be computed).

Conservative mechanics:
- If stop and target are both touched in the same 1m candle, stop wins.
- Stop gaps fill at the worse opening price.
- Targets fill at the target price, not an optimistic opening price.
- Trailing stop is updated after a candle closes and is active only from the next candle.
- Entry must match the exact next 1m candle timestamp in the signal file.

Costs:
- Default slippage: 5 bps per side.
- Brokerage: Rs 20 per executed order.
- Other costs: 0 bps by default (configurable).
- Quantity: 1 share by default.
These are transparent placeholders for the underlying simulator; adjust to your actual broker and instrument cost schedule. They are not the options cost model.

Important:
- This simulator evaluates each setup independently. Overlapping VWAP/breakout signals are not combined into one portfolio.
- It also evaluates each signal independently; simultaneous positions are not constrained by shared capital or portfolio risk.
- The previously examined Test window is descriptive only and is not a fresh untouched holdout.


## v1.4 — risk-budget sizing and cost attribution

The underlying simulator now defaults to risk-budget sizing rather than one share:

- Default illustrative risk budget: Rs 1,000 per trade.
- Shares = floor(risk budget / initial risk per share), optionally capped by `--max-shares`.
- Brokerage remains a fixed per-order amount; it is not multiplied by share count.
- Slippage, brokerage, and configurable turnover-based other charges are reported separately.
- Gross and net R, gross and net P&L, and total costs including slippage are reported separately.
- This is still a simplified cost model. Set `--other-cost-bps` to a researched estimate or extend the model for the exact broker, exchange, taxes, and instrument. Do not interpret placeholder charges as a broker-verified contract note.

Run with the default illustrative risk budget:

```bash
python pullback_backtest_underlying.py --data-dir data \
  --signals data/backtest/pullback_smoke/pullback_signals.csv \
  --symbols RELIANCE SBIN INFY --risk-budget 1000
```

To retain fixed quantity for a diagnostic comparison only:

```bash
python pullback_backtest_underlying.py --data-dir data \
  --signals data/backtest/pullback_smoke/pullback_signals.csv \
  --symbols RELIANCE SBIN INFY --fixed-quantity --quantity 1
```

`risk_budget_rs=None` in Python selects fixed-share sizing. The previous Test window remains previously examined and must not be treated as a fresh holdout.

## v1.5 — options execution simulator (separate from underlying simulator)

`pullback_options.py` simulates a long CE/PE contract selected for each underlying
signal. The underlying signal defines the stop/target trigger; the option contract's
own premium OHLC determines P&L. On an underlying stop/target trigger, the option is
sold at the **next available 1-minute option bar open**. If no subsequent option bar
exists, the trade is marked as missing an exit quote rather than inventing a fill.
If no trigger occurs, it exits at the last option 1-minute close by the EOD cutoff.

### Required CSV inputs

1. `signals.csv`: the pullback signal output.
2. `contracts.csv`: one selected historical contract per signal. Generate a blank
   template with:

```bash
python make_option_contract_template.py --signals signals.csv --out contracts.csv
```

   Populate `option_symbol`, `option_type` (CE/PE), `strike`, `expiry`, `lot_size`
   (units per lot on that historical date), and `lots`. Selection must use only
   historical option-chain data available at the signal time. The script does not
   fabricate historical contracts or infer lot sizes.
3. `underlying_1m.csv`: columns `symbol,ts,o,h,l,c`.
4. `options_1m.csv`: columns `option_symbol,ts,o,h,l,c`.

All timestamps are minute-bar **start** timestamps in the same timezone. The
underlying and option entry bars must both exist at the exact signal `entry_ts`.

### Run

```bash
python pullback_backtest_options.py \
  --signals signals.csv \
  --contracts contracts.csv \
  --underlying-1m underlying_1m.csv \
  --options-1m options_1m.csv \
  --out data/backtest/pullback_options \
  --target-r 2 \
  --slippage-grid 0,1,2,3,5
```

`--lots N` overrides the `lots` column for a fixed-size experiment. Otherwise each
contract row's lot count is used. Quantity units = `lot_size × lots`; premium outlay
= entry premium × quantity units. Premium turnover is used for option charges—not
underlying spot × quantity.

### Groww NSE equity-options cost model (current published schedule checked 25-Nov-2026)

- Brokerage: ₹20 per executed order (₹40 round trip).
- STT: 0.15% on sell-side option premium turnover.
- Stamp duty: 0.003% on buy-side premium turnover.
- NSE options exchange transaction charge: 0.03503% on buy + sell premium turnover.
- SEBI turnover fee: 0.0001% on both sides.
- NSE IPFT: 0.0005% on both sides.
- GST: 18% on brokerage, exchange, SEBI and IPFT charges.

Rates are encoded as explicit parameters in `OptionSimParams`; confirm them against
Groww's tariff/contract note before relying on a result. The default slippage grid
(0, 1, 2, 3, 5 bps per side) is a sensitivity analysis, not observed execution data.

Important limitations: this is a long-option premium simulator; it does not yet
model option-chain selection, delta filtering, bid/ask spread, market impact, partial
fills, liquidity/volume constraints, expiry settlement, or portfolio-level capital
and concurrency constraints. Contract selection is supplied externally and must be
historically correct. It is not approved for live trading.
