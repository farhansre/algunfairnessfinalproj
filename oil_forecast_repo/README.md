# oil_forecast_repo

This repository turns the uploaded EIA CSV exports into a reproducible modeling pipeline.

## What this repository does

1. **Profiles the raw files** so you can see what shape each export arrives in.
2. **Cleans each source** into standard year-by-state tables.
3. **Builds one modeling panel** with California and New York aligned by year.
4. **Trains a next-year price model** for annual state gasoline/oil-price proxies.
5. **Trains a next-year usage model** for annual petroleum consumption.
6. **Writes forecasts and metrics** into `outputs/`.

## Modeling choice

Because the uploaded data covers only **two states**, this repository uses a **small panel forecasting setup** rather than a large cross-state machine-learning system.

- **Price target**: next year's annual average retail gasoline price proxy for each state.
- **Usage target**: next year's total petroleum consumption (`P1TCP`) in thousand barrels.
- **Forecast horizon**: the next annual step after the last fully aligned annual panel year, which is **2025**.

This is the cleanest defensible setup because:

- the weekly price files run through 2026,
- most annual state consumption files run through 2024,
- the natural-gas supply files stop in 2021 and therefore need trend extrapolation for later years.

## Raw files used

- `retailgasolineprices_ca.csv`
- `retailgasolineprices_ny.csv`
- `petroleumconsumption_ca.csv`
- `petroleumconsumption_ny.csv`
- `totalenergyconsumption_ca.csv`
- `totalenergyconsumption_ny.csv`
- `futurenaturalgassuply_ca.csv`
- `futurenaturalgassupply_ny.csv`

## Cleaning logic

### 1. Weekly retail prices -> annual state prices

The retail price files are weekly EIA exports with metadata rows on top.

The script:

- drops the metadata rows,
- parses the YYYYMMDD dates,
- converts prices to numeric,
- groups by calendar year,
- calculates annual mean and median price.

### 2. Petroleum consumption and total energy files -> long annual tables

These files are exported in a repeated two-column pattern:

- left column = year/metadata labels,
- right column = values,
- repeated once per series.

The script walks across the file two columns at a time and produces a tidy long table with:

- `state`
- `year`
- `series_key`
- `value`
- `units`

### 3. Natural gas supply -> annual state series

The natural-gas files contain one annual series per state. The script converts them to a simple year/value/state table.

### 4. Final model panel

The panel keeps these core fields per state-year:

- annual price mean
- total petroleum consumption
- petroleum energy consumption
- total natural gas consumption
- total renewable consumption
- natural gas supply
- lag-1 and lag-2 versions of the major predictors
- year-over-year percent changes
- next-year targets for price and usage

## Feature engineering

### Price model features

- `state`
- `year`
- current annual price
- lag-1 price
- lag-2 price
- current petroleum usage
- lag-1 petroleum usage
- petroleum energy consumption
- natural gas consumption
- renewable energy consumption
- natural gas supply
- price year-over-year change

### Usage model features

- `state`
- `year`
- current annual price
- lag-1 price
- current petroleum usage
- lag-1 petroleum usage
- lag-2 petroleum usage
- petroleum energy consumption
- lag-1 petroleum energy consumption
- natural gas consumption
- renewable energy consumption
- natural gas supply
- usage year-over-year change

## Model class

Both tasks use the same structure:

- median imputation for missing numeric values,
- standard scaling for numeric variables,
- one-hot encoding for state,
- **Ridge regression** as the final estimator.

Ridge is used here because the dataset is small and highly collinear. A heavier model would overfit quickly.

## Time-series evaluation

The repository does **rolling-origin backtesting**:

- train on early years,
- predict the next unseen year,
- roll the training window forward,
- aggregate error metrics across all held-out years.

This is better than random train/test splitting because the task is forecasting.

## Files and scripts

### `src/01_profile_inputs.py`
Prints and saves a compact inventory of the raw CSV files.

### `src/02_clean_eia_series.py`
Runs all source-specific cleaning steps and writes cleaned CSVs.

### `src/03_build_panel.py`
Extracts the target series, merges them, creates lags, and builds the modeling panel.

### `src/04_train_price_model.py`
Backtests and trains the price model, then writes the next-year price forecast.

### `src/05_train_usage_model.py`
Backtests and trains the usage model, then writes the next-year usage forecast.

### `src/06_forecast_next_year.py`
Combines the outputs into one summary table.

### `src/run_all.py`
Runs the whole repository in order.

## How to run

```bash
cd /mnt/data/oil_forecast_repo
python src/run_all.py
```

## Main outputs

- `outputs/input_profile.csv`
- `outputs/clean_annual_prices.csv`
- `outputs/clean_petroleum_long.csv`
- `outputs/clean_total_energy_long.csv`
- `outputs/clean_natgas_supply.csv`
- `outputs/model_panel.csv`
- `outputs/price_backtest_predictions.csv`
- `outputs/usage_backtest_predictions.csv`
- `outputs/price_model_metrics.json`
- `outputs/usage_model_metrics.json`
- `outputs/price_next_year_forecast.csv`
- `outputs/usage_next_year_forecast.csv`
- `outputs/forecast_summary.csv`

## Important limitations

1. The dataset includes only **California and New York**, so these models are useful as a worked forecasting prototype, not as a broad U.S. state model.
2. Retail gasoline price is used as the practical state-level price proxy for oil-related consumer price exposure.
3. Natural-gas supply is only observed through 2021, so later years rely on a simple state-specific linear trend extrapolation.
4. The final forecast is therefore a **small-data scenario forecast**, not a production-grade energy market model.
