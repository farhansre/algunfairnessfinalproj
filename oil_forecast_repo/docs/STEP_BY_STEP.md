# Step-by-step walkthrough

## Step 1: Inspect the raw files

Run:

```bash
python src/01_profile_inputs.py
```

What it does:

- counts rows and columns,
- records sample column names,
- shows which files are weekly and which are annual EIA exports.

Why this matters:

- the uploaded CSVs are not all in the same shape,
- you should verify the structure before writing a parser.

## Step 2: Clean the raw EIA exports

Run:

```bash
python src/02_clean_eia_series.py
```

What it does:

- converts weekly price files into annual average price tables,
- converts stacked annual exports into tidy long tables,
- writes cleaned CSVs into `outputs/`.

## Step 3: Build the state-year modeling panel

Run:

```bash
python src/03_build_panel.py
```

What it does:

- isolates the total petroleum consumption series (`P1TCP`),
- isolates petroleum energy total (`PMTCB`),
- isolates natural gas total (`NNTCB`),
- isolates renewable total (`RETCB`),
- joins them to annual price data,
- creates lagged features,
- creates next-year targets.

## Step 4: Train and backtest the price model

Run:

```bash
python src/04_train_price_model.py
```

What it does:

- uses rolling-origin forecasting,
- saves held-out predictions,
- saves MAE, RMSE, and R-squared,
- trains the final Ridge model on the full training window,
- forecasts the next annual price for each state.

## Step 5: Train and backtest the usage model

Run:

```bash
python src/05_train_usage_model.py
```

What it does:

- predicts next-year petroleum consumption,
- conditions on the current price level and energy-system covariates,
- writes held-out predictions and next-year usage forecasts.

## Step 6: Combine results into one final forecast table

Run:

```bash
python src/06_forecast_next_year.py
```

What it does:

- merges the price and usage forecasts,
- computes predicted percentage change versus 2024 usage,
- writes a single summary CSV.

## Step 7: Run everything in one command

Run:

```bash
python src/run_all.py
```

This reproduces the entire pipeline from raw CSVs to final forecast summary.
