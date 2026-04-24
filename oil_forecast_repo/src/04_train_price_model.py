from __future__ import annotations

from pathlib import Path
import pandas as pd

from model_utils import rolling_origin_evaluation, save_json, train_final_ridge

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs'

panel = pd.read_csv(OUT / 'model_panel.csv')
train = panel[(panel['year'] >= 2002) & (panel['year'] <= 2024)].copy()
feature_cols = [
    'state',
    'year',
    'annual_price_mean',
    'annual_price_mean_lag1',
    'annual_price_mean_lag2',
    'petroleum_total_thousand_barrels',
    'petroleum_total_thousand_barrels_lag1',
    'petroleum_energy_billion_btu',
    'natural_gas_total_billion_btu',
    'renewable_total_billion_btu',
    'natgas_supply_bcf',
    'price_yoy_pct',
]

results = rolling_origin_evaluation(train, feature_cols, 'target_next_year_price', min_train_years=8)
results['predictions'].to_csv(OUT / 'price_backtest_predictions.csv', index=False)
save_json(results['metrics'], OUT / 'price_model_metrics.json')

final_model = train_final_ridge(train, feature_cols, 'target_next_year_price')
forecast_input = train.groupby('state', as_index=False).tail(1)[feature_cols].copy()
forecast_input['year'] = forecast_input['year'] + 1
forecast_input['predicted_next_year_price'] = final_model.predict(forecast_input[feature_cols])
forecast_input[['state', 'year', 'predicted_next_year_price']].to_csv(OUT / 'price_next_year_forecast.csv', index=False)

print(results['metrics'])
print(forecast_input[['state', 'year', 'predicted_next_year_price']].to_string(index=False))
