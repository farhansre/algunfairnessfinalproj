from __future__ import annotations

from pathlib import Path
import pandas as pd

from model_utils import fit_state_trend_forecast

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs'

price = pd.read_csv(OUT / 'clean_annual_prices.csv')
pet = pd.read_csv(OUT / 'clean_petroleum_long.csv')
energy = pd.read_csv(OUT / 'clean_total_energy_long.csv')
natgas = pd.read_csv(OUT / 'clean_natgas_supply.csv')

pet_total = pet[pet['series_key'].str.contains('P1TCP', na=False)].copy()
pet_total = pet_total.rename(columns={'petroleum_value': 'petroleum_total_thousand_barrels'})[['state', 'year', 'petroleum_total_thousand_barrels']]

energy_total = energy[energy['series_key'].str.contains('PMTCB', na=False)].copy()
energy_total = energy_total.rename(columns={'energy_value': 'petroleum_energy_billion_btu'})[['state', 'year', 'petroleum_energy_billion_btu']]

energy_ng = energy[energy['series_key'].str.contains('NNTCB', na=False)].copy()
energy_ng = energy_ng.rename(columns={'energy_value': 'natural_gas_total_billion_btu'})[['state', 'year', 'natural_gas_total_billion_btu']]

energy_renew = energy[energy['series_key'].str.contains('RETCB', na=False)].copy()
energy_renew = energy_renew.rename(columns={'energy_value': 'renewable_total_billion_btu'})[['state', 'year', 'renewable_total_billion_btu']]

natgas_trend = fit_state_trend_forecast(natgas[['state', 'year', 'natgas_supply_bcf']].copy(), 'natgas_supply_bcf', years=range(2022, 2028))
natgas_trend = natgas_trend.rename(columns={'natgas_supply_bcf_trend_forecast': 'natgas_supply_bcf'})
natgas_full = pd.concat([
    natgas[['state', 'year', 'natgas_supply_bcf']].copy(),
    natgas_trend[['state', 'year', 'natgas_supply_bcf']].copy(),
], ignore_index=True).drop_duplicates(['state', 'year'], keep='first')

panel = price.merge(pet_total, on=['state', 'year'], how='left')
panel = panel.merge(energy_total, on=['state', 'year'], how='left')
panel = panel.merge(energy_ng, on=['state', 'year'], how='left')
panel = panel.merge(energy_renew, on=['state', 'year'], how='left')
panel = panel.merge(natgas_full, on=['state', 'year'], how='left')
panel = panel.sort_values(['state', 'year']).reset_index(drop=True)

for col in ['annual_price_mean', 'petroleum_total_thousand_barrels', 'petroleum_energy_billion_btu', 'natural_gas_total_billion_btu', 'renewable_total_billion_btu', 'natgas_supply_bcf']:
    panel[f'{col}_lag1'] = panel.groupby('state')[col].shift(1)
    panel[f'{col}_lag2'] = panel.groupby('state')[col].shift(2)

panel['price_yoy_pct'] = panel.groupby('state')['annual_price_mean'].pct_change(fill_method=None)
panel['usage_yoy_pct'] = panel.groupby('state')['petroleum_total_thousand_barrels'].pct_change(fill_method=None)
panel['target_next_year_price'] = panel.groupby('state')['annual_price_mean'].shift(-1)
panel['target_next_year_usage'] = panel.groupby('state')['petroleum_total_thousand_barrels'].shift(-1)

panel.to_csv(OUT / 'model_panel.csv', index=False)
print(panel.tail(12).to_string(index=False))
print(f'Wrote {OUT / "model_panel.csv"}')
