from __future__ import annotations

from pathlib import Path
import pandas as pd

from model_utils import parse_single_series_annual_csv, parse_stacked_annual_series_csv, parse_weekly_price_csv, DATA_DIR

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs'

ca_price = parse_weekly_price_csv(DATA_DIR / 'retailgasolineprices_ca.csv', 'CA')
ny_price = parse_weekly_price_csv(DATA_DIR / 'retailgasolineprices_ny.csv', 'NY')
annual_price = pd.concat([ca_price, ny_price], ignore_index=True)
annual_price.to_csv(OUT / 'clean_annual_prices.csv', index=False)

ca_pet = parse_stacked_annual_series_csv(DATA_DIR / 'petroleumconsumption_ca.csv', 'CA', value_name='petroleum_value')
ny_pet = parse_stacked_annual_series_csv(DATA_DIR / 'petroleumconsumption_ny.csv', 'NY', value_name='petroleum_value')
pet = pd.concat([ca_pet, ny_pet], ignore_index=True)
pet.to_csv(OUT / 'clean_petroleum_long.csv', index=False)

ca_total = parse_stacked_annual_series_csv(DATA_DIR / 'totalenergyconsumption_ca.csv', 'CA', value_name='energy_value')
ny_total = parse_stacked_annual_series_csv(DATA_DIR / 'totalenergyconsumption_ny.csv', 'NY', value_name='energy_value')
total = pd.concat([ca_total, ny_total], ignore_index=True)
total.to_csv(OUT / 'clean_total_energy_long.csv', index=False)

ca_gas = parse_single_series_annual_csv(DATA_DIR / 'futurenaturalgassuply_ca.csv', 'CA', 'natgas_supply_bcf')
ny_gas = parse_single_series_annual_csv(DATA_DIR / 'futurenaturalgassupply_ny.csv', 'NY', 'natgas_supply_bcf')
natgas = pd.concat([ca_gas, ny_gas], ignore_index=True)
natgas.to_csv(OUT / 'clean_natgas_supply.csv', index=False)

print('Wrote cleaned files:')
for name in ['clean_annual_prices.csv', 'clean_petroleum_long.csv', 'clean_total_energy_long.csv', 'clean_natgas_supply.csv']:
    print('-', OUT / name)
