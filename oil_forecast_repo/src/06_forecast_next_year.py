from __future__ import annotations

from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs'

price = pd.read_csv(OUT / 'price_next_year_forecast.csv')
usage = pd.read_csv(OUT / 'usage_next_year_forecast.csv')
summary = price.merge(usage, on=['state', 'year'], how='outer')
summary['predicted_usage_change_pct_vs_2024'] = None

panel = pd.read_csv(OUT / 'model_panel.csv')
last_obs = panel[panel['year'] == 2024][['state', 'petroleum_total_thousand_barrels']].rename(columns={'petroleum_total_thousand_barrels': 'usage_2024'})
summary = summary.merge(last_obs, on='state', how='left')
summary['predicted_usage_change_pct_vs_2024'] = (summary['predicted_next_year_usage'] / summary['usage_2024'] - 1.0) * 100.0
summary.to_csv(OUT / 'forecast_summary.csv', index=False)

payload = {
    'price_metrics': json.loads((OUT / 'price_model_metrics.json').read_text()),
    'usage_metrics': json.loads((OUT / 'usage_model_metrics.json').read_text()),
}
(OUT / 'metrics_summary.json').write_text(json.dumps(payload, indent=2))
print(summary.to_string(index=False))
print(f'Wrote {OUT / "forecast_summary.csv"}')
