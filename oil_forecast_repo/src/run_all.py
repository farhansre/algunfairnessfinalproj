from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from model_utils import DATA_DIR

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
raw_csvs = sorted(DATA_DIR.glob('*.csv')) if DATA_DIR.exists() else []
steps = [
    '01_profile_inputs.py',
    '02_clean_eia_series.py',
    '03_build_panel.py',
    '04_train_price_model.py',
    '05_train_usage_model.py',
    '06_forecast_next_year.py',
]

if not raw_csvs:
    print(f"No raw CSV files found in {DATA_DIR}. Skipping data profiling and cleaning steps.", flush=True)
    steps = steps[2:]  # Skip first two steps

for step in steps:
    print(f'\n=== Running {step} ===', flush=True)
    subprocess.run([sys.executable, str(SRC / step)], check=True)
