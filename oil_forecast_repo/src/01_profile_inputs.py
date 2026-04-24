from __future__ import annotations

from pathlib import Path
import pandas as pd

from model_utils import DATA_DIR

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs' / 'input_profile.csv'

rows = []
for path in sorted(DATA_DIR.glob('*.csv')):
    df = pd.read_csv(path)
    rows.append({
        'file_name': path.name,
        'rows': df.shape[0],
        'cols': df.shape[1],
        'first_columns': ' | '.join(df.columns[:5]),
        'sample_top_left': str(df.iloc[0, 0]) if not df.empty else '',
    })

if not rows:
    profile = pd.DataFrame(columns=['file_name', 'rows', 'cols', 'first_columns', 'sample_top_left'])
    profile.to_csv(OUT, index=False)
    print(f'No raw CSV files found in {DATA_DIR}.')
    print(f'Wrote empty profile to {OUT}')
    raise SystemExit(0)

profile = pd.DataFrame(rows).sort_values('file_name')
profile.to_csv(OUT, index=False)
print(profile.to_string(index=False))
print(f'Wrote {OUT}')
