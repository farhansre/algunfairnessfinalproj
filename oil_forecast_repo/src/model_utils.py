from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple

import json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'
OUTPUT_DIR = ROOT / 'outputs'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_weekly_price_csv(path: Path, state: str) -> pd.DataFrame:
    raw = pd.read_csv(path)
    date_col = raw.columns[0]
    value_col = raw.columns[1]
    df = raw.iloc[6:, [0, 1]].copy()
    df.columns = ['date_raw', 'price_per_gallon']
    df['date'] = pd.to_datetime(df['date_raw'].astype(str), format='%Y%m%d', errors='coerce')
    df['price_per_gallon'] = pd.to_numeric(df['price_per_gallon'], errors='coerce')
    df = df.dropna(subset=['date', 'price_per_gallon']).sort_values('date')
    df['year'] = df['date'].dt.year
    annual = (
        df.groupby('year', as_index=False)
        .agg(
            annual_price_mean=('price_per_gallon', 'mean'),
            annual_price_median=('price_per_gallon', 'median'),
            weekly_points=('price_per_gallon', 'size'),
        )
    )
    annual['state'] = state
    annual['source_file'] = path.name
    return annual


def parse_stacked_annual_series_csv(path: Path, state: str, value_name: str | None = None) -> pd.DataFrame:
    raw = pd.read_csv(path)
    frames = []
    for idx in range(0, raw.shape[1], 2):
        if idx + 1 >= raw.shape[1]:
            continue
        key_col = raw.columns[idx]
        val_col = raw.columns[idx + 1]
        series_key = str(raw.iloc[0, idx + 1]).split(',')[0].strip()
        series_units = str(raw.iloc[1, idx + 1]).strip()
        df = raw.iloc[6:, [idx, idx + 1]].copy()
        df.columns = ['year', 'value']
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        df = df.dropna(subset=['year', 'value'])
        df['year'] = df['year'].astype(int)
        df['state'] = state
        df['series_key'] = series_key
        df['units'] = series_units
        if value_name:
            df = df.rename(columns={'value': value_name})
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def parse_single_series_annual_csv(path: Path, state: str, value_name: str) -> pd.DataFrame:
    raw = pd.read_csv(path)
    df = raw.iloc[6:, [0, 1]].copy()
    df.columns = ['year', value_name]
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df[value_name] = pd.to_numeric(df[value_name], errors='coerce')
    df = df.dropna(subset=['year', value_name])
    df['year'] = df['year'].astype(int)
    df['state'] = state
    df['series_key'] = str(raw.iloc[0, 1]).split(',')[0].strip()
    return df


def fit_state_trend_forecast(df: pd.DataFrame, value_col: str, years: Iterable[int]) -> pd.DataFrame:
    rows = []
    for state, g in df.groupby('state'):
        x = g['year'].to_numpy().reshape(-1, 1)
        y = g[value_col].to_numpy()
        coeffs = np.polyfit(g['year'], y, deg=1)
        trend = np.poly1d(coeffs)
        for year in years:
            rows.append({'state': state, 'year': int(year), f'{value_col}_trend_forecast': float(trend(year))})
    return pd.DataFrame(rows)


def build_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                'num',
                Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler()),
                ]),
                numeric_features,
            ),
            (
                'cat',
                Pipeline([
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore')),
                ]),
                categorical_features,
            ),
        ]
    )


def rolling_origin_evaluation(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    min_train_years: int = 8,
) -> Dict[str, object]:
    data = df.dropna(subset=[target_col]).sort_values(['year', 'state']).copy()
    years = sorted(data['year'].unique())
    numeric_features = [c for c in feature_cols if pd.api.types.is_numeric_dtype(data[c])]
    categorical_features = [c for c in feature_cols if c not in numeric_features]
    predictions = []
    for split_idx in range(min_train_years, len(years)):
        train_years = years[:split_idx]
        test_year = years[split_idx]
        train = data[data['year'].isin(train_years)].copy()
        test = data[data['year'] == test_year].copy()
        if train.empty or test.empty:
            continue
        pipe = Pipeline([
            ('prep', build_preprocessor(numeric_features, categorical_features)),
            ('model', Ridge(alpha=1.0)),
        ])
        pipe.fit(train[feature_cols], train[target_col])
        preds = pipe.predict(test[feature_cols])
        fold = test[['state', 'year', target_col]].copy()
        fold['prediction'] = preds
        predictions.append(fold)
    preds_df = pd.concat(predictions, ignore_index=True)
    metrics = {
        'mae': float(mean_absolute_error(preds_df[target_col], preds_df['prediction'])),
        'rmse': float(np.sqrt(mean_squared_error(preds_df[target_col], preds_df['prediction']))),
        'r2': float(r2_score(preds_df[target_col], preds_df['prediction'])),
        'n_predictions': int(len(preds_df)),
    }
    return {'predictions': preds_df, 'metrics': metrics}


def train_final_ridge(df: pd.DataFrame, feature_cols: list[str], target_col: str) -> Pipeline:
    data = df.dropna(subset=[target_col]).copy()
    numeric_features = [c for c in feature_cols if pd.api.types.is_numeric_dtype(data[c])]
    categorical_features = [c for c in feature_cols if c not in numeric_features]
    pipe = Pipeline([
        ('prep', build_preprocessor(numeric_features, categorical_features)),
        ('model', Ridge(alpha=1.0)),
    ])
    pipe.fit(data[feature_cols], data[target_col])
    return pipe


def save_json(payload: Dict[str, object], path: Path) -> None:
    path.write_text(json.dumps(payload, indent=2))
