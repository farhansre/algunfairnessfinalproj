"""
07_lime_interpretability.py

Purpose
-------
Add an interpretability standard to the oil/gas forecasting project using LIME.
This script explains the trained next-year price model and next-year usage model
by showing which features locally pushed a specific California/New York forecast
up or down.

Repository role
---------------
Run this after the cleaning, panel-building, and model-training scripts:

    python src/01_profile_inputs.py
    python src/02_clean_eia_series.py
    python src/03_build_panel.py
    python src/04_train_price_model.py
    python src/05_train_usage_model.py
    python src/06_forecast_next_year.py
    python src/07_lime_interpretability.py

Expected inputs
---------------
Preferred:
    data/processed/model_panel.csv
    models/price_model.joblib
    models/usage_model.joblib

Fallback behavior:
    If saved models are unavailable, this script retrains interpretable sklearn
    RandomForestRegressor models from the processed panel. This keeps the script
    runnable for the writeup/demo even if model artifacts were not committed.

Outputs
-------
    outputs/interpretability/lime_price_explanations.csv
    outputs/interpretability/lime_usage_explanations.csv
    outputs/interpretability/lime_price_explanations.html
    outputs/interpretability/lime_usage_explanations.html
    outputs/interpretability/interpretability_writeup.md

Install requirements
--------------------
    pip install pandas numpy scikit-learn joblib lime matplotlib
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from lime.lime_tabular import LimeTabularExplainer
except ImportError as exc:
    raise SystemExit(
        "LIME is not installed. Run: pip install lime\n"
        "Then rerun: python src/07_lime_interpretability.py"
    ) from exc


RANDOM_STATE = 3358
STATE_COLUMNS = {"state", "State", "state_abbrev", "STATE"}
YEAR_COLUMNS = {"year", "Year", "YEAR"}

PRICE_TARGET_CANDIDATES = [
    # Current pipeline name
    "target_next_year_price",
    # Older/alternate names kept so the script is reusable
    "next_year_price",
    "target_next_price",
    "price_next_year",
    "retail_gasoline_price_next_year",
    "annual_avg_price_next_year",
]

USAGE_TARGET_CANDIDATES = [
    # Current pipeline name
    "target_next_year_usage",
    # Older/alternate names kept so the script is reusable
    "next_year_petroleum_consumption",
    "target_next_petroleum_consumption",
    "petroleum_consumption_next_year",
    "total_petroleum_consumption_next_year",
    "P1TCP_next_year",
]


def find_project_root() -> Path:
    """Return the repository root whether the script is run from root or src/."""
    here = Path(__file__).resolve()
    if here.parent.name == "src":
        return here.parent.parent
    return Path.cwd()


def pick_first_existing(columns: Iterable[str], candidates: list[str], label: str) -> str:
    """Find a target column from common names and fail loudly if not found."""
    column_set = set(columns)
    for candidate in candidates:
        if candidate in column_set:
            return candidate
    raise ValueError(
        f"Could not find {label} target column. Tried: {candidates}.\n"
        f"Available columns are: {sorted(column_set)}"
    )


def load_panel(root: Path) -> pd.DataFrame:
    """Load the processed modeling panel."""
    candidate_paths = [
        root / "data" / "processed" / "model_panel.csv",
        root / "data" / "processed" / "panel.csv",
        root / "outputs" / "model_panel.csv",
        root / "outputs" / "panel.csv",
    ]
    for path in candidate_paths:
        if path.exists():
            return pd.read_csv(path)

    raise FileNotFoundError(
        "Could not find a processed panel. Expected one of:\n"
        + "\n".join(str(p) for p in candidate_paths)
        + "\nRun the cleaning/panel scripts first."
    )


def make_features_and_target(
    df: pd.DataFrame,
    target_col: str,
) -> Tuple[pd.DataFrame, pd.Series, list[str], pd.DataFrame]:
    """
    Keep numeric modeling features, one-hot encode state, and drop leakage columns.

    This function intentionally removes all next-year targets other than the current
    target so the explanation describes the forecasting model, not leakage.
    """
    working = df.copy()
    working = working.dropna(subset=[target_col])

    id_cols = [c for c in working.columns if c in STATE_COLUMNS or c in YEAR_COLUMNS]
    leakage_cols = [
        c
        for c in working.columns
        if c != target_col and (
            c.startswith("next_year_")
            or c.endswith("_next_year")
            or c.startswith("target_next")
        )
    ]

    y = working[target_col].astype(float)
    X_raw = working.drop(columns=[target_col] + leakage_cols, errors="ignore")

    # Keep year as a feature, because time trend is meaningful. Keep state via one-hot encoding.
    state_col = next((c for c in X_raw.columns if c in STATE_COLUMNS), None)
    if state_col is not None:
        X_raw[state_col] = X_raw[state_col].astype(str)
        X = pd.get_dummies(X_raw, columns=[state_col], drop_first=False)
    else:
        X = X_raw.copy()

    # Drop columns that are clearly identifiers but not useful model features.
    non_numeric_cols = X.select_dtypes(exclude=[np.number, "bool"]).columns.tolist()
    X = X.drop(columns=non_numeric_cols, errors="ignore")

    # Convert booleans from get_dummies to numeric 0/1.
    for col in X.columns:
        if X[col].dtype == bool:
            X[col] = X[col].astype(int)

    X = X.replace([np.inf, -np.inf], np.nan)
    return X, y, id_cols, working


def load_or_train_model(
    root: Path,
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
) -> Pipeline:
    """Load a saved model if present; otherwise train a fallback model."""
    candidate_paths = [
        root / "models" / f"{model_name}.joblib",
        root / "models" / f"{model_name}_model.joblib",
        root / "outputs" / f"{model_name}_model.joblib",
    ]
    for path in candidate_paths:
        if path.exists():
            loaded = joblib.load(path)
            # If the stored object is not a Pipeline, wrap it with imputation/scaling.
            if isinstance(loaded, Pipeline):
                return loaded
            return Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("model", loaded),
                ]
            )

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=500,
                    max_depth=4,
                    min_samples_leaf=2,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    model.fit(X, y)
    return model


def evaluate_model(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    """Produce simple in-sample diagnostics for the interpretation appendix."""
    preds = model.predict(X)
    return {
        "mae": float(mean_absolute_error(y, preds)),
        "r2": float(r2_score(y, preds)) if len(y) > 1 else float("nan"),
    }


def explain_with_lime(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    original_rows: pd.DataFrame,
    target_label: str,
    output_prefix: Path,
    num_features: int = 10,
) -> pd.DataFrame:
    """
    Run LIME on the latest observation for each state.

    LIME perturbs the selected row locally, queries the model on those perturbed
    rows, and fits a simple local surrogate model. The output weights are not
    global causal effects; they are local reasons for that specific prediction.
    """
    # LimeTabularExplainer expects fully numeric, finite training data.
    imputed_X = pd.DataFrame(
        SimpleImputer(strategy="median").fit_transform(X),
        columns=X.columns,
        index=X.index,
    )

    explainer = LimeTabularExplainer(
        training_data=imputed_X.to_numpy(),
        feature_names=list(imputed_X.columns),
        mode="regression",
        discretize_continuous=True,
        random_state=RANDOM_STATE,
    )

    year_col = next((c for c in original_rows.columns if c in YEAR_COLUMNS), None)
    state_col = next((c for c in original_rows.columns if c in STATE_COLUMNS), None)

    if year_col and state_col:
        rows_to_explain = (
            original_rows.assign(_row_id=original_rows.index)
            .sort_values([state_col, year_col])
            .groupby(state_col, as_index=False)
            .tail(1)
            .sort_values([state_col, year_col])
        )
        row_indices = rows_to_explain["_row_id"].tolist()
    else:
        row_indices = [int(imputed_X.index.max())]

    all_records: list[dict] = []
    html_blocks: list[str] = []

    for row_index in row_indices:
        x_row = imputed_X.loc[row_index]
        prediction = float(model.predict(pd.DataFrame([X.loc[row_index]], columns=X.columns))[0])

        exp = explainer.explain_instance(
            data_row=x_row.to_numpy(),
            predict_fn=lambda arr: model.predict(pd.DataFrame(arr, columns=X.columns)),
            num_features=num_features,
        )

        state_value = original_rows.loc[row_index, state_col] if state_col else "unknown_state"
        year_value = original_rows.loc[row_index, year_col] if year_col else "unknown_year"

        for rank, (feature_rule, weight) in enumerate(exp.as_list(), start=1):
            all_records.append(
                {
                    "target": target_label,
                    "state": state_value,
                    "source_year": year_value,
                    "predicted_next_year_value": prediction,
                    "rank": rank,
                    "lime_feature_rule": feature_rule,
                    "lime_local_weight": float(weight),
                    "direction": "pushes prediction up" if weight > 0 else "pushes prediction down",
                }
            )

        html_blocks.append(f"<h1>{target_label}: {state_value}, source year {year_value}</h1>")
        html_blocks.append(exp.as_html())

    explanation_df = pd.DataFrame(all_records)
    explanation_df.to_csv(output_prefix.with_suffix(".csv"), index=False)
    output_prefix.with_suffix(".html").write_text("\n<hr/>\n".join(html_blocks), encoding="utf-8")
    return explanation_df


def make_writeup(
    output_dir: Path,
    price_metrics: dict[str, float],
    usage_metrics: dict[str, float],
    price_explanations: pd.DataFrame,
    usage_explanations: pd.DataFrame,
) -> None:
    """Create a markdown paragraph students can adapt into the technical writeup."""

    def top_rules(df: pd.DataFrame, state: str) -> str:
        subset = df[df["state"].astype(str).str.upper().str.contains(state.upper(), regex=False)]
        if subset.empty:
            subset = df.head(3)
        subset = subset.sort_values(["state", "rank"]).head(3)
        return "; ".join(
            f"{r.lime_feature_rule} ({r.direction}, weight={r.lime_local_weight:.3f})"
            for r in subset.itertuples()
        )

    md = f"""# Interpretability Standard: LIME Local Explanations

## What this script adds
This script adds a model-interpretability layer to the oil and gas forecasting project. After training the next-year price model and the next-year petroleum-usage model, it applies LIME to explain the latest forecasted row for each state. LIME treats the trained model as a black box, creates small perturbations around one state-year observation, asks the model for predictions on those nearby points, and fits a simpler local surrogate model to approximate the model's behavior near that one forecast.

## Why this fits the interpretability standard
The interpretability lecture emphasized that a prediction alone is not enough when a model affects decisions; we also need a way to communicate which inputs mattered and how those inputs moved the output. In our project, the raw forecast says what the model expects for future prices and petroleum usage. The LIME layer says why a given California or New York forecast was high or low according to the model. This is useful for the writeup because it converts a black-box forecasting pipeline into an auditable explanation: for each forecast, we can identify the largest local contributors and whether they pushed the prediction upward or downward.

## How to read the LIME output
Each row of the CSV is one local explanation term. The `lime_feature_rule` column gives the local condition LIME used, such as a lagged price or consumption value falling in a certain range. The `lime_local_weight` column gives the direction and size of that term in the local surrogate explanation. Positive values push the forecast up; negative values push it down. These weights should not be read as causal effects. They are local explanations of this model's behavior around a specific state-year example.

## Price model explanation summary
In-sample diagnostics for the model explained here: MAE = {price_metrics['mae']:.3f}, R² = {price_metrics['r2']:.3f}. For the latest explained price forecasts, representative top rules include: {top_rules(price_explanations, 'CA')}; {top_rules(price_explanations, 'NY')}.

## Usage model explanation summary
In-sample diagnostics for the model explained here: MAE = {usage_metrics['mae']:.3f}, R² = {usage_metrics['r2']:.3f}. For the latest explained petroleum-usage forecasts, representative top rules include: {top_rules(usage_explanations, 'CA')}; {top_rules(usage_explanations, 'NY')}.

## Limitations to state clearly
LIME is local, not global. It explains one forecast at a time, not the entire model. The explanation can also change if the training panel changes, if different features are added, or if the perturbation neighborhood changes. Since this project currently only compares California and New York, the LIME results should be presented as a transparent explanation of this prototype model rather than as a general theory of oil usage across all states.
"""
    (output_dir / "interpretability_writeup.md").write_text(md, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LIME explanations for oil/gas forecasting models.")
    parser.add_argument("--num-features", type=int, default=10, help="Number of LIME features per explanation.")
    args = parser.parse_args()

    root = find_project_root()
    output_dir = root / "outputs" / "interpretability"
    output_dir.mkdir(parents=True, exist_ok=True)

    panel = load_panel(root)
    price_target = pick_first_existing(panel.columns, PRICE_TARGET_CANDIDATES, "price")
    usage_target = pick_first_existing(panel.columns, USAGE_TARGET_CANDIDATES, "usage")

    print(f"Using price target column: {price_target}")
    print(f"Using usage target column: {usage_target}")

    X_price, y_price, _, price_rows = make_features_and_target(panel, price_target)
    X_usage, y_usage, _, usage_rows = make_features_and_target(panel, usage_target)

    price_model = load_or_train_model(root, "price", X_price, y_price)
    usage_model = load_or_train_model(root, "usage", X_usage, y_usage)

    # Fit loaded fallback pipelines if necessary. Saved sklearn models should already be fitted.
    try:
        price_model.predict(X_price.head(1))
    except Exception:
        price_model.fit(X_price, y_price)

    try:
        usage_model.predict(X_usage.head(1))
    except Exception:
        usage_model.fit(X_usage, y_usage)

    price_metrics = evaluate_model(price_model, X_price, y_price)
    usage_metrics = evaluate_model(usage_model, X_usage, y_usage)

    price_explanations = explain_with_lime(
        model=price_model,
        X=X_price,
        y=y_price,
        original_rows=price_rows,
        target_label="next_year_price",
        output_prefix=output_dir / "lime_price_explanations",
        num_features=args.num_features,
    )

    usage_explanations = explain_with_lime(
        model=usage_model,
        X=X_usage,
        y=y_usage,
        original_rows=usage_rows,
        target_label="next_year_petroleum_usage",
        output_prefix=output_dir / "lime_usage_explanations",
        num_features=args.num_features,
    )

    make_writeup(output_dir, price_metrics, usage_metrics, price_explanations, usage_explanations)

    print("LIME interpretability outputs written to:")
    print(f"  {output_dir / 'lime_price_explanations.csv'}")
    print(f"  {output_dir / 'lime_usage_explanations.csv'}")
    print(f"  {output_dir / 'lime_price_explanations.html'}")
    print(f"  {output_dir / 'lime_usage_explanations.html'}")
    print(f"  {output_dir / 'interpretability_writeup.md'}")


if __name__ == "__main__":
    main()
