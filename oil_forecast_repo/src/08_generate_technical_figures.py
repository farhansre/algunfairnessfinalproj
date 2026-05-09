"""
08_generate_technical_figures.py

Purpose
-------
Generate matplotlib figures for the technical results section of the oil / gas
forecasting project writeup.

This script is intentionally self-contained: it can read the raw EIA-style CSVs
from the project `data/raw/` folder OR directly from `/mnt/data/` if you are
running inside the ChatGPT workspace.

Figures generated
-----------------
1. annual_retail_gasoline_price_with_forecast.png
2. annual_petroleum_usage_with_forecast.png
3. price_vs_petroleum_usage_scatter.png
4. year_over_year_changes.png
5. state_comparison_2025_forecast.png
6. feature_correlation_heatmap.png
7. petroleum_product_mix_latest_year.png

How to run
----------
From the repository root:

    python src/08_generate_technical_figures.py

Outputs are written to:

    outputs/figures/

Dependencies
------------
    pandas
    numpy
    matplotlib

No seaborn is used.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Path configuration
# -----------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR_CANDIDATES = [REPO_ROOT / "data" / "raw", Path("/mnt/data")]
OUTPUT_DIR = REPO_ROOT / "outputs" / "figures"
FORECAST_PATH = REPO_ROOT / "outputs" / "forecast_summary.csv"

STATE_FILES = {
    "CA": {
        "retail_price": "retailgasolineprices_ca.csv",
        "petroleum": "petroleumconsumption_ca.csv",
        "total_energy": "totalenergyconsumption_ca.csv",
        "gas_supply": "futurenaturalgassuply_ca.csv",  # uploaded filename has this spelling
    },
    "NY": {
        "retail_price": "retailgasolineprices_ny.csv",
        "petroleum": "petroleumconsumption_ny.csv",
        "total_energy": "totalenergyconsumption_ny.csv",
        "gas_supply": "futurenaturalgassupply_ny.csv",
    },
}

PETROLEUM_LABELS = {
    "DFTCP": "Distillate fuel oil",
    "HLTCP": "Hydrocarbon gas liquids",
    "JFTCP": "Jet fuel",
    "MGTCP": "Motor gasoline",
    "RFTCP": "Residual fuel oil",
    "P1TCP": "Total petroleum",
}

TOTAL_ENERGY_LABELS = {
    "CLTCB": "Coal",
    "NNTCB": "Natural gas",
    "NUETB": "Nuclear",
    "PMTCB": "Petroleum",
    "RETCB": "Renewables",
    "ELISB": "Electricity imports",
    "ELNIB": "Electricity net imports",
}


def find_raw_file(filename: str) -> Path:
    """Return the first matching raw CSV path from known raw-data locations."""
    for folder in RAW_DIR_CANDIDATES:
        candidate = folder / filename
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find {filename}. Put it in data/raw/ or run inside /mnt/data."
    )


# -----------------------------------------------------------------------------
# EIA CSV parsing helpers
# -----------------------------------------------------------------------------

def _extract_series_code(series_key: str) -> str:
    """Extract the MSN-like code from EIA keys such as SEDS.P1TCP.CA.A."""
    match = re.search(r"SEDS\.([A-Z0-9]+)\.", str(series_key))
    if match:
        return match.group(1)
    match = re.search(r"PET\.([A-Z0-9_]+)\.", str(series_key))
    if match:
        return match.group(1)
    match = re.search(r"NG\.([A-Z0-9_]+)\.", str(series_key))
    if match:
        return match.group(1)
    return str(series_key)


def parse_stacked_eia_wide(path: Path, state: str, value_name: str) -> pd.DataFrame:
    """
    Parse EIA files where each time series is stored as a repeated pair of
    columns: [metadata_or_year, value].

    This format appears in the petroleum, total-energy, and gas-supply files.
    """
    raw = pd.read_csv(path)
    frames = []

    for left_col, right_col in zip(raw.columns[0::2], raw.columns[1::2]):
        series_key = raw.loc[0, right_col] if len(raw) > 0 else right_col
        series_code = _extract_series_code(series_key)

        block = raw[[left_col, right_col]].copy()
        block.columns = ["year", value_name]
        block = block[pd.to_numeric(block["year"], errors="coerce").notna()]
        block["year"] = block["year"].astype(int)
        block[value_name] = pd.to_numeric(block[value_name], errors="coerce")
        block = block.dropna(subset=[value_name])
        block["state"] = state
        block["series_code"] = series_code
        frames.append(block[["state", "year", "series_code", value_name]])

    if not frames:
        return pd.DataFrame(columns=["state", "year", "series_code", value_name])
    return pd.concat(frames, ignore_index=True)


def parse_retail_price_weekly(path: Path, state: str) -> pd.DataFrame:
    """
    Parse weekly retail gasoline prices and aggregate to annual values.

    The state-level series is selected by looking for the state code in the EIA
    series key. If multiple series exist, the state-specific series is preferred
    over the regional comparison series.
    """
    raw = pd.read_csv(path)
    candidate_blocks = []

    for left_col, right_col in zip(raw.columns[0::2], raw.columns[1::2]):
        series_key = str(raw.loc[0, right_col])
        score = 1 if f"S{state}" in series_key else 0
        block = raw[[left_col, right_col]].copy()
        block.columns = ["date", "retail_price"]
        block = block[pd.to_numeric(block["date"], errors="coerce").notna()]
        block["date"] = block["date"].astype(str)
        block["retail_price"] = pd.to_numeric(block["retail_price"], errors="coerce")
        block = block.dropna(subset=["retail_price"])
        candidate_blocks.append((score, series_key, block))

    if not candidate_blocks:
        return pd.DataFrame(columns=["state", "year", "retail_price_mean", "retail_price_median"])

    # Prefer the state-specific series, then default to the first available block.
    candidate_blocks.sort(key=lambda item: item[0], reverse=True)
    _, _, chosen = candidate_blocks[0]
    chosen["date"] = pd.to_datetime(chosen["date"], format="%Y%m%d", errors="coerce")
    chosen = chosen.dropna(subset=["date"])
    chosen["year"] = chosen["date"].dt.year

    annual = (
        chosen.groupby("year", as_index=False)
        .agg(retail_price_mean=("retail_price", "mean"), retail_price_median=("retail_price", "median"))
    )
    annual["state"] = state
    return annual[["state", "year", "retail_price_mean", "retail_price_median"]]


def build_analysis_panel() -> pd.DataFrame:
    """Build a tidy state-year panel from raw CSVs for plotting."""
    price_frames = []
    petroleum_frames = []
    total_energy_frames = []
    gas_supply_frames = []

    for state, files in STATE_FILES.items():
        price_frames.append(parse_retail_price_weekly(find_raw_file(files["retail_price"]), state))
        petroleum_frames.append(parse_stacked_eia_wide(find_raw_file(files["petroleum"]), state, "petroleum_value"))
        total_energy_frames.append(parse_stacked_eia_wide(find_raw_file(files["total_energy"]), state, "energy_value"))
        gas_supply_frames.append(parse_stacked_eia_wide(find_raw_file(files["gas_supply"]), state, "gas_supply_bcf"))

    prices = pd.concat(price_frames, ignore_index=True)
    petroleum_long = pd.concat(petroleum_frames, ignore_index=True)
    energy_long = pd.concat(total_energy_frames, ignore_index=True)
    gas_supply_long = pd.concat(gas_supply_frames, ignore_index=True)

    petroleum = petroleum_long.pivot_table(
        index=["state", "year"], columns="series_code", values="petroleum_value", aggfunc="first"
    ).reset_index()
    petroleum.columns.name = None

    energy = energy_long.pivot_table(
        index=["state", "year"], columns="series_code", values="energy_value", aggfunc="first"
    ).reset_index()
    energy.columns.name = None

    gas_supply = (
        gas_supply_long.groupby(["state", "year"], as_index=False)["gas_supply_bcf"].mean()
    )

    panel = prices.merge(petroleum, on=["state", "year"], how="outer")
    panel = panel.merge(energy, on=["state", "year"], how="outer")
    panel = panel.merge(gas_supply, on=["state", "year"], how="outer")
    panel = panel.sort_values(["state", "year"]).reset_index(drop=True)

    # Derived features used only for graphs.
    for col in ["retail_price_mean", "P1TCP", "MGTCP", "PMTCB", "NNTCB"]:
        if col in panel.columns:
            panel[f"{col}_yoy_pct"] = panel.groupby("state")[col].pct_change() * 100

    return panel


# -----------------------------------------------------------------------------
# Plotting helpers
# -----------------------------------------------------------------------------

def savefig(filename: str) -> None:
    """Save current matplotlib figure with tight layout and close it."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=200, bbox_inches="tight")
    plt.close()


def maybe_load_forecast() -> Optional[pd.DataFrame]:
    if FORECAST_PATH.exists():
        return pd.read_csv(FORECAST_PATH)
    return None


def plot_annual_prices(panel: pd.DataFrame, forecast: Optional[pd.DataFrame]) -> None:
    plt.figure(figsize=(9, 5))
    for state, group in panel.dropna(subset=["retail_price_mean"]).groupby("state"):
        plt.plot(group["year"], group["retail_price_mean"], marker="o", label=f"{state} historical")

    if forecast is not None:
        for _, row in forecast.iterrows():
            plt.scatter(row["year"], row["predicted_next_year_price"], s=80, marker="X", label=f"{row['state']} 2025 forecast")

    plt.title("Annual retail gasoline price with 2025 forecast")
    plt.xlabel("Year")
    plt.ylabel("Annual average retail gasoline price ($/gallon)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    savefig("annual_retail_gasoline_price_with_forecast.png")


def plot_petroleum_usage(panel: pd.DataFrame, forecast: Optional[pd.DataFrame]) -> None:
    plt.figure(figsize=(9, 5))
    for state, group in panel.dropna(subset=["P1TCP"]).groupby("state"):
        plt.plot(group["year"], group["P1TCP"], marker="o", label=f"{state} historical")

    if forecast is not None:
        for _, row in forecast.iterrows():
            plt.scatter(row["year"], row["predicted_next_year_usage"], s=80, marker="X", label=f"{row['state']} 2025 forecast")

    plt.title("Total petroleum consumption with 2025 forecast")
    plt.xlabel("Year")
    plt.ylabel("Total petroleum consumption (thousand barrels)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    savefig("annual_petroleum_usage_with_forecast.png")


def plot_price_vs_usage(panel: pd.DataFrame) -> None:
    data = panel.dropna(subset=["retail_price_mean", "P1TCP"])
    plt.figure(figsize=(8, 5))
    for state, group in data.groupby("state"):
        plt.scatter(group["retail_price_mean"], group["P1TCP"], label=state, alpha=0.8)

    # Overall simple trendline for visual evidence, not causal inference.
    if len(data) >= 2:
        x = data["retail_price_mean"].to_numpy()
        y = data["P1TCP"].to_numpy()
        slope, intercept = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 100)
        plt.plot(x_line, slope * x_line + intercept, linestyle="--", label="overall trend")

    plt.title("Relationship between gasoline price and petroleum usage")
    plt.xlabel("Annual average retail gasoline price ($/gallon)")
    plt.ylabel("Total petroleum consumption (thousand barrels)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    savefig("price_vs_petroleum_usage_scatter.png")


def plot_yoy_changes(panel: pd.DataFrame) -> None:
    data = panel.dropna(subset=["retail_price_mean_yoy_pct", "P1TCP_yoy_pct"])
    plt.figure(figsize=(9, 5))
    for state, group in data.groupby("state"):
        plt.plot(group["year"], group["retail_price_mean_yoy_pct"], marker="o", label=f"{state} price YoY %")
        plt.plot(group["year"], group["P1TCP_yoy_pct"], marker="s", linestyle="--", label=f"{state} usage YoY %")

    plt.axhline(0, linewidth=1)
    plt.title("Year-over-year changes in price and petroleum usage")
    plt.xlabel("Year")
    plt.ylabel("Year-over-year change (%)")
    plt.legend(ncol=2)
    plt.grid(True, alpha=0.3)
    savefig("year_over_year_changes.png")


def plot_2025_forecast_bars(forecast: Optional[pd.DataFrame]) -> None:
    if forecast is None or forecast.empty:
        return

    states = forecast["state"].tolist()
    x = np.arange(len(states))
    width = 0.35

    price = forecast["predicted_next_year_price"].to_numpy()
    usage_scaled = forecast["predicted_next_year_usage"].to_numpy() / 1000.0

    plt.figure(figsize=(8, 5))
    plt.bar(x - width / 2, price, width, label="Predicted price ($/gal)")
    plt.bar(x + width / 2, usage_scaled, width, label="Predicted usage (million barrels)")
    plt.xticks(x, states)
    plt.title("2025 forecast comparison by state")
    plt.xlabel("State")
    plt.ylabel("Forecast value")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    savefig("state_comparison_2025_forecast.png")


def plot_feature_correlation_heatmap(panel: pd.DataFrame) -> None:
    candidate_cols = [
        "retail_price_mean",
        "P1TCP",
        "MGTCP",
        "DFTCP",
        "JFTCP",
        "PMTCB",
        "NNTCB",
        "RETCB",
        "gas_supply_bcf",
    ]
    cols = [col for col in candidate_cols if col in panel.columns]
    data = panel[cols].dropna(how="all")
    corr = data.corr(numeric_only=True)

    plt.figure(figsize=(8, 6))
    plt.imshow(corr, aspect="auto")
    plt.colorbar(label="Correlation")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    plt.yticks(range(len(corr.index)), corr.index)
    plt.title("Correlation among modeling variables")

    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            value = corr.iloc[i, j]
            if pd.notna(value):
                plt.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8)

    savefig("feature_correlation_heatmap.png")


def plot_petroleum_mix_latest_year(panel: pd.DataFrame) -> None:
    product_cols = [col for col in PETROLEUM_LABELS if col in panel.columns and col != "P1TCP"]
    latest_year = int(panel.dropna(subset=product_cols, how="all")["year"].max())
    data = panel[panel["year"] == latest_year].set_index("state")

    states = sorted(data.index.unique())
    x = np.arange(len(states))
    bottom = np.zeros(len(states))

    plt.figure(figsize=(9, 5))
    for col in product_cols:
        values = data.loc[states, col].fillna(0).to_numpy()
        plt.bar(x, values, bottom=bottom, label=PETROLEUM_LABELS.get(col, col))
        bottom += values

    plt.xticks(x, states)
    plt.title(f"Petroleum consumption mix by state ({latest_year})")
    plt.xlabel("State")
    plt.ylabel("Consumption (thousand barrels)")
    plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    plt.grid(axis="y", alpha=0.3)
    savefig("petroleum_product_mix_latest_year.png")


def write_figure_index() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    index_path = OUTPUT_DIR / "FIGURE_INDEX.md"
    index_path.write_text(
        "# Technical Results Figure Index\n\n"
        "Use these figures in the technical results section of the writeup.\n\n"
        "1. `annual_retail_gasoline_price_with_forecast.png` — shows historical CA/NY annual gasoline prices plus the 2025 price forecast.\n"
        "2. `annual_petroleum_usage_with_forecast.png` — shows historical total petroleum usage plus the 2025 usage forecast.\n"
        "3. `price_vs_petroleum_usage_scatter.png` — visualizes the empirical relationship between price and usage; describe this as associational, not causal.\n"
        "4. `year_over_year_changes.png` — compares price volatility against usage volatility.\n"
        "5. `state_comparison_2025_forecast.png` — compact comparison of the final forecast values for California and New York.\n"
        "6. `feature_correlation_heatmap.png` — documents which model variables move together historically.\n"
        "7. `petroleum_product_mix_latest_year.png` — decomposes petroleum usage by product category in the latest available year.\n\n"
        "Suggested writeup framing: These plots support the technical results section by showing (i) the historical signal used by the models, "
        "(ii) the actual forecast outputs, and (iii) diagnostic evidence about variable relationships and state-level differences.\n",
        encoding="utf-8",
    )


def main() -> None:
    print("Building analysis panel from raw CSVs...")
    panel = build_analysis_panel()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    panel.to_csv(OUTPUT_DIR / "analysis_panel_used_for_figures.csv", index=False)

    forecast = maybe_load_forecast()
    if forecast is None:
        print("No forecast_summary.csv found; forecast-point graphs will omit forecast markers.")

    print("Generating matplotlib figures...")
    plot_annual_prices(panel, forecast)
    plot_petroleum_usage(panel, forecast)
    plot_price_vs_usage(panel)
    plot_yoy_changes(panel)
    plot_2025_forecast_bars(forecast)
    plot_feature_correlation_heatmap(panel)
    plot_petroleum_mix_latest_year(panel)
    write_figure_index()

    print(f"Done. Figures saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
