from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).parent
DATA_PATH = ROOT / "EV_Charging_Grid_Optimization_Categorical.csv"
OUT_DIR = ROOT / "assets" / "eda"


def setup_style() -> None:
    plt.style.use("ggplot")
    plt.rcParams["figure.figsize"] = (12, 7)
    plt.rcParams["axes.titlesize"] = 18
    plt.rcParams["axes.labelsize"] = 13
    plt.rcParams["xtick.labelsize"] = 11
    plt.rcParams["ytick.labelsize"] = 11


def load_df() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    return df


def save_plot(name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUT_DIR / name, dpi=180, bbox_inches="tight")
    plt.close()


def chart_avg_power_by_station(df: pd.DataFrame) -> None:
    station_summary = (
        df.groupby("station_id", as_index=False)["charging_power"]
        .mean()
        .sort_values("charging_power", ascending=False)
    )
    plt.figure()
    colors = plt.cm.YlGnBu(np.linspace(0.35, 0.85, len(station_summary)))
    ax = plt.gca()
    ax.bar(station_summary["station_id"].astype(str), station_summary["charging_power"], color=colors)
    ax.set_title("Average Charging Power by Station")
    ax.set_xlabel("Station ID")
    ax.set_ylabel("Average Charging Power (kW)")
    save_plot("avg_charging_power_by_station.png")


def chart_grid_stability_by_hour(df: pd.DataFrame) -> None:
    hourly = df.groupby("hour", as_index=False)["grid_stability_score"].mean()
    plt.figure()
    ax = plt.gca()
    ax.plot(hourly["hour"], hourly["grid_stability_score"], marker="o", color="#1f7a8c", linewidth=2.5)
    ax.set_title("Average Grid Stability Score by Hour")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Average Grid Stability Score")
    ax.set_ylim(0.4, 0.58)
    save_plot("grid_stability_by_hour.png")


def chart_demand_vs_optimized(df: pd.DataFrame) -> None:
    plt.figure()
    ax = plt.gca()
    x = df["predicted_power_demand"].to_numpy()
    y = df["optimized_charging_power"].to_numpy()
    ax.scatter(x, y, alpha=0.4, s=28, color="#2a9d8f")
    slope, intercept = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 200)
    ax.plot(xs, slope * xs + intercept, color="#e76f51", linewidth=2.5)
    ax.set_title("Predicted Demand vs Optimized Charging Power")
    ax.set_xlabel("Predicted Power Demand (kW)")
    ax.set_ylabel("Optimized Charging Power (kW)")
    save_plot("predicted_demand_vs_optimized_power.png")


def chart_correlation_heatmap(df: pd.DataFrame) -> None:
    corr_cols = [
        "current_flow",
        "power_consumed",
        "power_loss",
        "voltage_fluctuation",
        "charging_time",
        "charging_power",
        "charging_cost",
        "predicted_power_demand",
        "optimized_charging_power",
        "grid_stability_score",
    ]
    corr = df[corr_cols].corr()
    plt.figure(figsize=(12, 9))
    ax = plt.gca()
    im = ax.imshow(corr, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index)
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9)
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Correlation Heatmap of Key Numeric Variables")
    save_plot("correlation_heatmap.png")


def chart_delta_histogram(df: pd.DataFrame) -> None:
    delta = df["optimized_charging_power"] - df["charging_power"]
    plt.figure()
    ax = plt.gca()
    ax.hist(delta, bins=30, color="#f4a261", edgecolor="white", alpha=0.9)
    ax.axvline(0, color="#264653", linestyle="--", linewidth=2)
    ax.set_title("Distribution of Optimized Power Minus Observed Charging Power")
    ax.set_xlabel("Optimized Charging Power - Observed Charging Power (kW)")
    ax.set_ylabel("Count")
    save_plot("optimized_minus_observed_histogram.png")


def chart_charging_type_comparison(df: pd.DataFrame) -> None:
    summary = (
        df.groupby("charging_type", as_index=False)[
            ["charging_power", "power_loss", "voltage_fluctuation", "grid_stability_score"]
        ]
        .mean()
        .set_index("charging_type")
    )
    plt.figure(figsize=(13, 7))
    ax = summary.plot(kind="bar", ax=plt.gca(), colormap="Set2")
    ax.set_title("Average Operational Metrics by Charging Type")
    ax.set_xlabel("Charging Type")
    ax.set_ylabel("Average Value")
    ax.legend(title="Metric", bbox_to_anchor=(1.02, 1), loc="upper left")
    save_plot("charging_type_comparison.png")


def main() -> None:
    setup_style()
    df = load_df()
    chart_avg_power_by_station(df)
    chart_grid_stability_by_hour(df)
    chart_demand_vs_optimized(df)
    chart_correlation_heatmap(df)
    chart_delta_histogram(df)
    chart_charging_type_comparison(df)
    print(f"Saved charts to {OUT_DIR}")


if __name__ == "__main__":
    main()
