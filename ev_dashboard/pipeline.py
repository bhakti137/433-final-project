from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


REGRESSION_TARGETS = [
    "optimized_charging_power",
    "power_loss",
    "voltage_fluctuation",
    "grid_stability_score",
]


DISPLAY_LABELS = {
    "charging_power": "Observed Baseline",
    "optimized_charging_power": "Dataset Optimized",
    "explicit_optimized_power": "Explicit Optimizer",
}


@dataclass(frozen=True)
class OptimizationWeights:
    scenario_name: str = "default"
    power_loss: float = 0.45
    voltage_fluctuation: float = 0.35
    grid_stability: float = 0.20


@dataclass(frozen=True)
class BoundsConfig:
    min_demand_fraction: float = 0.80
    max_demand_fraction: float = 1.20


def load_data(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values(["timestamp", "station_id", "ev_id"]).reset_index(drop=True)


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    prepared = df.copy()
    prepared["hour"] = prepared["timestamp"].dt.hour
    prepared["day_of_week"] = prepared["timestamp"].dt.dayofweek
    prepared["month"] = prepared["timestamp"].dt.month
    prepared["station_id"] = prepared["station_id"].astype(str)
    prepared["ev_id"] = prepared["ev_id"].astype(str)
    prepared["num_chargers"] = prepared["num_chargers"].astype(int)

    base_features = [
        "station_id",
        "location",
        "charging_type",
        "num_chargers",
        "voltage_level",
        "current_flow",
        "power_consumed",
        "battery_capacity",
        "charging_time",
        "charging_power",
        "charging_cost",
        "predicted_power_demand",
        "hour",
        "day_of_week",
        "month",
    ]
    categorical_features = ["station_id", "location", "charging_type"]
    numeric_features = [col for col in base_features if col not in categorical_features]
    metadata = {
        "base_features": base_features,
        "categorical_features": categorical_features,
        "numeric_features": numeric_features,
        "targets": REGRESSION_TARGETS,
    }
    return prepared, metadata


def _build_preprocessor(metadata: Dict[str, List[str]]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                metadata["numeric_features"],
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                metadata["categorical_features"],
            ),
        ]
    )


def train_predictive_models(
    df: pd.DataFrame,
    metadata: Dict[str, List[str]],
    test_size: float = 0.20,
    random_state: int = 42,
) -> Dict[str, object]:
    models: Dict[str, Pipeline] = {}
    metrics: List[Dict[str, float]] = []
    preprocessor = _build_preprocessor(metadata)
    X = df[metadata["base_features"]]

    for target in REGRESSION_TARGETS:
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        pipeline = Pipeline(
            steps=[("preprocessor", preprocessor), ("regressor", LinearRegression())]
        )
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        metrics.append(
            {
                "target": target,
                "model_name": "LinearRegression",
                "mae": mean_absolute_error(y_test, preds),
                "rmse": np.sqrt(mean_squared_error(y_test, preds)),
                "r2": r2_score(y_test, preds),
            }
        )
        models[target] = pipeline

    return {"models": models, "metrics": pd.DataFrame(metrics)}


def predict_outcomes(
    df: pd.DataFrame, metadata: Dict[str, List[str]], models: Dict[str, Pipeline]
) -> pd.DataFrame:
    predicted = df.copy()
    features = predicted[metadata["base_features"]]
    for target, prefix in [
        ("optimized_charging_power", "pred_optimized_charging_power"),
        ("power_loss", "pred_power_loss"),
        ("voltage_fluctuation", "pred_voltage_fluctuation"),
        ("grid_stability_score", "pred_grid_stability_score"),
    ]:
        predicted[prefix] = models[target].predict(features)
    return predicted


def build_station_capacity_map(df: pd.DataFrame) -> Dict[str, float]:
    station_totals = (
        df.groupby(["timestamp", "station_id"], as_index=False)["charging_power"].sum()
        .groupby("station_id")["charging_power"]
        .quantile(0.95)
    )
    return {str(key): float(value) for key, value in station_totals.items()}


def _coefficient_for_feature(model: Pipeline, feature_name: str) -> float:
    preprocessor: ColumnTransformer = model.named_steps["preprocessor"]
    regressor: LinearRegression = model.named_steps["regressor"]
    feature_names = preprocessor.get_feature_names_out()
    target_name = f"num__{feature_name}"
    if target_name not in feature_names:
        raise KeyError(f"Feature {feature_name} not found in model features.")
    idx = list(feature_names).index(target_name)
    return float(regressor.coef_[idx])


def _bounds_for_group(
    group_df: pd.DataFrame,
    capacity_map: Dict[str, float],
    bounds_config: BoundsConfig,
) -> List[Tuple[float, float]]:
    bounds: List[Tuple[float, float]] = []
    for _, row in group_df.iterrows():
        demand = float(row["predicted_power_demand"])
        lower = max(0.0, bounds_config.min_demand_fraction * demand)
        upper = bounds_config.max_demand_fraction * demand
        session_cap = max(float(row["charging_power"]) * 1.5, demand * 1.3)
        station_cap = capacity_map.get(str(row["station_id"]), session_cap)
        upper = min(upper, session_cap, station_cap)
        if upper < lower:
            upper = lower
        bounds.append((lower, upper))
    return bounds


def optimize_timestamp_allocation(
    group_df: pd.DataFrame,
    models: Dict[str, Pipeline],
    capacity_map: Dict[str, float],
    weights: OptimizationWeights | None = None,
    bounds_config: BoundsConfig | None = None,
) -> pd.DataFrame:
    weights = weights or OptimizationWeights()
    bounds_config = bounds_config or BoundsConfig()
    working = group_df.copy().reset_index(drop=True)
    n = len(working)
    if n == 0:
        return working

    loss_coef = _coefficient_for_feature(models["power_loss"], "charging_power")
    volt_coef = _coefficient_for_feature(models["voltage_fluctuation"], "charging_power")
    grid_coef = _coefficient_for_feature(models["grid_stability_score"], "charging_power")

    objective = np.repeat(
        weights.power_loss * loss_coef
        + weights.voltage_fluctuation * volt_coef
        - weights.grid_stability * grid_coef,
        n,
    )

    demand_total = float(working["predicted_power_demand"].sum())
    bounds = _bounds_for_group(working, capacity_map, bounds_config)
    lower_sum = sum(low for low, _ in bounds)
    upper_sum = sum(high for _, high in bounds)
    clipped_demand_total = min(max(demand_total, lower_sum), upper_sum)

    A_eq = np.ones((1, n))
    b_eq = np.array([clipped_demand_total])

    A_ub = []
    b_ub = []
    for station_id, idxs in working.groupby("station_id").groups.items():
        row = np.zeros(n)
        row[list(idxs)] = 1.0
        A_ub.append(row)
        b_ub.append(capacity_map.get(str(station_id), float(np.sum([b[1] for b in bounds]))))

    result = linprog(
        c=objective,
        A_ub=np.array(A_ub) if A_ub else None,
        b_ub=np.array(b_ub) if b_ub else None,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )

    if result.success:
        allocations = result.x
        feasible = True
        objective_value = float(result.fun)
    else:
        allocations = np.array([float(row["predicted_power_demand"]) for _, row in working.iterrows()])
        feasible = False
        objective_value = np.nan

    working["explicit_optimized_power"] = allocations
    working["optimization_feasible"] = feasible
    working["optimization_objective"] = objective_value
    working["power_adjustment"] = working["explicit_optimized_power"] - working["charging_power"]
    working["recommended_action"] = np.where(
        working["power_adjustment"] > 0.25,
        "Increase power",
        np.where(working["power_adjustment"] < -0.25, "Reduce power", "Within target"),
    )
    return working


def optimize_all_timestamps(
    df: pd.DataFrame,
    models: Dict[str, Pipeline],
    capacity_map: Dict[str, float],
    weights: OptimizationWeights | None = None,
    bounds_config: BoundsConfig | None = None,
) -> pd.DataFrame:
    optimized_groups = []
    for _, group in df.groupby("timestamp", sort=True):
        optimized_groups.append(
            optimize_timestamp_allocation(
                group,
                models=models,
                capacity_map=capacity_map,
                weights=weights,
                bounds_config=bounds_config,
            )
        )
    return pd.concat(optimized_groups, ignore_index=True)


def generate_weight_scenarios() -> List[OptimizationWeights]:
    raw_scenarios = [
        ("default", 0.45, 0.35, 0.20),
        ("loss_plus_small", 0.50, 0.30, 0.20),
        ("voltage_plus_small", 0.40, 0.40, 0.20),
        ("stability_plus_small", 0.40, 0.30, 0.30),
        ("loss_minus_small", 0.40, 0.40, 0.20),
        ("voltage_minus_small", 0.50, 0.25, 0.25),
        ("stability_minus_small", 0.50, 0.35, 0.15),
        ("balanced_equal", 0.34, 0.33, 0.33),
        ("loss_heavy", 0.60, 0.25, 0.15),
        ("voltage_heavy", 0.25, 0.60, 0.15),
        ("stability_heavy", 0.25, 0.25, 0.50),
    ]
    scenarios: List[OptimizationWeights] = []
    for name, loss, voltage, stability in raw_scenarios:
        total = loss + voltage + stability
        scenarios.append(
            OptimizationWeights(
                scenario_name=name,
                power_loss=loss / total,
                voltage_fluctuation=voltage / total,
                grid_stability=stability / total,
            )
        )
    return scenarios


def _predict_strategy_outcomes(
    df: pd.DataFrame,
    metadata: Dict[str, List[str]],
    models: Dict[str, Pipeline],
    power_column: str,
) -> pd.DataFrame:
    evaluation_frame = df.copy()
    evaluation_frame["charging_power"] = evaluation_frame[power_column]
    features = evaluation_frame[metadata["base_features"]]
    evaluation_frame["eval_power_loss"] = models["power_loss"].predict(features)
    evaluation_frame["eval_voltage_fluctuation"] = models["voltage_fluctuation"].predict(features)
    evaluation_frame["eval_grid_stability_score"] = np.clip(
        models["grid_stability_score"].predict(features), 0.0, 1.0
    )
    return evaluation_frame


def _predict_custom_power_outcomes(
    df: pd.DataFrame,
    metadata: Dict[str, List[str]],
    models: Dict[str, Pipeline],
    custom_power_column: str,
) -> pd.DataFrame:
    evaluation_frame = df.copy()
    evaluation_frame["charging_power"] = evaluation_frame[custom_power_column]
    features = evaluation_frame[metadata["base_features"]]
    evaluation_frame["eval_power_loss"] = models["power_loss"].predict(features)
    evaluation_frame["eval_voltage_fluctuation"] = models["voltage_fluctuation"].predict(features)
    evaluation_frame["eval_grid_stability_score"] = np.clip(
        models["grid_stability_score"].predict(features), 0.0, 1.0
    )
    return evaluation_frame


def evaluate_strategy(
    df: pd.DataFrame,
    power_column: str,
    metadata: Dict[str, List[str]],
    models: Dict[str, Pipeline],
    capacity_map: Dict[str, float],
    bounds_config: BoundsConfig | None = None,
) -> Dict[str, float]:
    bounds_config = bounds_config or BoundsConfig()
    evaluated = _predict_strategy_outcomes(df, metadata, models, power_column)
    lower = bounds_config.min_demand_fraction * evaluated["predicted_power_demand"]
    upper = bounds_config.max_demand_fraction * evaluated["predicted_power_demand"]
    evaluated["within_session_bounds"] = evaluated[power_column].between(lower, upper)
    station_loads = (
        evaluated.groupby(["timestamp", "station_id"], as_index=False)[power_column].sum()
    )
    station_loads["station_capacity"] = station_loads["station_id"].astype(str).map(capacity_map)
    capacity_ok = (station_loads[power_column] <= station_loads["station_capacity"] + 1e-6).mean()

    return {
        "strategy": DISPLAY_LABELS[power_column],
        "avg_allocated_power": float(evaluated[power_column].mean()),
        "total_allocated_power": float(evaluated[power_column].sum()),
        "avg_power_loss": float(evaluated["eval_power_loss"].mean()),
        "avg_voltage_fluctuation": float(evaluated["eval_voltage_fluctuation"].mean()),
        "avg_grid_stability_score": float(evaluated["eval_grid_stability_score"].mean()),
        "demand_satisfaction_rate": float(
            (evaluated[power_column] >= evaluated["predicted_power_demand"] * bounds_config.min_demand_fraction).mean()
        ),
        "session_bounds_rate": float(evaluated["within_session_bounds"].mean()),
        "station_capacity_rate": float(capacity_ok),
    }


def compare_strategies(
    df: pd.DataFrame,
    metadata: Dict[str, List[str]],
    models: Dict[str, Pipeline],
    capacity_map: Dict[str, float],
    bounds_config: BoundsConfig | None = None,
) -> pd.DataFrame:
    bounds_config = bounds_config or BoundsConfig()
    rows = [
        evaluate_strategy(df, "charging_power", metadata, models, capacity_map, bounds_config),
        evaluate_strategy(
            df, "optimized_charging_power", metadata, models, capacity_map, bounds_config
        ),
        evaluate_strategy(
            df, "explicit_optimized_power", metadata, models, capacity_map, bounds_config
        ),
    ]
    comparison = pd.DataFrame(rows)
    return comparison.sort_values("avg_grid_stability_score", ascending=False).reset_index(drop=True)


def build_time_series_summary(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for power_col in ["charging_power", "optimized_charging_power", "explicit_optimized_power"]:
        grouped = df.groupby("timestamp").agg(
            allocated_power=(power_col, "sum"),
            power_loss=("power_loss", "mean"),
            voltage_fluctuation=("voltage_fluctuation", "mean"),
            grid_stability_score=("grid_stability_score", "mean"),
        )
        grouped = grouped.reset_index()
        grouped["strategy"] = DISPLAY_LABELS[power_col]
        records.append(grouped)
    return pd.concat(records, ignore_index=True)


def build_recommendation_summary(
    filtered_df: pd.DataFrame,
    capacity_map: Dict[str, float],
) -> List[str]:
    if filtered_df.empty:
        return ["No charging sessions match the current filters."]

    recommendations: List[str] = []
    top_reductions = filtered_df.nsmallest(3, "power_adjustment")
    top_increases = filtered_df.nlargest(3, "power_adjustment")

    for _, row in top_reductions.iterrows():
        if row["power_adjustment"] < -0.25:
            recommendations.append(
                f"Reduce EV {row['ev_id']} at station {row['station_id']} by {abs(row['power_adjustment']):.2f} kW."
            )
    for _, row in top_increases.iterrows():
        if row["power_adjustment"] > 0.25:
            recommendations.append(
                f"Increase EV {row['ev_id']} at station {row['station_id']} by {row['power_adjustment']:.2f} kW."
            )

    station_loads = filtered_df.groupby("station_id")["explicit_optimized_power"].sum()
    for station_id, load in station_loads.items():
        capacity = capacity_map.get(str(station_id))
        if capacity and load >= 0.95 * capacity:
            recommendations.append(
                f"Station {station_id} is operating near its estimated capacity ({load:.2f} / {capacity:.2f} kW)."
            )

    if not recommendations:
        recommendations.append("Current allocations are already close to the optimizer recommendations.")
    return recommendations[:8]


def _station_rank_shift(default_df: pd.DataFrame, scenario_df: pd.DataFrame) -> float:
    default_totals = default_df.groupby("station_id")["explicit_optimized_power"].sum()
    scenario_totals = scenario_df.groupby("station_id")["explicit_optimized_power"].sum()
    stations = sorted(set(default_totals.index).union(set(scenario_totals.index)), key=str)
    default_ranks = default_totals.reindex(stations, fill_value=0.0).rank(ascending=False, method="average")
    scenario_ranks = scenario_totals.reindex(stations, fill_value=0.0).rank(ascending=False, method="average")
    return float((default_ranks - scenario_ranks).abs().mean())


def _classify_sensitivity_row(row: pd.Series) -> str:
    if (
        row["action_flip_rate"] < 0.10
        and row["mean_absolute_power_shift"] < 0.75
        and row["feasibility_rate"] == 1.0
    ):
        return "Stable"
    return "Sensitive"


def compare_weight_scenarios(
    default_df: pd.DataFrame,
    scenario_results: Dict[str, pd.DataFrame],
    metadata: Dict[str, List[str]],
    models: Dict[str, Pipeline],
    capacity_map: Dict[str, float],
    bounds_config: BoundsConfig | None = None,
) -> pd.DataFrame:
    bounds_config = bounds_config or BoundsConfig()
    default_eval = _predict_custom_power_outcomes(
        default_df, metadata, models, "explicit_optimized_power"
    )
    default_metrics = {
        "avg_power_loss": float(default_eval["eval_power_loss"].mean()),
        "avg_voltage_fluctuation": float(default_eval["eval_voltage_fluctuation"].mean()),
        "avg_grid_stability_score": float(default_eval["eval_grid_stability_score"].mean()),
    }
    default_station_ranks = _station_rank_shift(default_df, default_df)
    rows: List[Dict[str, float | str]] = []

    for scenario_name, scenario_df in scenario_results.items():
        scenario_eval = _predict_custom_power_outcomes(
            scenario_df, metadata, models, "explicit_optimized_power"
        )
        default_actions = default_df["recommended_action"].to_numpy()
        scenario_actions = scenario_df["recommended_action"].to_numpy()
        default_direction = np.sign(default_df["power_adjustment"].to_numpy())
        scenario_direction = np.sign(scenario_df["power_adjustment"].to_numpy())
        power_shift = (
            scenario_df["explicit_optimized_power"].to_numpy()
            - default_df["explicit_optimized_power"].to_numpy()
        )
        weights = scenario_df.iloc[0][
            ["weight_power_loss", "weight_voltage_fluctuation", "weight_grid_stability"]
        ]
        rows.append(
            {
                "scenario_name": scenario_name,
                "power_loss_weight": float(weights["weight_power_loss"]),
                "voltage_fluctuation_weight": float(weights["weight_voltage_fluctuation"]),
                "grid_stability_weight": float(weights["weight_grid_stability"]),
                "action_flip_rate": float((scenario_actions != default_actions).mean()),
                "direction_flip_rate": float((scenario_direction != default_direction).mean()),
                "mean_absolute_power_shift": float(np.abs(power_shift).mean()),
                "max_absolute_power_shift": float(np.abs(power_shift).max()),
                "station_rank_change": float(_station_rank_shift(default_df, scenario_df) - default_station_ranks),
                "avg_power_loss_delta": float(
                    scenario_eval["eval_power_loss"].mean() - default_metrics["avg_power_loss"]
                ),
                "avg_voltage_fluctuation_delta": float(
                    scenario_eval["eval_voltage_fluctuation"].mean()
                    - default_metrics["avg_voltage_fluctuation"]
                ),
                "avg_grid_stability_score_delta": float(
                    scenario_eval["eval_grid_stability_score"].mean()
                    - default_metrics["avg_grid_stability_score"]
                ),
                "feasibility_rate": float(scenario_df["optimization_feasible"].mean()),
            }
        )

    summary = pd.DataFrame(rows).sort_values("scenario_name").reset_index(drop=True)
    summary["stability_label"] = summary.apply(_classify_sensitivity_row, axis=1)
    return summary


def run_weight_sensitivity_analysis(
    df: pd.DataFrame,
    metadata: Dict[str, List[str]],
    models: Dict[str, Pipeline],
    capacity_map: Dict[str, float],
    bounds_config: BoundsConfig | None = None,
) -> Dict[str, object]:
    bounds_config = bounds_config or BoundsConfig()
    scenario_results: Dict[str, pd.DataFrame] = {}
    scenarios = generate_weight_scenarios()

    for weights in scenarios:
        scenario_df = optimize_all_timestamps(
            df,
            models=models,
            capacity_map=capacity_map,
            weights=weights,
            bounds_config=bounds_config,
        ).copy()
        scenario_df["scenario_name"] = weights.scenario_name
        scenario_df["weight_power_loss"] = weights.power_loss
        scenario_df["weight_voltage_fluctuation"] = weights.voltage_fluctuation
        scenario_df["weight_grid_stability"] = weights.grid_stability
        scenario_results[weights.scenario_name] = scenario_df

    default_df = scenario_results["default"]
    sensitivity_summary = compare_weight_scenarios(
        default_df=default_df,
        scenario_results=scenario_results,
        metadata=metadata,
        models=models,
        capacity_map=capacity_map,
        bounds_config=bounds_config,
    )

    small_perturbation_names = {
        "loss_plus_small",
        "voltage_plus_small",
        "stability_plus_small",
        "loss_minus_small",
        "voltage_minus_small",
        "stability_minus_small",
    }
    sensitive_small_count = int(
        sensitivity_summary[
            sensitivity_summary["scenario_name"].isin(small_perturbation_names)
            & (sensitivity_summary["stability_label"] == "Sensitive")
        ].shape[0]
    )
    overall_label = (
        "Potentially unstable" if sensitive_small_count >= 3 else "Reasonably stable"
    )
    interpretation = (
        "Recommendations are sensitive to minor weight changes and should be interpreted cautiously."
        if overall_label == "Potentially unstable"
        else "Small weight changes do not materially alter recommendations."
    )

    return {
        "scenario_results": scenario_results,
        "sensitivity_summary": sensitivity_summary,
        "overall_stability_label": overall_label,
        "overall_interpretation": interpretation,
        "small_perturbation_sensitive_count": sensitive_small_count,
    }
