from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ev_dashboard.pipeline import (
    BoundsConfig,
    build_recommendation_summary,
    build_station_capacity_map,
    build_time_series_summary,
    compare_strategies,
    load_data,
    predict_outcomes,
    prepare_features,
    run_weight_sensitivity_analysis,
    train_predictive_models,
)


DATA_PATH = Path(__file__).parent / "EV_Charging_Grid_Optimization_Categorical.csv"
PRIMARY = "#1f7a8c"
ACCENT = "#2a9d8f"
WARN = "#f4a261"
ALERT = "#e76f51"


st.set_page_config(
    page_title="EV Charging Grid Optimization Dashboard",
    page_icon="⚡",
    layout="wide",
)


st.markdown(
    f"""
    <style>
    .stApp {{
        background: linear-gradient(180deg, #f7fafc 0%, #edf2f7 100%);
    }}
    .hero {{
        padding: 1.1rem 1.3rem;
        border-radius: 18px;
        background: linear-gradient(120deg, {PRIMARY} 0%, {ACCENT} 100%);
        color: white;
        margin-bottom: 1rem;
        box-shadow: 0 12px 30px rgba(31, 122, 140, 0.18);
    }}
    .hero h1 {{
        margin: 0;
        font-size: 2rem;
    }}
    .hero p {{
        margin: 0.35rem 0 0;
        font-size: 1rem;
        max-width: 950px;
    }}
    .section-card {{
        background: rgba(255,255,255,0.92);
        border: 1px solid rgba(15, 23, 42, 0.08);
        border-radius: 18px;
        padding: 1rem 1rem 0.5rem;
        box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def get_base_data() -> pd.DataFrame:
    return load_data(DATA_PATH)


@st.cache_data(show_spinner=False)
def get_prepared_data() -> tuple[pd.DataFrame, dict]:
    return prepare_features(get_base_data())


@st.cache_resource(show_spinner=True)
def get_trained_assets() -> dict:
    df, metadata = get_prepared_data()
    trained = train_predictive_models(df, metadata)
    enriched = predict_outcomes(df, metadata, trained["models"])
    capacity_map = build_station_capacity_map(enriched)
    sensitivity = run_weight_sensitivity_analysis(
        enriched,
        metadata,
        trained["models"],
        capacity_map,
        BoundsConfig(),
    )
    optimized = sensitivity["scenario_results"]["default"]
    comparison = compare_strategies(
        optimized,
        metadata,
        trained["models"],
        capacity_map,
        BoundsConfig(),
    )
    time_series = build_time_series_summary(optimized)
    return {
        "df": optimized,
        "metadata": metadata,
        "models": trained["models"],
        "metrics": trained["metrics"],
        "capacity_map": capacity_map,
        "comparison": comparison,
        "time_series": time_series,
        "sensitivity_summary": sensitivity["sensitivity_summary"],
        "overall_stability_label": sensitivity["overall_stability_label"],
        "overall_interpretation": sensitivity["overall_interpretation"],
        "scenario_results": sensitivity["scenario_results"],
    }


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    timestamps = sorted(df["timestamp"].drop_duplicates())
    stations = ["All"] + sorted(df["station_id"].astype(str).unique().tolist())
    locations = ["All"] + sorted(df["location"].unique().tolist())
    charging_types = ["All"] + sorted(df["charging_type"].unique().tolist())

    c1, c2, c3, c4, c5, c6 = st.columns([1.4, 1, 1, 1, 1.2, 0.9])
    selected_timestamp = c1.selectbox("Timestamp", timestamps, index=min(12, len(timestamps) - 1))
    selected_station = c2.selectbox("Station", stations)
    selected_location = c3.selectbox("Location", locations)
    selected_type = c4.selectbox("Charging Type", charging_types)
    scenario = c5.radio(
        "Scenario",
        ["Observed Baseline", "Dataset Optimized", "Explicit Optimizer"],
        horizontal=False,
    )
    c6.button("Run Optimization", type="primary", use_container_width=True)

    filtered = df[df["timestamp"] == selected_timestamp].copy()
    if selected_station != "All":
        filtered = filtered[filtered["station_id"].astype(str) == selected_station]
    if selected_location != "All":
        filtered = filtered[filtered["location"] == selected_location]
    if selected_type != "All":
        filtered = filtered[filtered["charging_type"] == selected_type]

    power_column_map = {
        "Observed Baseline": "charging_power",
        "Dataset Optimized": "optimized_charging_power",
        "Explicit Optimizer": "explicit_optimized_power",
    }
    return filtered, scenario, power_column_map[scenario]


def render_kpis(filtered: pd.DataFrame, power_column: str) -> None:
    baseline_power = filtered["charging_power"].sum()
    allocated_power = filtered[power_column].sum()
    demand = filtered["predicted_power_demand"].sum()

    scale = allocated_power / baseline_power if baseline_power else 1.0
    predicted_loss = filtered["power_loss"].mean() * scale
    predicted_voltage = filtered["voltage_fluctuation"].mean() * scale
    predicted_stability = min(
        max(filtered["grid_stability_score"].mean() + (1 - scale) * 0.08, 0.0),
        1.0,
    )

    cols = st.columns(5)
    cols[0].metric("Total Charging Demand", f"{demand:,.2f} kW")
    cols[1].metric("Allocated Charging Power", f"{allocated_power:,.2f} kW", f"{allocated_power - baseline_power:+.2f} kW")
    cols[2].metric("Predicted Power Loss", f"{predicted_loss:.2f}", f"{predicted_loss - filtered['power_loss'].mean():+.2f}")
    cols[3].metric(
        "Voltage Fluctuation",
        f"{predicted_voltage:.2f}",
        f"{predicted_voltage - filtered['voltage_fluctuation'].mean():+.2f}",
    )
    cols[4].metric(
        "Grid Stability Score",
        f"{predicted_stability:.3f}",
        f"{predicted_stability - filtered['grid_stability_score'].mean():+.3f}",
    )


def render_session_table(filtered: pd.DataFrame) -> None:
    table = filtered[
        [
            "ev_id",
            "station_id",
            "location",
            "charging_type",
            "predicted_power_demand",
            "charging_power",
            "optimized_charging_power",
            "explicit_optimized_power",
            "recommended_action",
        ]
    ].rename(
        columns={
            "ev_id": "EV ID",
            "station_id": "Station",
            "location": "Location",
            "charging_type": "Charging Type",
            "predicted_power_demand": "Predicted Demand",
            "charging_power": "Observed Power",
            "optimized_charging_power": "Dataset Optimized",
            "explicit_optimized_power": "Explicit Optimizer",
            "recommended_action": "Status",
        }
    )
    st.dataframe(
        table.sort_values(["Station", "EV ID"]),
        use_container_width=True,
        hide_index=True,
    )


def build_session_overview_message(filtered: pd.DataFrame) -> str:
    if filtered.empty:
        return "No charging sessions match the current filters, so there is no active recommendation for this view."

    increase_rows = filtered[filtered["power_adjustment"] > 0.25]
    reduce_rows = filtered[filtered["power_adjustment"] < -0.25]
    within_rows = filtered[
        (filtered["power_adjustment"] >= -0.25) & (filtered["power_adjustment"] <= 0.25)
    ]

    total_sessions = len(filtered)
    increase_count = len(increase_rows)
    reduce_count = len(reduce_rows)
    within_count = len(within_rows)

    if increase_count == 0 and reduce_count == 0:
        return (
            f"For the {total_sessions} session(s) currently shown, keep charging close to the current levels. "
            "The optimizer does not see a meaningful benefit from shifting power in this filtered view."
        )

    parts = [
        f"For the {total_sessions} session(s) currently shown, the recommended action is to"
    ]
    if reduce_count:
        parts.append(f"reduce power for {reduce_count} session(s)")
    if increase_count:
        connector = "and" if reduce_count else ""
        parts.append(f"{connector} increase power for {increase_count} session(s)".strip())
    if within_count:
        parts.append(f"while leaving {within_count} session(s) close to their current charging level")

    station_changes = (
        filtered.groupby("station_id")["power_adjustment"].sum().sort_values()
    )
    lowest_station = station_changes.index[0]
    highest_station = station_changes.index[-1]

    message = " ".join(parts) + "."
    if station_changes.loc[lowest_station] < -0.25:
        message += f" The strongest reduction is at station {lowest_station}."
    if station_changes.loc[highest_station] > 0.25:
        message += f" The strongest increase is at station {highest_station}."
    return message


def render_grid_panel(filtered: pd.DataFrame, power_column: str) -> None:
    baseline_mean = filtered["charging_power"].mean()
    selected_mean = filtered[power_column].mean()
    scale = selected_mean / baseline_mean if baseline_mean else 1.0

    g1, g2 = st.columns([1, 1.3])
    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=float(
                min(
                    max(filtered["grid_stability_score"].mean() + (1 - scale) * 0.08, 0.0),
                    1.0,
                )
            ),
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Grid Stability"},
            gauge={
                "axis": {"range": [0, 1]},
                "bar": {"color": ACCENT},
                "steps": [
                    {"range": [0, 0.4], "color": "#fee2e2"},
                    {"range": [0.4, 0.7], "color": "#fef3c7"},
                    {"range": [0.7, 1.0], "color": "#dcfce7"},
                ],
            },
        )
    )
    gauge.update_layout(height=290, margin=dict(l=20, r=20, t=50, b=20))
    g1.plotly_chart(gauge, use_container_width=True)

    comparison = pd.DataFrame(
        {
            "Metric": ["Power Loss", "Voltage Fluctuation"],
            "Baseline": [
                filtered["power_loss"].mean(),
                filtered["voltage_fluctuation"].mean(),
            ],
            "Selected": [
                filtered["power_loss"].mean() * scale,
                filtered["voltage_fluctuation"].mean() * scale,
            ],
        }
    ).melt(id_vars="Metric", var_name="Scenario", value_name="Value")
    fig = px.bar(
        comparison,
        x="Metric",
        y="Value",
        color="Scenario",
        barmode="group",
        color_discrete_map={"Baseline": PRIMARY, "Selected": ACCENT},
    )
    fig.update_layout(height=290, margin=dict(l=20, r=20, t=30, b=10))
    g2.plotly_chart(fig, use_container_width=True)


def render_recommendations(filtered: pd.DataFrame, capacity_map: dict) -> None:
    st.subheader("Recommended Charging Actions")
    for line in build_recommendation_summary(filtered, capacity_map):
        st.markdown(f"- {line}")


def render_analytics(assets: dict) -> None:
    tabs = st.tabs(
        [
            "Strategy Comparison",
            "Demand & Capacity Insights",
            "Model Insights",
            "Sensitivity Analysis",
        ]
    )

    with tabs[0]:
        comparison = assets["comparison"]
        st.dataframe(
            comparison.round(3),
            use_container_width=True,
            hide_index=True,
        )

        series = assets["time_series"]
        fig = px.line(
            series,
            x="timestamp",
            y="allocated_power",
            color="strategy",
            title="Allocated Power Over Time",
            color_discrete_sequence=[PRIMARY, WARN, ACCENT],
        )
        st.plotly_chart(fig, use_container_width=True)

    with tabs[1]:
        df = assets["df"]
        station_summary = (
            df.groupby("station_id", as_index=False)[
                ["charging_power", "optimized_charging_power", "explicit_optimized_power"]
            ]
            .mean()
            .melt(id_vars="station_id", var_name="strategy", value_name="avg_power")
        )
        fig = px.bar(
            station_summary,
            x="station_id",
            y="avg_power",
            color="strategy",
            barmode="group",
            title="Average Allocated Power by Station",
        )
        st.plotly_chart(fig, use_container_width=True)

        heatmap = (
            df.groupby(["hour", "station_id"], as_index=False)["predicted_power_demand"]
            .mean()
            .pivot(index="hour", columns="station_id", values="predicted_power_demand")
        )
        fig = px.imshow(
            heatmap,
            aspect="auto",
            color_continuous_scale="YlGnBu",
            title="Average Predicted Demand by Hour and Station",
        )
        st.plotly_chart(fig, use_container_width=True)

    with tabs[2]:
        metrics_display = assets["metrics"].copy()
        for column in ["mae", "rmse", "r2"]:
            metrics_display[column] = metrics_display[column].round(3)
        st.dataframe(
            metrics_display,
            use_container_width=True,
            hide_index=True,
        )

        feature_story = pd.DataFrame(
            {
                "Insight": [
                    "Charging power is the direct decision variable for all optimization models.",
                    "Predicted demand anchors minimum service levels during optimization.",
                    "Station capacity limits are estimated from historical 95th percentile observed loads.",
                ]
            }
        )
        st.dataframe(feature_story, use_container_width=True, hide_index=True)

    with tabs[3]:
        sensitivity = assets["sensitivity_summary"].copy()
        numeric_cols = [
            "power_loss_weight",
            "voltage_fluctuation_weight",
            "grid_stability_weight",
            "action_flip_rate",
            "direction_flip_rate",
            "mean_absolute_power_shift",
            "max_absolute_power_shift",
            "station_rank_change",
            "avg_power_loss_delta",
            "avg_voltage_fluctuation_delta",
            "avg_grid_stability_score_delta",
            "feasibility_rate",
        ]
        for column in numeric_cols:
            sensitivity[column] = sensitivity[column].round(3)

        st.markdown(
            f"""
            **Overall stability:** `{assets["overall_stability_label"]}`  
            {assets["overall_interpretation"]}
            """
        )
        st.caption(
            "Scenarios are marked Stable when action flip rate is below 10%, mean absolute power shift "
            "is below 0.75 kW, and feasibility remains perfect."
        )
        st.dataframe(sensitivity, use_container_width=True, hide_index=True)

        bar_left, bar_right = st.columns(2)
        flip_fig = px.bar(
            sensitivity,
            x="scenario_name",
            y="action_flip_rate",
            color="stability_label",
            title="Action Flip Rate by Weight Scenario",
            color_discrete_map={"Stable": ACCENT, "Sensitive": ALERT},
        )
        flip_fig.update_layout(xaxis_title="Scenario", yaxis_title="Action Flip Rate")
        bar_left.plotly_chart(flip_fig, use_container_width=True)

        shift_fig = px.bar(
            sensitivity,
            x="scenario_name",
            y="mean_absolute_power_shift",
            color="stability_label",
            title="Mean Absolute Power Shift vs Default",
            color_discrete_map={"Stable": ACCENT, "Sensitive": ALERT},
        )
        shift_fig.update_layout(
            xaxis_title="Scenario",
            yaxis_title="Mean Absolute Power Shift (kW)",
        )
        bar_right.plotly_chart(shift_fig, use_container_width=True)


def main() -> None:
    assets = get_trained_assets()
    st.markdown(
        """
        <div class="hero">
            <h1>EV Charging Grid Optimization Dashboard</h1>
            <p>
                Explore observed charging behavior, compare it against the dataset's optimized policy,
                and review an explicit optimization recommendation that balances power loss,
                voltage fluctuation, and grid stability under operational constraints.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    filtered, scenario, power_column = apply_filters(assets["df"])
    render_kpis(filtered, power_column)
    st.markdown("</div>", unsafe_allow_html=True)

    left, right = st.columns([1.8, 1.2])
    with left:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Station and Session Overview")
        st.caption(build_session_overview_message(filtered))
        render_session_table(filtered)
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader(f"Grid Condition Panel: {scenario}")
        render_grid_panel(filtered, power_column)
        render_recommendations(filtered, assets["capacity_map"])
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    render_analytics(assets)
    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
