"""
RouteWise: Supply Chain Delivery Analytics & Prediction
Exploratory Data Analysis (EDA) Module (Person 1 - Data Engineering & EDA)

Generates all 9 required exploratory visualizations with professional styling,
units, titles, and saves figures to 'outputs/figures/'. Also generates OLAP
summaries saved to 'outputs/tables/'.
"""

from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# Set consistent visualization theme
sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams.update({
    "figure.autolayout": True,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300
})


def plot_delivery_time_distribution(df: pd.DataFrame, output_dir: Path) -> Path:
    """Plot 1: Distribution of delivery times with KDE and summary statistics."""
    fig, ax = plt.subplots(figsize=(9, 5))
    mean_val = df["delivery_time_hours"].mean()
    median_val = df["delivery_time_hours"].median()

    sns.histplot(
        df["delivery_time_hours"],
        kde=True,
        color="#2b5c8f",
        bins=35,
        ax=ax,
        edgecolor="white",
        alpha=0.75
    )
    ax.axvline(mean_val, color="#d95f02", linestyle="--", linewidth=2, label=f"Mean: {mean_val:.2f} hrs")
    ax.axvline(median_val, color="#7570b3", linestyle="-.", linewidth=2, label=f"Median: {median_val:.2f} hrs")

    ax.set_title("Distribution of Delivery Times (Transit Duration)")
    ax.set_xlabel("Delivery Time (Hours)")
    ax.set_ylabel("Frequency (Parcel Count)")
    ax.legend(frameon=True, loc="upper right")

    save_path = output_dir / "01_delivery_time_distribution.png"
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    return save_path


def plot_avg_delivery_time_by_carrier(df: pd.DataFrame, output_dir: Path) -> Path:
    """Plot 2: Average delivery time and 95% CI by logistics carrier."""
    fig, ax = plt.subplots(figsize=(8, 5))
    carrier_order = df.groupby("carrier")["delivery_time_hours"].mean().sort_values().index

    palette = sns.color_palette("Blues_d", len(carrier_order))
    sns.barplot(
        data=df,
        x="carrier",
        y="delivery_time_hours",
        order=carrier_order,
        palette=palette,
        errorbar=("ci", 95),
        capsize=0.1,
        ax=ax
    )

    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f"{height:.2f} h",
                    xy=(p.get_x() + p.get_width() / 2, height / 2),
                    xytext=(0, 0),
                    textcoords="offset points",
                    ha="center", va="center",
                    color="white" if height > 2.0 else "black",
                    fontweight="bold")

    ax.set_title("Average Delivery Time by Logistics Carrier (with 95% CI)")
    ax.set_xlabel("Logistics Carrier")
    ax.set_ylabel("Mean Delivery Time (Hours)")

    save_path = output_dir / "02_avg_delivery_time_by_carrier.png"
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    return save_path


def plot_avg_delivery_time_by_route(df: pd.DataFrame, output_dir: Path) -> Path:
    """Plot 3: Average delivery time by transit route corridor."""
    fig, ax = plt.subplots(figsize=(9, 5))
    route_order = df.groupby("route")["delivery_time_hours"].mean().sort_values().index

    palette = sns.color_palette("mako", len(route_order))
    sns.boxplot(
        data=df,
        x="route",
        y="delivery_time_hours",
        order=route_order,
        palette=palette,
        showmeans=True,
        meanprops={"marker": "D", "markerfacecolor": "red", "markeredgecolor": "red"},
        ax=ax
    )

    ax.set_title("Delivery Duration Distribution Across Transit Routes")
    ax.set_xlabel("Transit Route Corridor")
    ax.set_ylabel("Delivery Time (Hours)")

    save_path = output_dir / "03_avg_delivery_time_by_route.png"
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    return save_path


def plot_delivery_time_by_weather(df: pd.DataFrame, output_dir: Path) -> Path:
    """Plot 4: Delivery time by weather condition."""
    fig, ax = plt.subplots(figsize=(9, 5))
    weather_order = ["Clear", "Rainy", "Foggy", "Snowy", "Stormy"]

    palette = ["#e6ab02", "#1f78b4", "#7570b3", "#a6cee3", "#e31a1c"]
    sns.violinplot(
        data=df,
        x="weather",
        y="delivery_time_hours",
        order=weather_order,
        palette=palette,
        inner="quartile",
        cut=0,
        ax=ax
    )

    ax.set_title("Delivery Duration by Prevailing Weather Condition")
    ax.set_xlabel("Atmospheric Weather Condition")
    ax.set_ylabel("Delivery Time (Hours)")

    save_path = output_dir / "04_delivery_time_by_weather.png"
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    return save_path


def plot_parcel_weight_vs_delivery_time(df: pd.DataFrame, output_dir: Path) -> Path:
    """Plot 5: Scatter plot with regression trend of parcel weight vs delivery time."""
    fig, ax = plt.subplots(figsize=(9, 5))

    sns.regplot(
        data=df.sample(n=min(2500, len(df)), random_state=42),
        x="parcel_weight_kg",
        y="delivery_time_hours",
        scatter_kws={"alpha": 0.25, "color": "#1b9e77", "s": 18},
        line_kws={"color": "#d95f02", "linewidth": 2.5},
        ax=ax
    )

    ax.set_title("Parcel Weight vs. Delivery Time (Sampled Points & Linear Fit)")
    ax.set_xlabel("Parcel Weight (Kilograms)")
    ax.set_ylabel("Delivery Time (Hours)")

    save_path = output_dir / "05_parcel_weight_vs_delivery_time.png"
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    return save_path


def plot_distance_vs_delivery_time(df: pd.DataFrame, output_dir: Path) -> Path:
    """Plot 6: Transit distance versus delivery time colored by traffic level."""
    fig, ax = plt.subplots(figsize=(10, 6))

    traffic_order = ["Low", "Medium", "High", "Jam"]
    traffic_palette = {"Low": "#2ca02c", "Medium": "#1f77b4", "High": "#ff7f0e", "Jam": "#d62728"}

    sns.scatterplot(
        data=df,
        x="distance_km",
        y="delivery_time_hours",
        hue="traffic_level",
        hue_order=traffic_order,
        palette=traffic_palette,
        alpha=0.5,
        s=25,
        ax=ax
    )

    ax.set_title("Transit Distance vs. Delivery Duration by Traffic Intensity")
    ax.set_xlabel("Delivery Distance (Kilometres)")
    ax.set_ylabel("Delivery Time (Hours)")
    ax.legend(title="Traffic Level", loc="upper left", frameon=True)

    save_path = output_dir / "06_distance_vs_delivery_time.png"
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    return save_path


def plot_traffic_level_vs_delivery_time(df: pd.DataFrame, output_dir: Path) -> Path:
    """Plot 7: Traffic level versus delivery time with box & point summary."""
    fig, ax = plt.subplots(figsize=(8, 5))
    traffic_order = ["Low", "Medium", "High", "Jam"]
    palette = ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728"]

    sns.boxplot(
        data=df,
        x="traffic_level",
        y="delivery_time_hours",
        order=traffic_order,
        palette=palette,
        showmeans=True,
        meanprops={"marker": "o", "markerfacecolor": "black", "markeredgecolor": "black"},
        ax=ax
    )

    ax.set_title("Impact of Traffic Congestion Levels on Delivery Duration")
    ax.set_xlabel("Traffic Level Tier")
    ax.set_ylabel("Delivery Time (Hours)")

    save_path = output_dir / "07_traffic_level_vs_delivery_time.png"
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    return save_path


def plot_correlation_heatmap(df: pd.DataFrame, output_dir: Path) -> Path:
    """Plot 8: Correlation heatmap for numerical features."""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Numerical columns
    num_cols = ["distance_km", "parcel_weight_kg", "delivery_time_hours", "delayed"]
    corr_matrix = df[num_cols].corr()

    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        fmt=".3f",
        linewidths=1,
        square=True,
        cbar_kws={"shrink": 0.8},
        ax=ax
    )

    ax.set_title("Correlation Heatmap: Quantitative Delivery Metrics")

    save_path = output_dir / "08_correlation_heatmap.png"
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    return save_path


def plot_delivery_trends_over_time(df: pd.DataFrame, output_dir: Path) -> Path:
    """Plot 9: Monthly and weekly delivery performance trends over time."""
    fig, ax = plt.subplots(figsize=(11, 5))

    temp_df = df.copy()
    temp_df["order_date_dt"] = pd.to_datetime(temp_df["order_date"])
    weekly_perf = temp_df.groupby(pd.Grouper(key="order_date_dt", freq="W-MON")).agg(
        mean_duration=("delivery_time_hours", "mean"),
        delay_rate=("delayed", lambda x: np.mean(x) * 100)
    ).reset_index()

    ax.plot(weekly_perf["order_date_dt"], weekly_perf["mean_duration"],
            color="#1f77b4", marker="o", linewidth=2.5, label="Mean Delivery Time (Hrs)")

    ax2 = ax.twinx()
    ax2.plot(weekly_perf["order_date_dt"], weekly_perf["delay_rate"],
             color="#d62728", marker="s", linestyle="--", linewidth=2, label="Delay Rate (%)")

    ax.set_title("Weekly Supply Chain Delivery Performance & Delay Trends (Jan - Jun 2024)")
    ax.set_xlabel("Calendar Week")
    ax.set_ylabel("Mean Delivery Duration (Hours)", color="#1f77b4")
    ax2.set_ylabel("SLA Delay Rate (%)", color="#d62728")
    ax.grid(True, linestyle=":", alpha=0.6)

    # Combined legend
    lines_1, labels_1 = ax.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right", frameon=True)

    save_path = output_dir / "09_delivery_trends_over_time.png"
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    return save_path


def generate_olap_summaries(df: pd.DataFrame, output_dir: Path) -> dict[str, pd.DataFrame]:
    """
    Generate all 8 required OLAP-style operational summaries and save to CSV.
    """
    tables = {}

    # OLAP 1: Average delivery time by carrier
    olap_1 = df.groupby("carrier").agg(
        total_deliveries=("delivery_id", "count"),
        mean_hours=("delivery_time_hours", "mean"),
        median_hours=("delivery_time_hours", "median"),
        std_hours=("delivery_time_hours", "std"),
        delay_pct=("delayed", lambda x: np.round(np.mean(x) * 100, 2))
    ).round(2).reset_index()
    tables["olap_1_carrier_performance.csv"] = olap_1

    # OLAP 2: Average delivery time by route
    olap_2 = df.groupby("route").agg(
        total_deliveries=("delivery_id", "count"),
        mean_distance_km=("distance_km", "mean"),
        mean_hours=("delivery_time_hours", "mean"),
        delay_pct=("delayed", lambda x: np.round(np.mean(x) * 100, 2))
    ).round(2).reset_index()
    tables["olap_2_route_performance.csv"] = olap_2

    # OLAP 3: Delivery counts by weather and carrier (cross-tabulation)
    olap_3 = pd.crosstab(
        df["carrier"],
        df["weather"],
        margins=True,
        margins_name="Total Deliveries"
    )
    tables["olap_3_weather_by_carrier_counts.csv"] = olap_3

    # OLAP 4: Delay percentage by route (ranked)
    olap_4 = df.groupby("route").agg(
        total_shipments=("delivery_id", "count"),
        delayed_shipments=("delayed", "sum"),
        delay_percentage=("delayed", lambda x: np.round(np.mean(x) * 100, 2))
    ).sort_values("delay_percentage", ascending=False).reset_index()
    olap_4["delay_rank"] = olap_4["delay_percentage"].rank(ascending=False, method="dense").astype(int)
    tables["olap_4_delay_percentage_by_route.csv"] = olap_4

    # OLAP 5: Carrier performance across different routes (Pivot Table)
    olap_5 = df.pivot_table(
        index="route",
        columns="carrier",
        values="delivery_time_hours",
        aggfunc="mean"
    ).round(2)
    tables["olap_5_carrier_by_route_matrix.csv"] = olap_5

    # OLAP 6: Monthly delivery performance
    temp_df = df.copy()
    temp_df["month"] = pd.to_datetime(temp_df["order_date"]).dt.to_period("M").astype(str)
    olap_6 = temp_df.groupby("month").agg(
        total_orders=("delivery_id", "count"),
        mean_delivery_hours=("delivery_time_hours", "mean"),
        total_delays=("delayed", "sum"),
        delay_rate_pct=("delayed", lambda x: np.round(np.mean(x) * 100, 2))
    ).reset_index()
    tables["olap_6_monthly_delivery_performance.csv"] = olap_6

    # OLAP 7: Route-versus-weather pivot table (mean delivery hours)
    olap_7 = df.pivot_table(
        index="route",
        columns="weather",
        values="delivery_time_hours",
        aggfunc="mean"
    ).round(2)
    tables["olap_7_route_vs_weather_matrix.csv"] = olap_7

    # OLAP 8: Average delivery time by parcel-weight category
    weight_bins = [0.0, 3.0, 8.0, 15.0, 50.0]
    weight_labels = ["Light (<3kg)", "Standard (3-8kg)", "Heavy (8-15kg)", "Freight (>15kg)"]
    temp_df["weight_category"] = pd.cut(temp_df["parcel_weight_kg"], bins=weight_bins, labels=weight_labels)
    olap_8 = temp_df.groupby("weight_category", observed=False).agg(
        parcel_count=("delivery_id", "count"),
        mean_weight_kg=("parcel_weight_kg", "mean"),
        mean_delivery_hours=("delivery_time_hours", "mean"),
        delay_rate_pct=("delayed", lambda x: np.round(np.mean(x) * 100, 2))
    ).round(2).reset_index()
    tables["olap_8_weight_category_summary.csv"] = olap_8

    # Save all tables
    for filename, tbl in tables.items():
        tbl.to_csv(output_dir / filename)

    return tables


def main():
    parser = argparse.ArgumentParser(description="Generate EDA figures and OLAP tables.")
    parser.add_argument("--input", type=str, default="data/processed/cleaned_delivery_data.csv")
    parser.add_argument("--fig_dir", type=str, default="outputs/figures")
    parser.add_argument("--tbl_dir", type=str, default="outputs/tables")
    args = parser.parse_args()

    clean_path = Path(args.input)
    fig_dir = Path(args.fig_dir)
    tbl_dir = Path(args.tbl_dir)

    fig_dir.mkdir(parents=True, exist_ok=True)
    tbl_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading cleaned dataset from {clean_path}...")
    df = pd.read_csv(clean_path)

    print("Generating 9 required EDA visualizations...")
    plot_delivery_time_distribution(df, fig_dir)
    plot_avg_delivery_time_by_carrier(df, fig_dir)
    plot_avg_delivery_time_by_route(df, fig_dir)
    plot_delivery_time_by_weather(df, fig_dir)
    plot_parcel_weight_vs_delivery_time(df, fig_dir)
    plot_distance_vs_delivery_time(df, fig_dir)
    plot_traffic_level_vs_delivery_time(df, fig_dir)
    plot_correlation_heatmap(df, fig_dir)
    plot_delivery_trends_over_time(df, fig_dir)
    print(f"All 9 visualizations generated and saved to {fig_dir.resolve()}")

    print("\nGenerating 8 OLAP-style operational performance summaries...")
    tables = generate_olap_summaries(df, tbl_dir)
    for name in tables.keys():
        print(f"- Saved {name}")

    print("\nEDA and OLAP generation successfully completed.")


if __name__ == "__main__":
    main()
