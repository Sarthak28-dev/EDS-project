"""
RouteWise: Supply Chain Delivery Analytics & Prediction
Data Generation Module (Person 1 - Data Engineering & EDA)

Generates a realistic synthetic delivery logistics dataset of ~7,000 records
with controlled data-quality imperfections, realistic domain dependencies,
and strict schema conformance.
"""

from datetime import datetime, timedelta
from pathlib import Path
import argparse
import numpy as np
import pandas as pd


# Set default random seed for full reproducibility
DEFAULT_SEED = 42
DEFAULT_NUM_RECORDS = 7150


def generate_synthetic_delivery_data(num_records: int = DEFAULT_NUM_RECORDS,
                                     random_seed: int = DEFAULT_SEED) -> pd.DataFrame:
    """
    Generate synthetic logistics delivery data with realistic physics, operational
    correlations, and intentional real-world data quality issues.

    Parameters
    ----------
    num_records : int
        Number of raw delivery records to generate (default: 7150).
    random_seed : int
        Seed for pseudo-random number generator to guarantee reproducibility.

    Returns
    -------
    pd.DataFrame
        Raw delivery DataFrame conforming to agreed RouteWise schema.
    """
    rng = np.random.default_rng(random_seed)

    # 1. Define Routes, Origins, Destinations, and Base Travel Characteristics
    routes_config = {
        "Route_Central": {
            "origin": "Warehouse_Central",
            "destination": "Zone_Metro",
            "dist_range": (15.0, 55.0),
            "base_speed_kmh": 36.0,  # Dense urban traffic zone
            "sla_buffer_hr": 0.65,
        },
        "Route_North": {
            "origin": "Warehouse_North",
            "destination": "Zone_Suburbs",
            "dist_range": (80.0, 210.0),
            "base_speed_kmh": 62.0,  # Mixed highway & suburban arterial
            "sla_buffer_hr": 0.85,
        },
        "Route_South": {
            "origin": "Warehouse_South",
            "destination": "Zone_Industrial",
            "dist_range": (60.0, 160.0),
            "base_speed_kmh": 58.0,  # Freight corridor
            "sla_buffer_hr": 0.80,
        },
        "Route_East": {
            "origin": "Warehouse_Central",
            "destination": "Zone_Rural",
            "dist_range": (150.0, 360.0),
            "base_speed_kmh": 68.0,  # Intercity rural highway
            "sla_buffer_hr": 1.10,
        },
        "Route_West": {
            "origin": "Warehouse_West",
            "destination": "Zone_Industrial",
            "dist_range": (70.0, 190.0),
            "base_speed_kmh": 55.0,  # Industrial ring road
            "sla_buffer_hr": 0.90,
        },
    }

    route_names = list(routes_config.keys())
    route_probs = [0.28, 0.22, 0.20, 0.16, 0.14]

    assigned_routes = rng.choice(route_names, size=num_records, p=route_probs)

    # 2. Assign Carriers with Realistic Market Shares & Operational Efficiencies
    # Efficiency modifier: multiplier on delivery duration (< 1.0 = faster, > 1.0 = slower)
    carriers = ["ApexDelivery", "FastShip", "NationalExpress", "QuickLogistics"]
    carrier_probs = [0.32, 0.28, 0.22, 0.18]
    carrier_multipliers = {
        "ApexDelivery": 0.98,      # Reliable premium carrier
        "FastShip": 0.94,          # High-speed courier
        "NationalExpress": 1.06,   # Lower-cost, slightly slower
        "QuickLogistics": 1.02     # Mid-tier balanced provider
    }
    assigned_carriers = rng.choice(carriers, size=num_records, p=carrier_probs)

    # 3. Weather Conditions & Environmental Delays
    weather_conditions = ["Clear", "Rainy", "Foggy", "Snowy", "Stormy"]
    weather_probs = [0.52, 0.22, 0.12, 0.08, 0.06]
    weather_multipliers = {
        "Clear": 1.00,
        "Rainy": 1.16,
        "Foggy": 1.25,
        "Snowy": 1.42,
        "Stormy": 1.62
    }
    assigned_weather = rng.choice(weather_conditions, size=num_records, p=weather_probs)

    # 4. Traffic Levels & Congestion Penalties
    traffic_levels = ["Low", "Medium", "High", "Jam"]
    traffic_probs = [0.30, 0.40, 0.22, 0.08]
    traffic_multipliers = {
        "Low": 1.00,
        "Medium": 1.18,
        "High": 1.45,
        "Jam": 1.82
    }
    assigned_traffic = rng.choice(traffic_levels, size=num_records, p=traffic_probs)

    # 5. Parcel Weights (Log-normal distribution: typical small parcels with long right tail)
    # Median weight ~ 4.5 kg, mean ~ 6.5 kg, range 0.2 kg to ~35 kg
    raw_weights = rng.lognormal(mean=1.5, sigma=0.75, size=num_records)
    raw_weights = np.clip(raw_weights, 0.2, 40.0)

    # 6. Distance Generation based on Assigned Route
    distances = np.zeros(num_records)
    origins = []
    destinations = []
    base_speeds = np.zeros(num_records)
    sla_buffers = np.zeros(num_records)

    for i, r in enumerate(assigned_routes):
        cfg = routes_config[r]
        origins.append(cfg["origin"])
        destinations.append(cfg["destination"])
        min_d, max_d = cfg["dist_range"]
        distances[i] = rng.uniform(min_d, max_d)
        base_speeds[i] = cfg["base_speed_kmh"]
        sla_buffers[i] = cfg["sla_buffer_hr"]

    # 7. Order Date & Dispatch Timestamps (spanning 6 months: Jan 1, 2024 to June 30, 2024)
    start_date = datetime(2024, 1, 1, 6, 0, 0)
    total_seconds_span = 181 * 24 * 3600  # ~181 days

    random_second_offsets = rng.integers(0, total_seconds_span, size=num_records)
    dispatch_datetimes = [start_date + timedelta(seconds=int(sec)) for sec in random_second_offsets]

    order_dates = [(dt - timedelta(hours=int(rng.integers(2, 24)))).strftime("%Y-%m-%d") for dt in dispatch_datetimes]

    # 8. Compute Ground-Truth Delivery Duration (Hours) with Controlled Non-deterministic Noise
    # Physics formula:
    # duration = (distance / base_speed) * carrier_mod * weather_mod * traffic_mod + weight_penalty + residual_noise
    delivery_durations = np.zeros(num_records)
    for i in range(num_records):
        dist = distances[i]
        speed = base_speeds[i]
        c_mod = carrier_multipliers[assigned_carriers[i]]
        w_mod = weather_multipliers[assigned_weather[i]]
        t_mod = traffic_multipliers[assigned_traffic[i]]

        # Minor loading/unloading delay for heavy parcels
        weight_penalty = 0.018 * raw_weights[i]

        # Base nominal time
        nominal_hours = (dist / speed) * c_mod * w_mod * t_mod + weight_penalty

        # Realistic stochastic variance (Gaussian noise ~ 10% standard deviation)
        noise = rng.normal(loc=0.0, scale=0.08 * nominal_hours)
        actual_hours = max(0.4, nominal_hours + noise)
        delivery_durations[i] = actual_hours

    delivery_durations = np.round(delivery_durations, 2)

    # 9. Compute Delivery Timestamps & Delay Flags
    delivery_datetimes = [
        disp + timedelta(seconds=int(dur * 3600))
        for disp, dur in zip(dispatch_datetimes, delivery_durations)
    ]

    # SLA baseline calculation: standard baseline speed on route under mild traffic
    delays = np.zeros(num_records, dtype=int)
    for i in range(num_records):
        route_cfg = routes_config[assigned_routes[i]]
        expected_base_time = (distances[i] / route_cfg["base_speed_kmh"]) * 1.15 + route_cfg["sla_buffer_hr"]
        if delivery_durations[i] > expected_base_time:
            delays[i] = 1

    # Unique Delivery Identifiers
    delivery_ids = [f"DEL-{10000 + i}" for i in range(num_records)]

    # Format Datetime Strings
    dispatch_strs = [dt.strftime("%Y-%m-%d %H:%M:%S") for dt in dispatch_datetimes]
    delivery_strs = [dt.strftime("%Y-%m-%d %H:%M:%S") for dt in delivery_datetimes]

    # Assemble Base DataFrame
    df = pd.DataFrame({
        "delivery_id": delivery_ids,
        "order_date": order_dates,
        "dispatch_time": dispatch_strs,
        "delivery_time": delivery_strs,
        "route": assigned_routes,
        "origin": origins,
        "destination": destinations,
        "carrier": assigned_carriers,
        "weather": assigned_weather,
        "parcel_weight_kg": np.round(raw_weights, 2),
        "distance_km": np.round(distances, 1),
        "traffic_level": assigned_traffic,
        "delivery_time_hours": delivery_durations,
        "delayed": delays
    })

    # 10. Inject Realistic Controlled Data-Quality Imperfections
    # a) Missing weather observations (~2.8% of records)
    missing_weather_idx = rng.choice(num_records, size=int(num_records * 0.028), replace=False)
    df.loc[missing_weather_idx, "weather"] = np.nan

    # b) Missing parcel weights (~2.5% of records)
    missing_weight_idx = rng.choice(num_records, size=int(num_records * 0.025), replace=False)
    df.loc[missing_weight_idx, "parcel_weight_kg"] = np.nan

    # c) Negative parcel weights (e.g. data-entry keystroke error, ~0.7% of records)
    negative_weight_idx = rng.choice(
        [i for i in range(num_records) if i not in missing_weight_idx],
        size=int(num_records * 0.007),
        replace=False
    )
    df.loc[negative_weight_idx, "parcel_weight_kg"] = -1.0 * np.abs(df.loc[negative_weight_idx, "parcel_weight_kg"])

    # d) Extreme erroneous parcel weight entries (e.g., missing decimal point / 2-decimal shift: 289 instead of 2.89 kg)
    extreme_weight_idx = rng.choice(
        [i for i in range(num_records) if i not in missing_weight_idx and i not in negative_weight_idx],
        size=8,
        replace=False
    )
    df.loc[extreme_weight_idx, "parcel_weight_kg"] = df.loc[extreme_weight_idx, "parcel_weight_kg"] * 100.0

    # e) Inconsistent carrier spellings & casing variations (~3.5% of records)
    carrier_typo_idx = rng.choice(num_records, size=int(num_records * 0.035), replace=False)
    carrier_variations = {
        "FastShip": ["fastship", "Fast_Ship", "FASTSHIP"],
        "ApexDelivery": ["Apex_Delivery", "apex delivery", "APEX"],
        "NationalExpress": ["Natl_Express", "National Express", "national express"],
        "QuickLogistics": ["quick logistics", "Quick_Logistics", "QUICK LOGISTICS"]
    }
    for idx in carrier_typo_idx:
        original_carrier = df.loc[idx, "carrier"]
        replacement = rng.choice(carrier_variations[original_carrier])
        df.loc[idx, "carrier"] = replacement

    # f) Invalid categorical values (~0.6% of records)
    invalid_traffic_idx = rng.choice(num_records, size=20, replace=False)
    df.loc[invalid_traffic_idx[:10], "traffic_level"] = "UNKNOWN"
    df.loc[invalid_traffic_idx[10:15], "traffic_level"] = "Extremely High"
    df.loc[invalid_traffic_idx[15:], "traffic_level"] = "med"

    invalid_weather_idx = rng.choice(
        [i for i in range(num_records) if i not in missing_weather_idx],
        size=15,
        replace=False
    )
    df.loc[invalid_weather_idx[:8], "weather"] = "Thunderstorm"
    df.loc[invalid_weather_idx[8:], "weather"] = "Sunny_Windy"

    # g) Suspicious numerical & timestamp glitches (e.g. clock desync, delivery before dispatch)
    glitch_idx = rng.choice(num_records, size=14, replace=False)
    for idx in glitch_idx[:7]:
        # Negative duration / inverted timestamps
        df.loc[idx, "delivery_time_hours"] = -0.5
        dt_disp = datetime.strptime(df.loc[idx, "dispatch_time"], "%Y-%m-%d %H:%M:%S")
        df.loc[idx, "delivery_time"] = (dt_disp - timedelta(minutes=30)).strftime("%Y-%m-%d %H:%M:%S")
    for idx in glitch_idx[7:]:
        # Unrealistic zero delivery duration
        df.loc[idx, "delivery_time_hours"] = 0.0
        df.loc[idx, "delivery_time"] = df.loc[idx, "dispatch_time"]

    # h) Duplicate records:
    # 1. Exact duplicate rows (~50 rows)
    exact_dup_idx = rng.choice(num_records, size=50, replace=False)
    exact_duplicates = df.iloc[exact_dup_idx].copy()

    # 2. Duplicate delivery_id with conflicting rows (~25 rows)
    id_dup_idx = rng.choice(num_records, size=25, replace=False)
    conflicting_duplicates = df.iloc[id_dup_idx].copy()
    conflicting_duplicates["parcel_weight_kg"] = conflicting_duplicates["parcel_weight_kg"] + 1.5

    df = pd.concat([df, exact_duplicates, conflicting_duplicates], ignore_index=True)

    # Shuffle the final dataset rows to mix in duplicates realistically
    df = df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)

    return df


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic RouteWise logistics delivery dataset.")
    parser.add_argument("--records", type=int, default=DEFAULT_NUM_RECORDS, help="Number of records to generate")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for reproducibility")
    parser.add_argument("--output", type=str, default="data/raw/delivery_data.csv", help="Target output CSV path")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating synthetic RouteWise dataset (target: {args.records} base records, seed={args.seed})...")
    df = generate_synthetic_delivery_data(num_records=args.records, random_seed=args.seed)

    df.to_csv(output_path, index=False)
    print(f"Successfully generated {len(df)} total raw records.")
    print(f"Saved raw delivery dataset to: {output_path.resolve()}")
    print("\nInitial Quality Audit Summary of Generated Raw Data:")
    print(f"- Shape: {df.shape}")
    print(f"- Null values per column:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    print(f"- Duplicate delivery_ids: {df['delivery_id'].duplicated().sum()}")
    print(f"- Negative weights: {(df['parcel_weight_kg'] < 0).sum()}")
    print(f"- Delivery durations <= 0: {(df['delivery_time_hours'] <= 0).sum()}")


if __name__ == "__main__":
    main()
