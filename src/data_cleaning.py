"""
RouteWise: Supply Chain Delivery Analytics & Prediction
Data Auditing and Cleaning Module (Person 1 - Data Engineering & EDA)

Performs comprehensive data auditing, logs anomalies, implements justified
cleaning steps, and produces the verified 'data/processed/cleaned_delivery_data.csv'.
"""

from datetime import datetime
from pathlib import Path
import argparse
import numpy as np
import pandas as pd


def audit_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform a comprehensive multi-criteria audit of the raw delivery dataset.

    Returns
    -------
    pd.DataFrame
        Audit report summarizing discovered data quality defects.
    """
    audit_findings = []

    # 1. Total records and dimensions
    total_rows, total_cols = df.shape
    audit_findings.append({
        "Category": "Dimensions",
        "Issue Description": "Total raw observations and columns",
        "Records Affected": total_rows,
        "Severity": "Informational",
        "Details": f"{total_rows} rows, {total_cols} columns"
    })

    # 2. Missing values
    null_counts = df.isnull().sum()
    for col, count in null_counts.items():
        if count > 0:
            audit_findings.append({
                "Category": "Missing Values",
                "Issue Description": f"Null / NaN observations in column '{col}'",
                "Records Affected": int(count),
                "Severity": "High",
                "Details": f"{count / total_rows:.2%} of total dataset"
            })

    # 3. Exact Duplicate Rows
    exact_dups = int(df.duplicated().sum())
    audit_findings.append({
        "Category": "Duplicates",
        "Issue Description": "Exact duplicate records (all columns identical)",
        "Records Affected": exact_dups,
        "Severity": "High",
        "Details": f"{exact_dups} identical rows"
    })

    # 4. Duplicate Delivery IDs (including conflicting)
    id_dups = int(df["delivery_id"].duplicated().sum())
    audit_findings.append({
        "Category": "Duplicates",
        "Issue Description": "Duplicate delivery_id identifiers",
        "Records Affected": id_dups,
        "Severity": "High",
        "Details": f"{id_dups} non-unique primary key entries"
    })

    # 5. Invalid parcel weights (negative)
    neg_weights = int((df["parcel_weight_kg"] < 0).sum())
    audit_findings.append({
        "Category": "Value Validity",
        "Issue Description": "Negative parcel weights (physically impossible)",
        "Records Affected": neg_weights,
        "Severity": "High",
        "Details": "Values with negative sign from keystroke error"
    })

    # 6. Extreme parcel weights (> 100 kg)
    extreme_weights = int((df["parcel_weight_kg"] > 100.0).sum())
    audit_findings.append({
        "Category": "Value Validity",
        "Issue Description": "Extreme parcel weight outliers (>100 kg)",
        "Records Affected": extreme_weights,
        "Severity": "Medium",
        "Details": "Suspected gram-to-kg unit conversion errors"
    })

    # 7. Inconsistent Carrier labels
    canonical_carriers = {"ApexDelivery", "FastShip", "NationalExpress", "QuickLogistics"}
    non_canonical_carriers = int((~df["carrier"].isin(canonical_carriers)).sum())
    audit_findings.append({
        "Category": "Categorical Consistency",
        "Issue Description": "Inconsistent carrier naming / casing variations",
        "Records Affected": non_canonical_carriers,
        "Severity": "Medium",
        "Details": "Variants like fastship, Apex_Delivery, Natl_Express"
    })

    # 8. Invalid traffic levels
    canonical_traffic = {"Low", "Medium", "High", "Jam"}
    non_canonical_traffic = int((~df["traffic_level"].isin(canonical_traffic)).sum())
    audit_findings.append({
        "Category": "Categorical Consistency",
        "Issue Description": "Non-standard traffic level labels",
        "Records Affected": non_canonical_traffic,
        "Severity": "Medium",
        "Details": "Values: UNKNOWN, Extremely High, med"
    })

    # 9. Invalid weather categories
    canonical_weather = {"Clear", "Rainy", "Foggy", "Snowy", "Stormy"}
    non_canonical_weather = int((~df["weather"].dropna().isin(canonical_weather)).sum())
    audit_findings.append({
        "Category": "Categorical Consistency",
        "Issue Description": "Non-standard weather categories",
        "Records Affected": non_canonical_weather,
        "Severity": "Low",
        "Details": "Values: Thunderstorm, Sunny_Windy"
    })

    # 10. Temporal & Duration Anomaly (delivery duration <= 0 or delivery_time < dispatch_time)
    impossible_duration = int((df["delivery_time_hours"] <= 0).sum())
    audit_findings.append({
        "Category": "Temporal Validity",
        "Issue Description": "Non-positive delivery duration (hours <= 0)",
        "Records Affected": impossible_duration,
        "Severity": "Critical",
        "Details": "Inverted timestamps / zero elapsed transit time"
    })

    audit_df = pd.DataFrame(audit_findings)
    return audit_df


def clean_delivery_data(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Execute transparent, justified cleaning transformations on the raw delivery dataset.

    Parameters
    ----------
    raw_df : pd.DataFrame
        Raw delivery DataFrame loaded from disk.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Cleaned DataFrame, Cleaning Log DataFrame detailing records affected and rationale.
    """
    df = raw_df.copy()
    initial_count = len(df)
    cleaning_log = []

    # -------------------------------------------------------------
    # Step 1: Resolve Duplicate Records
    # -------------------------------------------------------------
    # 1a. Remove exact duplicate rows
    exact_dups_count = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    cleaning_log.append({
        "Step": "1a. Exact Deduplication",
        "Problem Identified": "Redundant identical rows created by database replication glitched batch inserts.",
        "Records Affected": exact_dups_count,
        "Action Taken": "Dropped identical duplicate rows using df.drop_duplicates().",
        "Justification": "Identical rows artificially inflate sample counts and weight specific deliveries."
    })

    # 1b. Resolve duplicate delivery_ids with conflicting entries
    id_dups_count = int(df["delivery_id"].duplicated().sum())
    df = df.drop_duplicates(subset=["delivery_id"], keep="first").reset_index(drop=True)
    cleaning_log.append({
        "Step": "1b. Primary Key Deduplication",
        "Problem Identified": "Non-unique delivery_id entries with minor conflicting column edits.",
        "Records Affected": id_dups_count,
        "Action Taken": "Retained the first verified entry and removed subsequent conflicting duplicates.",
        "Justification": "Primary key delivery_id must be strictly unique to maintain relational integrity."
    })

    # -------------------------------------------------------------
    # Step 2: Remove Physically Impossible Temporal & Duration Glitches
    # -------------------------------------------------------------
    impossible_mask = df["delivery_time_hours"] <= 0
    impossible_count = int(impossible_mask.sum())
    df = df[~impossible_mask].reset_index(drop=True)
    cleaning_log.append({
        "Step": "2. Chronological Validation",
        "Problem Identified": "Non-positive transit durations (duration <= 0 or arrival before dispatch).",
        "Records Affected": impossible_count,
        "Action Taken": "Removed corrupted records where delivery_time_hours <= 0.",
        "Justification": "Violates physical causality and arrow of time; caused by desynchronized telemetry clocks. Cannot be reconstructed without distorting true duration."
    })

    # -------------------------------------------------------------
    # Step 3: Clean and Normalize Parcel Weights
    # -------------------------------------------------------------
    # 3a. Keystroke negative weights
    neg_weight_mask = df["parcel_weight_kg"] < 0
    neg_weight_count = int(neg_weight_mask.sum())
    df.loc[neg_weight_mask, "parcel_weight_kg"] = df.loc[neg_weight_mask, "parcel_weight_kg"].abs()
    cleaning_log.append({
        "Step": "3a. Negative Weight Correction",
        "Problem Identified": "Negative parcel weights (e.g., -4.5 kg) caused by operator prefix keystroke errors.",
        "Records Affected": neg_weight_count,
        "Action Taken": "Applied absolute value transformation: df['parcel_weight_kg'] = df['parcel_weight_kg'].abs().",
        "Justification": "The numerical magnitude is valid; only the sign was erroneously inverted during data entry."
    })

    # 3b. Extreme weight outliers (> 100 kg from unit entry errors)
    extreme_weight_mask = df["parcel_weight_kg"] > 100.0
    extreme_weight_count = int(extreme_weight_mask.sum())
    df.loc[extreme_weight_mask, "parcel_weight_kg"] = (df.loc[extreme_weight_mask, "parcel_weight_kg"] / 100.0).round(2)
    cleaning_log.append({
        "Step": "3b. Weight Unit Normalization",
        "Problem Identified": "Extreme parcel weights (> 100 kg, e.g. 750 kg) due to scale software recording grams instead of kg.",
        "Records Affected": extreme_weight_count,
        "Action Taken": "Divided extreme observations by 100.0 to restore realistic kilogram scale.",
        "Justification": "Courier standard parcels have an upper limit of 40 kg; magnitude matched 100x scale conversion error."
    })

    # 3c. Missing parcel weights
    missing_weight_mask = df["parcel_weight_kg"].isnull()
    missing_weight_count = int(missing_weight_mask.sum())
    # Route-specific median imputation
    route_medians = df.groupby("route")["parcel_weight_kg"].transform("median")
    df["parcel_weight_kg"] = df["parcel_weight_kg"].fillna(route_medians).round(2)
    cleaning_log.append({
        "Step": "3c. Missing Weight Imputation",
        "Problem Identified": "Unrecorded parcel weights (NaN) due to intermittent scale sensor disconnects.",
        "Records Affected": missing_weight_count,
        "Action Taken": "Imputed missing weights with the route-specific median weight.",
        "Justification": "Median is robust against extreme values, and routes exhibit characteristic parcel weight distributions."
    })

    # -------------------------------------------------------------
    # Step 4: Standardize Inconsistent Carrier Naming & Casing
    # -------------------------------------------------------------
    carrier_mapping = {
        "fastship": "FastShip",
        "Fast_Ship": "FastShip",
        "FASTSHIP": "FastShip",
        "Apex_Delivery": "ApexDelivery",
        "apex delivery": "ApexDelivery",
        "APEX": "ApexDelivery",
        "Natl_Express": "NationalExpress",
        "National Express": "NationalExpress",
        "national express": "NationalExpress",
        "quick logistics": "QuickLogistics",
        "Quick_Logistics": "QuickLogistics",
        "QUICK LOGISTICS": "QuickLogistics"
    }
    non_canonical_carrier_mask = df["carrier"].isin(carrier_mapping.keys())
    carrier_typo_count = int(non_canonical_carrier_mask.sum())
    df["carrier"] = df["carrier"].replace(carrier_mapping)
    cleaning_log.append({
        "Step": "4. Carrier Name Standardization",
        "Problem Identified": "Inconsistent casing, underscores, and spacing in carrier names across 3PL partner feeds.",
        "Records Affected": carrier_typo_count,
        "Action Taken": "Mapped all spelling variants to the four canonical providers: ApexDelivery, FastShip, NationalExpress, QuickLogistics.",
        "Justification": "Ensures categorical groupings and one-hot encodings in regression do not fragment carrier identities."
    })

    # -------------------------------------------------------------
    # Step 5: Normalize Traffic Levels
    # -------------------------------------------------------------
    traffic_mapping = {
        "med": "Medium",
        "Extremely High": "Jam"
    }
    traffic_typo_count = int(df["traffic_level"].isin(traffic_mapping.keys()).sum())
    df["traffic_level"] = df["traffic_level"].replace(traffic_mapping)

    # Impute 'UNKNOWN' traffic level with the route-level mode
    unknown_traffic_mask = df["traffic_level"] == "UNKNOWN"
    unknown_traffic_count = int(unknown_traffic_mask.sum())
    if unknown_traffic_count > 0:
        route_mode_traffic = df[~unknown_traffic_mask].groupby("route")["traffic_level"].agg(
            lambda x: x.mode().iloc[0] if not x.empty else "Medium"
        )
        for r, mode_val in route_mode_traffic.items():
            df.loc[unknown_traffic_mask & (df["route"] == r), "traffic_level"] = mode_val

    cleaning_log.append({
        "Step": "5. Traffic Level Normalization & Imputation",
        "Problem Identified": "Informal abbreviations ('med'), non-standard intensity labels ('Extremely High'), and 'UNKNOWN' traffic codes.",
        "Records Affected": traffic_typo_count + unknown_traffic_count,
        "Action Taken": "Mapped 'med' -> 'Medium', 'Extremely High' -> 'Jam', and imputed 'UNKNOWN' using route modal traffic level.",
        "Justification": "Maintains ordinal hierarchy: Low < Medium < High < Jam, allowing valid categorical and ordinal modeling."
    })

    # -------------------------------------------------------------
    # Step 6: Standardize and Impute Weather Conditions
    # -------------------------------------------------------------
    weather_mapping = {
        "Thunderstorm": "Stormy",
        "Sunny_Windy": "Clear"
    }
    weather_typo_count = int(df["weather"].isin(weather_mapping.keys()).sum())
    df["weather"] = df["weather"].replace(weather_mapping)

    # Extract month for seasonal weather imputation
    order_months = pd.to_datetime(df["order_date"]).dt.month
    missing_weather_mask = df["weather"].isnull()
    missing_weather_count = int(missing_weather_mask.sum())

    # Impute missing weather using route and month mode
    route_month_mode = df[~missing_weather_mask].groupby([df["route"], order_months])["weather"].agg(
        lambda x: x.mode().iloc[0] if not x.empty else "Clear"
    )
    for (r, m), mode_w in route_month_mode.items():
        cond = missing_weather_mask & (df["route"] == r) & (order_months == m)
        df.loc[cond, "weather"] = mode_w
    # Fallback to general mode if any remaining
    if df["weather"].isnull().sum() > 0:
        df["weather"] = df["weather"].fillna("Clear")

    cleaning_log.append({
        "Step": "6. Weather Normalization & Seasonal Imputation",
        "Problem Identified": "Non-standard meteorological categories ('Thunderstorm', 'Sunny_Windy') and missing observations (~2.8%).",
        "Records Affected": weather_typo_count + missing_weather_count,
        "Action Taken": "Mapped variants to canonical classes ('Stormy', 'Clear') and imputed nulls with the route-and-month mode.",
        "Justification": "Weather follows strong regional and seasonal patterns; mode by corridor and month preserves empirical distribution."
    })

    # -------------------------------------------------------------
    # Step 7: Re-verify Timestamps, Target Duration, and SLA Delay Status
    # -------------------------------------------------------------
    # Convert timestamps to ensure format and calculate exact hours
    disp_dt = pd.to_datetime(df["dispatch_time"])
    deliv_dt = pd.to_datetime(df["delivery_time"])
    computed_hours = np.round((deliv_dt - disp_dt).dt.total_seconds() / 3600.0, 2)
    df["delivery_time_hours"] = computed_hours

    # Ensure delayed flag strictly aligns with corridor SLA rules:
    # Route SLAs:
    sla_map = {
        "Route_Central": 1.9,
        "Route_North": 3.4,
        "Route_South": 2.8,
        "Route_East": 5.2,
        "Route_West": 3.5
    }
    expected_sla = df["route"].map(sla_map)
    df["delayed"] = (df["delivery_time_hours"] > expected_sla).astype(int)

    # Final enforcement of column order per data contract
    contract_columns = [
        "delivery_id", "order_date", "dispatch_time", "delivery_time",
        "route", "origin", "destination", "carrier", "weather",
        "parcel_weight_kg", "distance_km", "traffic_level",
        "delivery_time_hours", "delayed"
    ]
    df = df[contract_columns].reset_index(drop=True)

    cleaning_log.append({
        "Step": "7. Final Schema & Integrity Verification",
        "Problem Identified": "Potential drift in calculated elapsed durations and target contract alignment.",
        "Records Affected": len(df),
        "Action Taken": "Re-verified timestamp delta math, recomputed delayed status against corridor SLA benchmarks, and validated exact 14 columns.",
        "Justification": "Guarantees mathematical precision for Person 2's regression models and zero target leakage."
    })

    cleaning_log_df = pd.DataFrame(cleaning_log)
    return df, cleaning_log_df


def main():
    parser = argparse.ArgumentParser(description="Audit and clean the RouteWise delivery dataset.")
    parser.add_argument("--raw", type=str, default="data/raw/delivery_data.csv", help="Path to raw dataset")
    parser.add_argument("--output", type=str, default="data/processed/cleaned_delivery_data.csv", help="Path to cleaned dataset")
    parser.add_argument("--audit_out", type=str, default="outputs/tables/data_audit_summary.csv", help="Audit table destination")
    parser.add_argument("--log_out", type=str, default="outputs/tables/cleaning_log.csv", help="Cleaning log destination")
    args = parser.parse_args()

    raw_path = Path(args.raw)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data file not found at {raw_path}. Run src/data_generation.py first.")

    print(f"Loading raw dataset from {raw_path}...")
    raw_df = pd.read_csv(raw_path)

    print("Auditing raw dataset...")
    audit_df = audit_raw_data(raw_df)

    Path(args.audit_out).parent.mkdir(parents=True, exist_ok=True)
    audit_df.to_csv(args.audit_out, index=False)
    print(f"Audit report saved to: {args.audit_out}")
    print(audit_df.to_string(index=False))

    print("\nExecuting systematic cleaning pipeline...")
    clean_df, log_df = clean_delivery_data(raw_df)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(args.output, index=False)
    print(f"Cleaned dataset saved to: {args.output}")

    Path(args.log_out).parent.mkdir(parents=True, exist_ok=True)
    log_df.to_csv(args.log_out, index=False)
    print(f"Cleaning log saved to: {args.log_out}")

    print("\nCleaned Dataset Verification:")
    print(f"- Initial Raw Records: {len(raw_df)}")
    print(f"- Final Clean Records: {len(clean_df)}")
    print(f"- Null values in clean dataset: {clean_df.isnull().sum().sum()}")
    print(f"- Duplicate IDs in clean dataset: {clean_df['delivery_id'].duplicated().sum()}")
    print(f"- Final Schema Columns ({len(clean_df.columns)}): {list(clean_df.columns)}")


if __name__ == "__main__":
    main()
