"""
Builds and executes notebooks/01_problem_and_data_audit.ipynb using NotebookExecutor.
"""

from pathlib import Path
import nbformat as nbf
from runner import NotebookExecutor


def main():
    executor = NotebookExecutor()
    nb = nbf.v4.new_notebook()

    executor.add_markdown(nb, """# RouteWise: Supply Chain Delivery Analytics & Prediction
## Notebook 01: Problem Framing, Data Quality Audit & Cleaning Pipeline
**Academic Exploratory Data Science Project**  
**Role: Person 1 — Data Engineering, Data Cleaning, EDA & Statistical Analysis**  
**Branch:** `person-1-eda` | **Integration Contract:** `docs/data_contract.md`

---

### Notebook Objectives:
1. **Problem Framing & Stakeholder Analysis**: Define the business environment, core operational challenges, stakeholders, and analytical questions.
2. **Data Dictionary**: Provide comprehensive variable specifications, units, data types, and operational semantics.
3. **Reproducible Synthetic Generation**: Verify the generation mechanics of the 7,000+ delivery record dataset.
4. **Comprehensive Data Quality Audit**: Systematically diagnose missing values, duplicate entries, typographical variations, invalid physical numbers, and temporal causality errors.
5. **Justified Cleaning Transformations**: Implement and log explicit, non-destructive cleaning actions with quantitative accounting.
6. **Integration Contract Compliance**: Output the final verified dataset to `data/processed/cleaned_delivery_data.csv` for downstream EDA, statistical analysis, and Person 2's regression models.""")

    executor.add_markdown(nb, """---
## 1. Business Scenario & Problem Framing

### 1.1 Operational Context
A nationwide distributor, **RouteWise Logistics Network**, oversees scheduled freight and parcel deliveries across five primary transit corridors connecting regional warehouse fulfillment centers with regional drop-off zones (Metro, Suburbs, Industrial, and Rural). Delivery operations face compounded friction from traffic congestion, variable weather events, 3PL carrier dispatch delays, and varying parcel weights.

### 1.2 Core Business Problem
Fluctuating delivery durations trigger frequent Service Level Agreement (SLA) breaches, leading to customer churn, warehouse backlogs, and financial penalty clauses. Operations leadership lacks empirical clarity on which factors drive delays.

### 1.3 Primary Analytical Question
> **"How accurately can delivery time be predicted using route, carrier, weather, parcel weight, distance, and traffic conditions?"**

### 1.4 Supporting Analytical Questions
1. **Weather Impact:** To what degree do adverse weather fronts (Rain, Fog, Snow, Storms) extend transit durations relative to clear conditions?
2. **Carrier Efficiency:** Do performance discrepancies between 3PL carriers persist after controlling for route assignment and distance?
3. **Traffic Congestion:** What is the marginal transit duration penalty incurred when moving from *Low* to *Medium*, *High*, and *Jam* traffic levels?
4. **Parcel Weight Sensitivity:** Does cargo mass exert a statistically or practically significant penalty on delivery duration?
5. **Corridor SLA Fragility:** Which specific transit corridors exhibit disproportionate rates of SLA failure?""")

    executor.add_markdown(nb, """---
## 2. Stakeholder Analysis & Decision Support

| Stakeholder Group | Primary Operational Concern | Decisions Supported by Analysis |
|---|---|---|
| **VP of Supply Chain** | Network-wide SLA compliance, margin optimization | Corridor budget allocation, carrier contract negotiations, dynamic capacity planning. |
| **Logistics Dispatch Managers** | Daily route dispatching & delivery scheduling | Setting weather-adjusted departure windows, real-time rerouting under severe congestion. |
| **Carrier Procurement Lead** | 3PL carrier tiering & vendor accountability | Performance-based carrier contracting, route assignment based on carrier velocity. |
| **Customer Experience & Sales** | Delivery ETA precision, SLA transparency | Communicating accurate delivery timeframes, reducing customer ETA variance. |
| **Data Science / ML Team (Person 2)** | Predictive regression model accuracy | High-integrity, leak-free training dataset for linear and regularized regression.""")

    executor.add_markdown(nb, """---
## 3. Data Dictionary & Contract Specifications

The dataset strictly conforms to the agreed schema negotiated in `docs/data_contract.md`.

| Column | Data Type | Nullable | Example | Description & Constraints |
|---|---|---|---|---|
| `delivery_id` | `object` (string) | No | `DEL-10245` | Unique delivery identifier (format `DEL-[0-9]{5}`). |
| `order_date` | `object` (date string) | No | `2024-03-15` | ISO 8601 date (`YYYY-MM-DD`) when order was placed. |
| `dispatch_time` | `datetime` | No | `2024-03-15 08:30:00` | Dispatch timestamp when vehicle departs origin depot. |
| `delivery_time` | `datetime` | No | `2024-03-15 12:45:00` | Actual customer delivery timestamp. Strictly after dispatch. |
| `route` | `object` (category) | No | `Route_Central` | Transit corridor (`Route_Central`, `Route_North`, `Route_South`, `Route_East`, `Route_West`). |
| `origin` | `object` (category) | No | `Warehouse_Central` | Origin warehouse facility. |
| `destination` | `object` (category) | No | `Zone_Metro` | Destination zone category. |
| `carrier` | `object` (category) | No | `ApexDelivery` | 3PL carrier (`ApexDelivery`, `FastShip`, `NationalExpress`, `QuickLogistics`). |
| `weather` | `object` (category) | No | `Clear` | Atmospheric condition (`Clear`, `Rainy`, `Foggy`, `Snowy`, `Stormy`). |
| `parcel_weight_kg` | `float64` | No | `6.45` | Parcel mass in kg. Must be strictly positive ($0.1 \\le w \\le 40.0$). |
| `distance_km` | `float64` | No | `124.5` | Total road transit distance in kilometres. |
| `traffic_level` | `object` (ordinal) | No | `Medium` | Congestion tier: `Low` < `Medium` < `High` < `Jam`. |
| `delivery_time_hours` | `float64` | No | `3.82` | **Primary Regression Target:** Actual duration in decimal hours. |
| `delayed` | `int64` | No | `0` | Binary SLA flag: `1` if duration exceeded expected corridor threshold, `0` otherwise. |""")

    executor.add_markdown(nb, """---
## 4. Setup, Environment & Raw Data Loading

We import required libraries and load the raw delivery dataset from `data/raw/delivery_data.csv`.  
Dataset generation is reproducible via `src/data_generation.py` using fixed seed `42`.""")

    code_setup = """import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure project root is in sys.path
PROJECT_ROOT = Path("..").resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_cleaning import audit_raw_data, clean_delivery_data

# Set styling
pd.set_option('display.max_columns', 15)
pd.set_option('display.width', 1000)

raw_data_path = PROJECT_ROOT / "data" / "raw" / "delivery_data.csv"
print(f"Loading raw dataset from: {raw_data_path}")
raw_df = pd.read_csv(raw_data_path)
print(f"Loaded Raw Shape: {raw_df.shape} (Rows: {raw_df.shape[0]:,}, Columns: {raw_df.shape[1]})")
raw_df.head()"""
    executor.add_code(nb, code_setup)

    executor.add_markdown(nb, """### 4.1 Initial Structural Inspection""")
    code_inspect = """print("--- Column Data Types & Non-Null Counts ---")
raw_df.info()

print("\\n--- Numerical Summary Statistics (Raw Data) ---")
raw_df.describe()"""
    executor.add_code(nb, code_inspect)

    executor.add_markdown(nb, """---
## 5. Systematic Data Quality Audit

We execute `audit_raw_data()` from `src/data_cleaning.py` to inspect dimensions, missing values, duplicate keys, typographical discrepancies, numerical validity, and temporal causality.""")

    code_audit = """audit_report = audit_raw_data(raw_df)
print("=== COMPREHENSIVE DATA QUALITY AUDIT REPORT ===")
audit_report"""
    executor.add_code(nb, code_audit)

    executor.add_markdown(nb, """### 5.1 Deep-Dive on Discovered Data Quality Defects

1. **Missing Data Patterns:**
   - `weather`: 202 missing entries (~2.8%). Caused by weather station telemetry dropouts.
   - `parcel_weight_kg`: 184 missing entries (~2.5%). Caused by warehouse weighbridge sensor disconnects.
2. **Duplicate Records:**
   - 52 exact duplicate rows resulting from database retry transactions.
   - 75 duplicate `delivery_id` instances, including 23 conflicting rows where parcel weights were re-weighed.
3. **Numerical Anomalies:**
   - 51 negative parcel weights (e.g., `-4.5 kg`) caused by terminal minus-key keystrokes.
   - 8 extreme weights (> 100 kg, e.g. 700+ kg) caused by operators entering grams instead of kilograms.
4. **Categorical Inconsistencies:**
   - 253 carrier records with non-canonical casing or formatting (`fastship`, `Apex_Delivery`, `Natl_Express`, `quick logistics`).
   - 21 invalid traffic labels (`UNKNOWN`, `Extremely High`, `med`).
   - 15 non-standard weather categories (`Thunderstorm`, `Sunny_Windy`).
5. **Temporal & Physical Violations:**
   - 14 records with non-positive delivery duration (`delivery_time_hours <= 0` or arrival before dispatch), caused by unsynchronized vehicle transponders.""")

    executor.add_markdown(nb, """---
## 6. Step-by-Step Justified Data Cleaning Pipeline

We now execute our systematic, non-destructive cleaning pipeline. Every transformation is quantitatively tracked in a persistent cleaning log.""")

    code_clean = """clean_df, cleaning_log_df = clean_delivery_data(raw_df)

print("=== CLEANING PIPELINE EXECUTION LOG ===")
cleaning_log_df"""
    executor.add_code(nb, code_clean)

    executor.add_markdown(nb, """---
## 7. Post-Cleaning Verification & Contract Conformance

We audit the cleaned dataset against all constraints set forth in the Person 2 Data Contract (`docs/data_contract.md`):
- **Missing Values:** Exactly 0 nulls across all 14 columns.
- **Uniqueness:** 0 duplicate rows, 0 duplicate delivery IDs.
- **Numerical Validity:** All parcel weights and delivery durations strictly positive.
- **Categorical Purity:** All carrier names, routes, and weather conditions strictly match canonical vocabularies.""")

    code_verify = """print("--- POST-CLEANING INTEGRITY AUDIT ---")
print(f"Raw record count:     {len(raw_df):,}")
print(f"Cleaned record count: {len(clean_df):,}")
print(f"Total rows dropped:   {len(raw_df) - len(clean_df)} ({(len(raw_df) - len(clean_df)) / len(raw_df):.2%})")
print(f"Total nulls in clean dataset: {clean_df.isnull().sum().sum()}")
print(f"Duplicate delivery_ids:       {clean_df['delivery_id'].duplicated().sum()}")
print(f"Min parcel weight: {clean_df['parcel_weight_kg'].min():.2f} kg, Max: {clean_df['parcel_weight_kg'].max():.2f} kg")
print(f"Min delivery time: {clean_df['delivery_time_hours'].min():.2f} hrs, Max: {clean_df['delivery_time_hours'].max():.2f} hrs")

print("\\n--- Canonical Value Sets ---")
print("Carriers:      ", sorted(clean_df['carrier'].unique()))
print("Traffic Levels:", sorted(clean_df['traffic_level'].unique()))
print("Weather:       ", sorted(clean_df['weather'].unique()))
print("Routes:        ", sorted(clean_df['route'].unique()))

print("\\nCleaned Dataset Preview:")
clean_df.head()"""
    executor.add_code(nb, code_verify)

    executor.add_markdown(nb, """### 7.1 Exporting Cleaned Dataset to Processed Storage""")
    code_save = """output_processed_path = PROJECT_ROOT / "data" / "processed" / "cleaned_delivery_data.csv"
clean_df.to_csv(output_processed_path, index=False)
print(f"Successfully saved verified cleaned dataset to: {output_processed_path.resolve()}")"""
    executor.add_code(nb, code_save)

    target_path = Path("notebooks/01_problem_and_data_audit.ipynb")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Notebook 01 successfully generated and saved at: {target_path}")


if __name__ == "__main__":
    main()
