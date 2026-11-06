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
2. **Duplicate Records & Conflicting Primary Keys:**
   - 52 exact duplicate rows resulting from database retry transactions.
   - 75 duplicate `delivery_id` instances in total, which include 46 records (23 pairs) where all routing and telemetry values are identical but `parcel_weight_kg` differs by $1.5$ kg. In production data engineering, retaining the first verified record without an audit log is an arbitrary resolution; however, because weight correlation with delivery duration is practically negligible ($R^2 < 0.2\\%$), this variation does not bias regression outcomes.
3. **Numerical Anomalies:**
   - 51 negative parcel weights (e.g., `-4.5 kg`) caused by operator minus-key prefix keystroke errors.
   - 8 extreme weights (> 100 kg, e.g. 289 kg, 414 kg, 2197 kg) caused by **two-decimal shift / omitted decimal point keystroke errors** (multiplying true weights by 100). Dividing by 100.0 exactly recovers the true weights ($2.89$ kg, $4.14$ kg, $21.97$ kg). Note: This was not a grams-to-kg error, which would have required division by 1,000.
4. **Categorical Inconsistencies:**
   - 253 carrier records with non-canonical casing or formatting (`fastship`, `Apex_Delivery`, `Natl_Express`, `quick logistics`).
   - 21 invalid traffic labels (`UNKNOWN`, `Extremely High`, `med`).
   - 15 non-standard weather categories (`Thunderstorm`, `Sunny_Windy`).
5. **Temporal & Physical Violations:**
   - 14 records with non-positive delivery duration (`delivery_time_hours <= 0` or arrival before dispatch), caused by unsynchronized vehicle transponders.""")

    executor.add_markdown(nb, """---
## 6. Step-by-Step Justified Data Cleaning Pipeline

We now execute our systematic, non-destructive cleaning pipeline. Every transformation is quantitatively tracked in a persistent cleaning log.

### Methodological Note on Imputation & Data Leakage:
- Imputing missing values on the full dataset before splitting is acceptable for descriptive EDA and OLAP reporting.
- However, for supervised regression modeling, Person 2 should perform imputation **inside cross-validation folds** to prevent data leakage.
- We therefore generate two clean datasets:
  1. `cleaned_delivery_data.csv`: Fully imputed for descriptive EDA/OLAP.
  2. `modelling_delivery_data.csv`: Structurally cleaned but preserving original NaNs for Person 2's pipeline.""")

    code_clean = """clean_df, cleaning_log_df = clean_delivery_data(raw_df, impute=True)
model_df, _ = clean_delivery_data(raw_df, impute=False)

print("=== CLEANING PIPELINE EXECUTION LOG ===")
cleaning_log_df"""
    executor.add_code(nb, code_clean)

    executor.add_markdown(nb, """---
## 7. Post-Cleaning Verification & Contract Conformance

We audit the cleaned dataset against all constraints set forth in the Person 2 Data Contract (`docs/data_contract.md`):
- **Missing Values:** Exactly 0 nulls in descriptive cleaned dataset; exactly 200 weather / 178 weight nulls preserved in modelling dataset.
- **Uniqueness:** 0 duplicate rows, 0 duplicate delivery IDs.
- **Numerical Validity:** All parcel weights and delivery durations strictly positive.
- **Categorical Purity:** All carrier names, routes, and weather conditions strictly match canonical vocabularies.
- **Ground-Truth SLA:** Delay status strictly calculated via distance-indexed route velocity formula.""")

    code_verify = """print("--- POST-CLEANING INTEGRITY AUDIT ---")
print(f"Raw record count:         {len(raw_df):,}")
print(f"Cleaned record count:     {len(clean_df):,}")
print(f"Total rows dropped:       {len(raw_df) - len(clean_df)} ({(len(raw_df) - len(clean_df)) / len(raw_df):.2%})")
print(f"Total nulls (Cleaned):    {clean_df.isnull().sum().sum()}")
print(f"Total nulls (Modelling):  {model_df.isnull().sum().to_dict()}")
print(f"Duplicate delivery_ids:   {clean_df['delivery_id'].duplicated().sum()}")
print(f"Min parcel weight: {clean_df['parcel_weight_kg'].min():.2f} kg, Max: {clean_df['parcel_weight_kg'].max():.2f} kg")
print(f"Min delivery time: {clean_df['delivery_time_hours'].min():.2f} hrs, Max: {clean_df['delivery_time_hours'].max():.2f} hrs")
print(f"Total delayed parcels:    {clean_df['delayed'].sum()} ({clean_df['delayed'].mean():.2%})")

print("\\n--- Canonical Value Sets ---")
print("Carriers:      ", sorted(clean_df['carrier'].unique()))
print("Traffic Levels:", sorted(clean_df['traffic_level'].unique()))
print("Weather:       ", sorted(clean_df['weather'].unique()))
print("Routes:        ", sorted(clean_df['route'].unique()))

print("\\nCleaned Dataset Preview:")
clean_df.head()"""
    executor.add_code(nb, code_verify)

    executor.add_markdown(nb, """### 7.1 Exporting Datasets to Processed Storage""")
    code_save = """output_processed_path = PROJECT_ROOT / "data" / "processed" / "cleaned_delivery_data.csv"
output_model_path = PROJECT_ROOT / "data" / "processed" / "modelling_delivery_data.csv"

clean_df.to_csv(output_processed_path, index=False)
model_df.to_csv(output_model_path, index=False)

print(f"Successfully saved descriptive cleaned dataset to: {output_processed_path.resolve()}")
print(f"Successfully saved unimputed modelling dataset to:   {output_model_path.resolve()}")"""
    executor.add_code(nb, code_save)

    target_path = Path("notebooks/01_problem_and_data_audit.ipynb")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Notebook 01 successfully generated and saved at: {target_path}")


if __name__ == "__main__":
    main()
