# RouteWise: Supply Chain Delivery Analytics & Prediction
**Academic Exploratory Data Science (EDS) Collaborative Project**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Branch: person-1-eda](https://img.shields.io/badge/branch-person--1--eda-green.svg)](https://github.com/)
[![Status: Person 1 Complete](https://img.shields.io/badge/status-Person_1_Complete-success.svg)](https://github.com/)

---

## 1. Project Background & Overview
A fictional distributor, **RouteWise Logistics Network**, oversees scheduled freight and parcel deliveries across five primary transit corridors connecting regional warehouses with customer drop-off clusters (Metro, Suburbs, Industrial, Rural). Delivery operations face compounded friction from traffic congestion, adverse weather, 3PL carrier dispatch variability, and cargo weights.

The primary objective of RouteWise is to determine:
> **"How accurately can delivery time be predicted using route, carrier, weather, parcel weight, distance, and traffic conditions?"**

This collaborative project is divided into two distinct responsibilities:
- **Person 1 (Current Branch: `person-1-eda`)**: Data Engineering, Data Auditing & Cleaning, Preprocessing, Exploratory Data Analysis (EDA), OLAP Analysis, and Statistical Hypothesis Testing.
- **Person 2 (Downstream Branch: `person-2-modelling`)**: Feature Engineering Pipelines, Regression Modeling (Linear, Ridge/Lasso, Tree-based), Cross-Validation, Hyperparameter Tuning, and Residual Diagnostics.

---

## 2. Repository Structure

```
RouteWise/
├── data/
│   ├── raw/
│   │   └── delivery_data.csv                   # Raw generated dataset (7,225 records with controlled imperfections)
│   └── processed/
│       ├── cleaned_delivery_data.csv           # Descriptive dataset (7,136 clean records, 0 missing, fully imputed)
│       └── modelling_delivery_data.csv         # Modelling dataset (7,136 records, 0 leaks, NaNs preserved for CV imputation)
├── docs/
│   ├── problem_framing.md                     # Business problem, stakeholders & decision analysis
│   ├── data_dictionary.md                     # Comprehensive variable metadata, units & domains
│   └── data_contract.md                       # Formal Person 1 & Person 2 schema integration contract
├── notebooks/
│   ├── 01_problem_and_data_audit.ipynb        # Problem framing, raw audit & step-by-step cleaning pipeline
│   ├── 02_eda_and_olap.ipynb                  # 9 EDA figures, Pandas/NumPy operations & 8 OLAP tables
│   └── 03_statistical_analysis.ipynb          # Hypothesis testing (ANOVA/Kruskal-Wallis, Dunn's test, correlation)
├── outputs/
│   ├── figures/                               # 9 high-resolution exploratory figures (PNG, 300 DPI)
│   │   ├── 01_delivery_time_distribution.png
│   │   ├── 02_avg_delivery_time_by_carrier.png
│   │   ├── 03_avg_delivery_time_by_route.png
│   │   ├── 04_delivery_time_by_weather.png
│   │   ├── 05_parcel_weight_vs_delivery_time.png
│   │   ├── 06_distance_vs_delivery_time.png
│   │   ├── 07_traffic_level_vs_delivery_time.png
│   │   ├── 08_correlation_heatmap.png
│   │   └── 09_delivery_trends_over_time.png
│   └── tables/                                # Audit reports, cleaning logs, OLAP & statistical tables
│       ├── data_audit_summary.csv
│       ├── cleaning_log.csv
│       ├── olap_1_carrier_performance.csv
│       ├── olap_2_route_performance.csv
│       ├── olap_3_weather_by_carrier_counts.csv
│       ├── olap_4_delay_percentage_by_route.csv
│       ├── olap_5_carrier_by_route_matrix.csv
│       ├── olap_6_monthly_delivery_performance.csv
│       ├── olap_7_route_vs_weather_matrix.csv
│       ├── olap_8_weight_category_summary.csv
│       ├── hypothesis_tests_weather.csv
│       ├── pairwise_weather_comparisons.csv
│       ├── weather_group_descriptives.csv
│       └── parcel_weight_correlation_analysis.csv
├── scripts/
│   ├── runner.py                              # Notebook execution and output capture utility
│   ├── create_notebook_01.py
│   ├── create_notebook_02.py
│   └── create_notebook_03.py
├── src/
│   ├── data_generation.py                     # Synthetic data generation engine (fixed seed 42)
│   ├── data_cleaning.py                       # Audit & cleaning pipeline module
│   ├── eda.py                                 # Visualizations & OLAP table generator
│   └── statistical_analysis.py                # Hypothesis testing & Dunn's post-hoc module
└── README.md
```

---

## 3. Person 2 Integration Contract Summary

Person 2 is provided with two clean datasets depending on workflow needs:
1. **For Descriptive Analysis**: `data/processed/cleaned_delivery_data.csv` (0 nulls; imputed via route medians/modes).
2. **For Predictive Regression Pipelines**: `data/processed/modelling_delivery_data.csv` (all structural defects resolved, but NaNs in `parcel_weight_kg` [178] and `weather` [200] preserved to allow fold-level imputation inside cross-validation without pre-split data leakage).

Both datasets strictly conform to the agreed 14 columns:
1. `delivery_id`: Unique shipment identifier string (`DEL-[0-9]{5}`).
2. `order_date`: Order date (`YYYY-MM-DD`).
3. `dispatch_time`: Dispatch timestamp (`YYYY-MM-DD HH:MM:SS`).
4. `delivery_time`: Delivery timestamp (`YYYY-MM-DD HH:MM:SS`). *(Do not use as a feature in ML: leaks duration)*.
5. `route`: Transit corridor nominal category (`Route_Central`, `Route_North`, `Route_South`, `Route_East`, `Route_West`).
6. `origin`: Origin depot category.
7. `destination`: Destination zone category.
8. `carrier`: 3PL provider category (`ApexDelivery`, `FastShip`, `NationalExpress`, `QuickLogistics`).
9. `weather`: Weather condition category (`Clear`, `Rainy`, `Foggy`, `Snowy`, `Stormy`).
10. `parcel_weight_kg`: Numerical continuous weight ($0.1 \le w \le 40.0$ kg).
11. `distance_km`: Numerical continuous odometer distance ($15.0 \le d \le 360.0$ km).
12. `traffic_level`: Ordinal congestion tier (`Low`, `Medium`, `High`, `Jam`).
13. `delivery_time_hours`: **Primary Continuous Target ($y$)** in decimal hours ($0.8 \le t \le 12.5$).
14. `delayed`: Binary SLA audit flag ($0 = \text{on-time}, 1 = \text{delayed}$ based on distance-indexed formula). *(Do not use as a predictor: target derivative)*.

See [`docs/data_contract.md`](docs/data_contract.md) for full technical contract details.

---

## 4. Reproducibility Guide

All stages of the data engineering and analytical pipeline are 100% reproducible through simple Python CLI commands:

### Step 1: Generate Synthetic Raw Dataset (~7,000 records)
```bash
python src/data_generation.py --records 7150 --seed 42 --output data/raw/delivery_data.csv
```

### Step 2: Run Data Audit & Systematic Cleaning Pipeline
```bash
python src/data_cleaning.py --raw data/raw/delivery_data.csv --output data/processed/cleaned_delivery_data.csv
```

### Step 3: Generate All 9 EDA Visualizations & 8 OLAP Summaries
```bash
python src/eda.py --input data/processed/cleaned_delivery_data.csv
```

### Step 4: Execute Statistical Hypothesis Testing & Dunn's Post-Hoc Tests
```bash
python src/statistical_analysis.py --input data/processed/cleaned_delivery_data.csv
```

### Step 5: (Re-)generate & Execute Full Jupyter Notebooks
```bash
python scripts/create_notebook_01.py
python scripts/create_notebook_02.py
python scripts/create_notebook_03.py
```

---

## 5. Key Empirical Findings & Operational Insights

### Pattern 1: Corridor Distance Dominates Delivery Durations ($r = 0.882$)
Spatial corridor length accounts for $>77\%$ of direct transit time variance. Long-haul rural corridors (`Route_East`, mean distance $255.4$ km) average $4.50$ hours with a $38.6\%$ delay rate, compared to short urban corridors (`Route_Central`, mean distance $34.8$ km) which average $1.39$ hours with a $18.7\%$ delay rate.

### Pattern 2: Multiplicative Compound Delay from Severe Weather and Traffic Jams
While congestion alone doubles baseline delivery times, severe weather events (`Stormy`, `Snowy`) interacting with `Jam` traffic multiply transit duration by $+243\%$ (jumping from $2.08$ hours to $7.14$ hours), triggering a $68.4\%$ SLA failure rate. Dispatch protocols should hold or dynamically reroute consignments under compound weather/traffic alerts.

### Pattern 3: Carrier Velocity Parameter Recovery Across Corridors
Within the parameterized synthetic simulation environment, `FastShip` ($0.94\times$ multiplier) consistently achieves the lowest transit durations across all corridors ($1.30$h Central, $4.26$h East; $23.2\%$ delay rate), whereas `NationalExpress` ($1.06\times$ multiplier) consistently requires the highest durations ($1.48$h Central, $4.76$h East; $37.9\%$ delay rate). Because carrier assignments are evenly balanced across routes, this difference reflects recovery of the generator's velocity factors rather than route confounding. (These reflect validated synthetic simulation parameters rather than real-world commercial performance).

### Statistical Hypothesis Testing Highlights:
- **Weather Impact (RQ 1):** Both One-Way ANOVA ($F = 149.77, p = 3.37 \times 10^{-123}$) and Kruskal-Wallis ($H = 429.95, p = 9.37 \times 10^{-92}$) strongly reject $H_0$ ($\eta^2 = 0.078, \epsilon^2 = 0.060$). 
  - Because Levene's test proved heteroscedasticity ($p = 7.17 \times 10^{-49}$), Kruskal-Wallis is interpreted as testing **stochastic dominance / mean rank distributions** rather than strict equality of medians.
  - **Dunn's Post-Hoc Test** with Bonferroni correction ($\alpha_{\text{adj}} = 0.005$) confirms that mean ranks monotonically increase with atmospheric severity:
    $$\text{Clear } (\bar{R}=3187) < \text{Rainy } (\bar{R}=3600) < \text{Foggy } (\bar{R}=3912) < \text{Snowy } (\bar{R}=4396) < \text{Stormy } (\bar{R}=4891)$$
    with all 10 pairwise comparisons maintaining statistical significance ($p_{\text{adj}} < 0.005$).
- **Parcel Weight Sensitivity (RQ 2):** Pearson correlation is $r = 0.0440$ ($p = 0.000203$). While statistically significant due to large sample size ($N = 7,136$), it is **practically negligible** ($R^2 = 0.19\%$). Parcel weight explains less than $0.2\%$ of transit variance.
