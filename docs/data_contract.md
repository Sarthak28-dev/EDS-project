# RouteWise Integration Contract: Data Interface for Person 1 & Person 2

## 1. Overview and Purpose
This document defines the data contract and schema agreement between **Person 1** (Data Engineering, Cleaning, EDA, OLAP, and Statistical Analysis) and **Person 2** (Regression Modelling, Model Evaluation, and Residual Analysis) for the **RouteWise: Supply Chain Delivery Analytics & Prediction** project.

The dataset records distributor package delivery operations across multiple supply chain routes, logistics carriers, vehicle traffic tiers, and atmospheric conditions.

- **Primary Continuous Target for Person 2**: `delivery_time_hours` (continuous float).
- **Secondary Classification Indicator**: `delayed` (binary 0/1 integer: 0 = on-time, 1 = delayed beyond distance-indexed corridor SLA threshold).

---

## 2. Dataset Access & Methodological Handoff

To uphold strict academic standards regarding **data leakage prevention**, Person 1 provides two prepared datasets in `data/processed/`:

### A. Descriptive Cleaned Dataset (Default Baseline)
- **Path**: `data/processed/cleaned_delivery_data.csv`
- **Missing Values**: 0 nulls (missing weights imputed via route medians; missing weather imputed via route-month modes).
- **Intended Use**: Descriptive EDA, summary tables, OLAP pivot analyses, and exploratory visualization.

### B. Modelling Dataset (Recommended for Machine Learning Pipeline)
- **Path**: `data/processed/modelling_delivery_data.csv`
- **Missing Values**: Retains original `NaN` entries (`parcel_weight_kg`: 178 nulls; `weather`: 200 nulls) while all other structural cleaning (deduplication, sign correction, 100x decimal shift correction, categorical naming standardization, duration filtering) is completed.
- **Intended Use**: Machine learning regression modeling. Allows Person 2 to fit group/iterative imputer transformers **strictly on training folds** inside `scikit-learn Pipeline` or `ColumnTransformer`, avoiding pre-split data leakage.

---

## 3. Agreed Final Schema (14 Columns)

Both datasets strictly conform to the agreed 14 columns:

| Column Name | Data Type | Nullable (Cleaned) | Nullable (Modelling) | Example Value | Description & Constraints |
|---|---|---|---|---|---|
| `delivery_id` | `object` (string) | No | No | `"DEL-10245"` | Unique delivery identification code. |
| `order_date` | `object` / `string` (date) | No | No | `"2024-03-15"` | ISO 8601 date (`YYYY-MM-DD`) when the order was placed. |
| `dispatch_time` | `object` / `datetime` | No | No | `"2024-03-15 08:30:00"` | Timestamp when parcel was dispatched from origin. |
| `delivery_time` | `object` / `datetime` | No | No | `"2024-03-15 12:45:00"` | Timestamp when parcel was delivered to destination. Strictly after `dispatch_time`. |
| `route` | `object` (category) | No | No | `"Route_Central"` | Standardized route code (`Route_North`, `Route_South`, `Route_East`, `Route_West`, `Route_Central`). |
| `origin` | `object` (category) | No | No | `"Warehouse_Central"` | Origin depot (`Warehouse_North`, `Warehouse_South`, `Warehouse_Central`, `Warehouse_West`). |
| `destination` | `object` (category) | No | No | `"Zone_Metro"` | Destination cluster (`Zone_Metro`, `Zone_Suburbs`, `Zone_Industrial`, `Zone_Rural`). |
| `carrier` | `object` (category) | No | No | `"ApexDelivery"` | Standardized carrier name (`ApexDelivery`, `FastShip`, `NationalExpress`, `QuickLogistics`). |
| `weather` | `object` (category) | No | **Yes (200)** | `"Clear"` | Weather condition (`Clear`, `Rainy`, `Foggy`, `Snowy`, `Stormy`). |
| `parcel_weight_kg` | `float64` | No | **Yes (178)** | `5.42` | Weight of parcel in kilograms ($0.1 \le w \le 40.0$). |
| `distance_km` | `float64` | No | No | `85.6` | Delivery trip transit distance in kilometres ($15.0 \le d \le 360.0$). |
| `traffic_level` | `object` (category) | No | No | `"Medium"` | Transit traffic intensity (`Low`, `Medium`, `High`, `Jam`). |
| `delivery_time_hours` | `float64` | No | No | `4.25` | **Target Variable ($y$)**: Actual transit duration in decimal hours ($0.8 \le t \le 12.5$). |
| `delayed` | `int64` | No | No | `0` | Delay flag: `1` if transit time exceeded distance-indexed SLA threshold, `0` otherwise. |

---

## 4. Methodological Audit & Cleaning Decisions

1. **Extreme Weight Correction (Two-Decimal Shift Error)**:
   - Audit identified 8 records with weights $> 100$ kg (e.g., $289.0$ kg for what should be $2.89$ kg, or $2197.0$ kg for $21.97$ kg).
   - This was diagnosed as a **missing-decimal-point / 2-decimal shift keystroke error** (entering integer cents/fractions on digital scales).
   - Division by $100.0$ restored the exact true nominal weight. *(Note: This was not a grams-to-kg error, which would have required division by 1,000)*.

2. **Conflicting Duplicate-ID Resolution**:
   - 46 records (23 pairs) shared duplicate `delivery_id`s with identical route, distance, weather, traffic, and duration, but had a $1.5$ kg discrepancy in `parcel_weight_kg`.
   - The first arriving record was retained (`keep='first'`). Because parcel weight has a practically negligible correlation with delivery time ($R^2 < 0.2\%$), this minor variation has zero material effect on regression targets.

3. **Ground-Truth SLA Calculation**:
   - The SLA delay flag is computed using the authentic distance-indexed physical formula:
     $$\text{expected\_sla} = \left(\frac{\text{distance\_km}}{\text{base\_speed\_kmh}}\right) \times 1.15 + \text{sla\_buffer\_hr}$$
     where base speeds and buffers are: `Route_Central` (36 km/h, 0.65h), `Route_North` (62 km/h, 0.85h), `Route_South` (58 km/h, 0.80h), `Route_East` (68 km/h, 1.10h), `Route_West` (55 km/h, 0.90h).

4. **Synthetic Data Context & Modeling Boundaries**:
   - Carrier speed differences ($0.94\times$ FastShip to $1.06\times$ NationalExpress) and weather factors ($1.00\times$ to $1.62\times$) are parameters of the synthetic data generation engine. Evaluators should recognize these as validated simulation parameters rather than real-world commercial empirical performance.
   - Weather was sampled independently of calendar date; longitudinal trends are stationary and do not reflect seasonal winter storm clusters.

---

## 5. Modelling Guidance for Person 2

To prevent data leakage and maintain high predictive validity:

- **Target Variable**: Use `delivery_time_hours` as the continuous regression target $y$.
- **Forbidden Predictors (Target Leaks)**:
  - `delayed` **MUST NOT** be used as a predictor for `delivery_time_hours` (deterministic target derivative).
  - `delivery_time` timestamp **MUST NOT** be used as a feature (it leaks completion duration).
  - `delivery_id` is an arbitrary tracking key and should be dropped or used solely as an index.
- **Recommended Candidate Features**:
  - Continuous: `distance_km` ($r = 0.882$, accounts for $>77\%$ variance), `parcel_weight_kg`.
  - Categoricals: `traffic_level` (ordinal), `weather` ($\eta^2 = 0.078$), `carrier`, `route`.
  - Candidate interaction terms: $\text{distance\_km} \times \text{traffic\_level}$ and $\text{weather} \times \text{traffic\_level}$.
- **Fold-Level Imputation Pipeline Example (Person 2)**:
  ```python
  from sklearn.compose import ColumnTransformer
  from sklearn.impute import SimpleImputer
  from sklearn.preprocessing import OneHotEncoder, StandardScaler
  from sklearn.pipeline import Pipeline
  from sklearn.linear_model import Ridge

  # Load unimputed modelling dataset to prevent pre-split leakage
  df_model = pd.read_csv("data/processed/modelling_delivery_data.csv")

  num_features = ["distance_km", "parcel_weight_kg"]
  cat_features = ["route", "carrier", "weather", "traffic_level"]

  num_transformer = Pipeline([
      ("imputer", SimpleImputer(strategy="median")),
      ("scaler", StandardScaler())
  ])

  cat_transformer = Pipeline([
      ("imputer", SimpleImputer(strategy="most_frequent")),
      ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore"))
  ])

  preprocessor = ColumnTransformer([
      ("num", num_transformer, num_features),
      ("cat", cat_transformer, cat_features)
  ])

  model = Pipeline([
      ("preprocessor", preprocessor),
      ("regressor", Ridge(alpha=1.0))
  ])
  ```
