# RouteWise Integration Contract: Data Interface for Person 1 & Person 2

## 1. Overview and Purpose
This document defines the data contract and schema agreement between **Person 1** (Data Engineering, Cleaning, EDA, OLAP, and Statistical Analysis) and **Person 2** (Regression Modelling, Model Evaluation, and Residual Analysis) for the **RouteWise: Supply Chain Delivery Analytics & Prediction** project.

The dataset records distributor package delivery operations across multiple supply chain routes, logistics carriers, vehicle traffic tiers, and atmospheric conditions.

- **Primary Target Variable for Person 2**: `delivery_time_hours` (continuous float).
- **Secondary Classification Indicator**: `delayed` (binary 0/1 integer: 0 = on-time, 1 = delayed beyond expected threshold).
- **Data File Location**: `data/processed/cleaned_delivery_data.csv`

---

## 2. Agreed Final Schema

The cleaned dataset saved at `data/processed/cleaned_delivery_data.csv` strictly conforms to the following 14 columns:

| Column Name | Data Type | Nullable | Example Value | Description & Constraints |
|---|---|---|---|---|
| `delivery_id` | `object` (string) | No | `"DEL-10245"` | Unique delivery identification code. |
| `order_date` | `object` / `string` (date) | No | `"2024-03-15"` | ISO 8601 date (`YYYY-MM-DD`) when the order was placed. |
| `dispatch_time` | `object` / `datetime` | No | `"2024-03-15 08:30:00"` | Timestamp when parcel was dispatched from origin. |
| `delivery_time` | `object` / `datetime` | No | `"2024-03-15 12:45:00"` | Timestamp when parcel was delivered to destination. Must be strictly after `dispatch_time`. |
| `route` | `object` (category) | No | `"Route_Central"` | Standardized route code (`Route_North`, `Route_South`, `Route_East`, `Route_West`, `Route_Central`). |
| `origin` | `object` (category) | No | `"Warehouse_Central"` | Origin depot (`Warehouse_North`, `Warehouse_South`, `Warehouse_Central`, `Warehouse_West`). |
| `destination` | `object` (category) | No | `"Zone_Metro"` | Destination cluster (`Zone_Metro`, `Zone_Suburbs`, `Zone_Industrial`, `Zone_Rural`). |
| `carrier` | `object` (category) | No | `"ApexDelivery"` | Standardized carrier name (`ApexDelivery`, `FastShip`, `NationalExpress`, `QuickLogistics`). All inconsistent spellings corrected. |
| `weather` | `object` (category) | No | `"Clear"` | Weather condition (`Clear`, `Rainy`, `Foggy`, `Stormy`, `Snowy`). Imputed where originally missing. |
| `parcel_weight_kg` | `float64` | No | `5.42` | Weight of parcel in kilograms. Strictly positive ($0.1 \le w \le 40.0$). Missing values imputed; negative values corrected/filtered. |
| `distance_km` | `float64` | No | `85.6` | Delivery trip transit distance in kilometres ($5.0 \le d \le 450.0$). |
| `traffic_level` | `object` (category) | No | `"Medium"` | Transit traffic intensity (`Low`, `Medium`, `High`, `Jam`). |
| `delivery_time_hours` | `float64` | No | `4.25` | **Target Variable**: Actual transit duration in decimal hours ($0.5 \le t \le 20.0$), calculated as $(\text{delivery\_time} - \text{dispatch\_time})$. |
| `delayed` | `int64` | No | `0` | Delay flag: `1` if transit time exceeded route SLA buffer threshold, `0` otherwise. |

---

## 3. Data Integrity & Cleaning Decisions

Person 1 enforces the following data cleanliness standards on `data/processed/cleaned_delivery_data.csv`:

1. **No Missing Values**: Every column in the processed file has 0 null/NaN entries.
   - Missing `weather` observations are imputed using mode by route and month.
   - Missing `parcel_weight_kg` observations are imputed using route-specific median weights.
2. **Duplicate Resolution**: Deduplication performed on `delivery_id` ensuring exactly one record per shipment.
3. **Canonical Categorical Naming**:
   - `carrier` is strictly cleaned of case irregularities, typos (e.g. `Apex_Delivery` -> `ApexDelivery`, `fastship` -> `FastShip`, `Natl_Express` -> `NationalExpress`).
   - `weather` is normalized to title-case canonical names (`Clear`, `Rainy`, `Foggy`, `Stormy`, `Snowy`).
   - `traffic_level` is standardized (`Low`, `Medium`, `High`, `Jam`).
4. **Range & Validity Constraints**:
   - Negative or zero `parcel_weight_kg` are cleaned or imputed based on log-normal distribution constraints.
   - Unrealistic transit speeds (e.g., speed > 140 km/h or duration < 0.2 hours for long hauls) are audited and resolved.
   - `delivery_time` is mathematically verified: `delivery_time_hours == round((delivery_time - dispatch_time).total_seconds() / 3600.0, 2)`.

---

## 4. Modelling Guidance for Person 2

To prevent data leakage and maintain experimental integrity:

- **Target Variable**: Use `delivery_time_hours` as the continuous target $y$.
- **Forbidden Predictors**:
  - `delayed` **MUST NOT** be used as a predictor for `delivery_time_hours`, as `delayed` is directly derived from `delivery_time_hours`.
  - `delivery_time` timestamp **MUST NOT** be used as a feature (it leaks target completion duration).
  - `delivery_id` is an arbitrary tracking key and should be dropped or used solely as an index.
- **Recommended Candidate Features for Regression**:
  - Continuous: `distance_km`, `parcel_weight_kg` (and polynomial / interaction terms e.g., $\text{distance} \times \text{traffic}$).
  - Categorical: `carrier`, `route`, `weather`, `traffic_level`, `origin`, `destination`.
  - Temporal features extractable from `dispatch_time` or `order_date`: `dispatch_hour`, `day_of_week`, `is_weekend`, `month`.
- **Pipeline Encoding**:
  - Categorical features should be encoded (e.g., `OneHotEncoder(drop='first')` or ordinal encoding for `traffic_level`) inside a `scikit-learn Pipeline` fitted strictly on training folds to prevent target leakage.
