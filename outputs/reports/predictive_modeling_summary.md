# RouteWise: Predictive Modeling & Residual Diagnostics Report
**Component:** Component 4 — Machine Learning & Predictive Analytics  
**Author / Contributor:** Person 2  
**Project:** RouteWise: Supply Chain Delivery Analytics & Prediction  
**Repository Branch:** `person-2-modeling`  
**Execution Timestamp:** 2026-11-06  

---

## 1. Executive Summary

A regional freight distributor commissioned this exploratory data science initiative to determine the feasibility of predicting shipment delivery durations from pre-dispatch and environmental attributes. 

The primary analytical question addressed is:
> **"How accurately can delivery time be predicted using route, carrier, weather, parcel weight, distance, and traffic conditions?"**

### Core Findings
1. **Predictive Accuracy:** Machine learning pipelines explain **96.0% of delivery duration variance ($R^2 \approx 0.960$)** under 5-fold cross-validation, and **96.0% on an untouched 20% test partition** ($N = 1,428$).
2. **Operational Error Bounds:**
   - **Mean Absolute Error (MAE):** **0.221 hours (~13.3 minutes)** on held-out test deliveries.
   - **Root Mean Squared Error (RMSE):** **0.353 hours (~21.2 minutes)**, confirming that even when larger delays occur, prediction errors remain bounded and well-controlled.
3. **Algorithm Hierarchy & Selection:**
   - **HistGradientBoosting Regressor** achieved top cross-validated performance ($R^2 = 0.960 \pm 0.003$, $\text{MAE} = 0.234 \pm 0.006$ hrs, $\text{RMSE} = 0.358 \pm 0.018$ hrs) and was selected as the final production model.
   - **Random Forest Regressor** demonstrated strong competitive performance ($R^2 = 0.948 \pm 0.003$, $\text{MAE} = 0.265 \pm 0.004$ hrs, $\text{RMSE} = 0.405 \pm 0.018$ hrs), but was strictly outperformed by HistGradientBoosting across all metrics.
   - **Linear Regression** served as an interpretable baseline ($R^2 = 0.912 \pm 0.002$, $\text{MAE} = 0.365 \pm 0.005$ hrs), but exhibited structural bias due to unmodeled non-linear weather/traffic interaction terms.
   - **Dummy Baseline** failed completely ($R^2 = -0.001 \pm 0.001$, $\text{MAE} = 1.366 \pm 0.018$ hrs), validating the necessity of machine learning over naive historical central-tendency quoting.
4. **Key Diagnostic Insights:**
   - **Severe Weather Events:** `Stormy` conditions produce the highest prediction error ($\text{MAE} = 0.442$ hrs, $\text{RMSE} = 0.652$ hrs), followed by `Snowy` conditions ($\text{MAE} = 0.250$ hrs).
   - **Corridor Dispersion:** Longer-distance routes (`Route_East` $\text{MAE} = 0.414$ hrs, `Route_West` $\text{MAE} = 0.275$ hrs, `Route_North` $\text{MAE} = 0.251$ hrs) display wider residual dispersion than compact urban routes (`Route_Central` $\text{MAE} = 0.101$ hrs).
   - **Carrier Neutrality:** All carriers (`ApexDelivery`, `FastShip`, `NationalExpress`, `QuickLogistics`) show symmetrical residual distributions with means near zero ($-0.014$ to $+0.026$ hrs), confirming model fairness across logistics partners.

---

## 2. Problem Formulation & Feature Architecture

### Target Variable
- **`delivery_time_hours`**: Continuous elapsed duration from dispatch to customer receipt (range: 0.43 to 13.17 hours, mean: 3.05 hours).

### Predictor Set
- **Continuous Numerical Predictors:**
  - `distance_km`: Journey distance in kilometers.
  - `parcel_weight_kg`: Weight of the consignment in kilograms (contains 178 intentional missing values in processed dataset).
- **Categorical Nominal Predictors:**
  - `route`: Transportation corridor (`Route_Central`, `Route_North`, `Route_South`, `Route_East`, `Route_West`).
  - `carrier`: Commercial courier partner (`ApexDelivery`, `FastShip`, `NationalExpress`, `QuickLogistics`).
  - `weather`: Meteorological condition at dispatch (`Clear`, `Rainy`, `Foggy`, `Snowy`, `Stormy` — contains 200 intentional missing values).
  - `traffic_level`: Traffic congestion tier (`Low`, `Medium`, `High`, `Jam`).

### Target Leakage Prevention & Excluded Variables
The following columns are strictly excluded from the predictor matrix:
1. `delivery_id`: Arbitrary transaction identifier.
2. `delivery_time`: Exact delivery timestamp (directly reveals target $y = \text{delivery\_time} - \text{dispatch\_time}$).
3. `delayed`: Binary boolean indicator derived directly from delivery duration relative to SLA (fatal target leakage).
4. `order_date` and `dispatch_time`: Excluded from static regression to prevent memorizing calendar-specific artifacts without physical causal justification.
5. `origin` and `destination`: Collinear location identifiers redundant with defined corridor routes.

---

## 3. Preprocessing & Validation Methodology

### Featurization Ordering & Train/Test Partitioning
To maintain strict methodological rigor, the primary dataset (`data/processed/modelling_delivery_data.csv`, 7,136 rows) is partitioned into an **80% training split (5,708 samples)** and a **20% held-out test split (1,428 samples)** using a deterministic seed (`random_state=42`) **before** any transformations or imputations are fitted.

### Scikit-Learn Pipeline Design & Leakage Prevention
All feature transformations are encapsulated inside a `ColumnTransformer` embedded in unified `Pipeline` objects:
- **Numerical Sub-pipeline:** `SimpleImputer(strategy='median')` $\rightarrow$ `StandardScaler()`.
- **Categorical Sub-pipeline:** `SimpleImputer(strategy='most_frequent')` $\rightarrow$ `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`.
- **In-Fold Imputation Isolation:** Because imputation and scaling steps are enclosed within Scikit-learn Pipeline objects, every cross-validation fold fits its own imputation statistics strictly on training fold data. At no point is the full dataset or test partition used to calculate preprocessing statistics, completely eliminating data leakage.

---

## 4. Cross-Validation Benchmarking Results

Cross-validation was conducted with 5 folds on `X_train` (5,708 samples). Metrics are reported with their mean and standard deviation:

| Model | CV MAE (hours) | CV RMSE (hours) | CV MSE ($\text{hours}^2$) | CV $R^2$ |
|:---|:---:|:---:|:---:|:---:|
| **HistGradientBoosting** | **0.234 $\pm$ 0.006** | **0.358 $\pm$ 0.018** | **0.129 $\pm$ 0.013** | **0.960 $\pm$ 0.003** |
| **Random Forest** | 0.265 $\pm$ 0.004 | 0.405 $\pm$ 0.018 | 0.164 $\pm$ 0.015 | 0.948 $\pm$ 0.003 |
| **Linear Regression** | 0.365 $\pm$ 0.005 | 0.528 $\pm$ 0.014 | 0.279 $\pm$ 0.015 | 0.912 $\pm$ 0.002 |
| **Dummy Baseline** | 1.366 $\pm$ 0.018 | 1.785 $\pm$ 0.054 | 3.188 $\pm$ 0.192 | -0.001 $\pm$ 0.001 |

### Model Selection Justification
- **Dominance Across Metrics:** HistGradientBoosting achieves the lowest error across every metric (MAE 0.234 hrs vs 0.265 hrs for Random Forest; RMSE 0.358 hrs vs 0.405 hrs) and the highest explanatory power ($R^2 = 0.960$).
- **Architectural Suitability:** Gradient-boosted decision trees sequentially minimize residuals, efficiently capturing non-linear interactions between weather events, traffic levels, and transit distances.
- **Decision:** **HistGradientBoostingRegressor** is advanced as the final model for production deployment.

---

## 5. Held-Out Test Set Generalization

Evaluating the selected **HistGradientBoosting Regressor** on the completely untouched 20% test partition (1,428 shipments) confirms exceptional generalization:

| Metric | Test Set Score | Operational Meaning |
|:---|:---:|:---|
| **Test MAE** | **0.221 hours** | Average prediction deviation of **13.3 minutes** |
| **Test RMSE** | **0.353 hours** | Root mean square error of **21.2 minutes** |
| **Test MSE** | **0.125 $\text{hours}^2$** | Squared dispersion error |
| **Test $R^2$** | **0.960** | **96.0% of duration variance explained** on unseen shipments |

The test metrics closely match cross-validation estimates (CV MAE 0.234 hrs, CV RMSE 0.358 hrs), demonstrating zero overfitting.

---

## 6. Comprehensive Residual Diagnostics

Residuals are calculated strictly as:
$$\text{Residual} = \text{Actual Delivery Duration} - \text{Predicted Delivery Duration}$$

- **Positive Residual ($e_i > 0$):** Actual duration exceeded prediction (**Underprediction / Delivery Delay**).
- **Negative Residual ($e_i < 0$):** Actual duration was less than prediction (**Overprediction / Early Arrival**).

### Summary of Diagnostic Findings

1. **Actual vs. Predicted Plot (`actual_vs_predicted.png`):**
   - High alignment along the ideal $y=x$ reference line across 0.5 to 13 hours.
   - Minor dispersion increases only at journeys exceeding 8 hours.
2. **Residuals vs. Fitted Values (`residuals_vs_predicted.png`):**
   - Evenly centered around zero error line across the full prediction range.
   - Polynomial trend line is flat near zero, indicating absence of systematic structural curvature.
   - Mild heteroscedasticity: variance expands slightly for longer journeys due to cumulative traffic and environmental volatility.
3. **Residual Distribution & Q-Q Plot (`residual_distribution.png`, `residual_qq_plot.png`):**
   - Bell-shaped, symmetric distribution centered at zero (mean residual $\approx 0.001$ hrs).
   - Q-Q plot confirms normal residuals through the central quantiles, with mild divergence at extreme positive quantiles representing rare severe delays.
   - **Outlier Policy:** In accordance with academic best practice, **extreme positive residuals are NOT discarded**. Real-world traffic gridlocks and severe weather disruptions must be preserved to assess operational tail-risk.
4. **Slice-Based Residual Behavior:**
   - **Route Breakdown:** `Route_East` ($\text{MAE} = 0.414$ hrs) and `Route_West` ($\text{MAE} = 0.275$ hrs) show wider dispersion than `Route_Central` ($\text{MAE} = 0.101$ hrs), reflecting highway distance scaling.
   - **Weather Breakdown:** `Stormy` conditions produce $\text{MAE} = 0.442$ hrs ($\text{RMSE} = 0.652$ hrs), followed by `Snowy` ($\text{MAE} = 0.250$ hrs) and `Clear` ($\text{MAE} = 0.185$ hrs).
   - **Carrier Breakdown:** All carriers cluster tightly around zero bias (mean residuals between $-0.014$ and $+0.026$ hrs), showing no favoritism or provider-specific drift.
   - **Parcel Weight Breakdown:** Uniform MAE across weight categories (Light: 0.217 hrs, Medium: 0.248 hrs, Heavy: 0.243 hrs), confirming that weight acts as a minor linear loading factor rather than a source of transit volatility.

---

## 7. Feature Importance & Non-Causality Warning

Feature importance was extracted using **Permutation Importance** on the held-out test partition ($N=1,428$):

| Feature | Permutation Importance ($R^2$ Mean Drop) | Standard Deviation |
|:---|:---:|:---:|
| `distance_km` | **1.693** | $\pm$ 0.049 |
| `traffic_level` | **0.232** | $\pm$ 0.010 |
| `weather` | **0.160** | $\pm$ 0.009 |
| `route` | **0.017** | $\pm$ 0.001 |
| `carrier` | **0.014** | $\pm$ 0.002 |
| `parcel_weight_kg` | **0.005** | $\pm$ 0.001 |

### Key Drivers & Non-Causality Analysis
1. **Distance Dominance:** `distance_km` is by far the primary predictive driver, establishing physical transit limits.
2. **Speed Modulators:** `traffic_level` and `weather` act as key velocity retardants.
3. **Correlation vs. Causation Caution:** High predictive importance confirms strong statistical association, but **does NOT prove causation**. For example, carrier or route coefficients reflect logistical assignments and network geography rather than vendor capability. Attributing delivery delays directly to carrier fault would be statistically invalid.

---

## 8. Operational Recommendations for Distributor Management

1. **Dynamic SLA Quoting:** Replace static distance lookups with real-time inference via `predict_delivery_time()`, enabling precise arrival windows with ~13-minute average accuracy.
2. **Automated Weather Buffers:** Automatically apply a 30-to-45-minute contingency buffer to customer ETA notifications whenever severe weather alerts (`Snowy` or `Stormy`) are active along the transit corridor.
3. **Corridor-Specific Buffer Scheduling:** Dynamically scale dispatch lead times on longer corridors (`Route_East`, `Route_West`, `Route_North`) during peak congestion windows (`Jam`, `High`).
4. **Outlier Anomaly Auditing:** Establish an automated monitoring threshold where deliveries with residuals $> +1.5$ hours trigger supply chain incident reviews.

---

## 9. Artifact Manifest

The complete deliverable suite is organized as follows:

```text
RouteWise/
├── requirements.txt                              # Frozen environment dependencies
├── notebooks/
│   └── 04_regression_and_residuals.ipynb         # 32-cell fully executed narrative notebook
├── src/
│   ├── modeling.py                               # Modular ML and inference functions
│   └── visualization.py                          # Diagnostic plotting functions
├── tests/
│   ├── test_modeling.py                          # Automated unit test suite (12 tests, all passing)
│   ├── build_notebook.py                         # Deterministic notebook builder script
│   └── run_notebook.py                           # End-to-end headless notebook execution script
└── outputs/
    ├── figures/
    │   ├── actual_vs_predicted.png
    │   ├── residuals_vs_predicted.png
    │   ├── residual_distribution.png
    │   ├── residual_qq_plot.png
    │   ├── residuals_by_route.png
    │   ├── residuals_by_carrier.png
    │   ├── residuals_by_weather.png
    │   ├── residuals_by_weight_category.png
    │   ├── model_cv_comparison.png
    │   └── feature_importance.png
    ├── tables/
    │   ├── cv_model_comparison.csv & .md
    │   ├── test_set_evaluation.csv & .md
    │   ├── slice_residual_summary.md
    │   ├── feature_importance_permutation.csv
    │   └── feature_importance_primary.csv
    └── reports/
        └── predictive_modeling_summary.md
```
