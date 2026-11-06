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
1. **Predictive Accuracy:** Machine learning pipelines explain **93.1% to 95.4% of delivery duration variance ($R^2$)** under 5-fold cross-validation, and **93.3% on an untouched 20% test partition**.
2. **Operational Error Bounds:**
   - **Mean Absolute Error (MAE):** **0.68 to 0.73 hours (41 to 44 minutes)** on realistic multi-stop journeys ranging from 15 to 420 km.
   - **Root Mean Squared Error (RMSE):** **1.07 to 1.14 hours**, heavily penalizing occasional acute operational delays while remaining bounded.
3. **Algorithm Hierarchy:**
   - **HistGradientBoosting Regressor** achieved top cross-validated performance ($R^2 = 0.954 \pm 0.004$, $\text{MAE} = 0.557 \pm 0.021$ hrs).
   - **Random Forest Regressor** demonstrated strong competitive performance and high stability ($R^2 = 0.931 \pm 0.007$, $\text{MAE} = 0.685 \pm 0.031$ hrs).
   - **Linear Regression** served as an interpretable baseline ($R^2 = 0.895 \pm 0.012$, $\text{MAE} = 0.926 \pm 0.043$ hrs), but exhibited structural bias due to unmodeled non-linear weather/traffic interaction terms.
   - **Dummy Baseline** failed completely ($R^2 = -0.000$, $\text{MAE} = 3.218$ hrs), validating the necessity of machine learning over central-tendency quoting.
4. **Key Diagnostic Weaknesses:**
   - **Severe Weather Events:** `Stormy` conditions produce the highest prediction error ($\text{MAE} = 1.96$ hrs, mean overrun $+1.65$ hrs), followed by `Snowy` conditions ($\text{MAE} = 1.01$ hrs).
   - **Volatile Routes:** `Route_D_Mountain` ($\text{MAE} = 1.10$ hrs) and `Route_B_UrbanExpress` ($\text{MAE} = 0.77$ hrs) experience high transit friction compared to intercity highway links.

---

## 2. Problem Formulation & Feature Architecture

### Target Variable
- **`delivery_time_hours`**: Continuous elapsed duration from dispatch to customer receipt.

### Predictor Set
- **Continuous Numerical Predictors:**
  - `distance_km`: Journey distance in kilometers.
  - `parcel_weight_kg`: Weight of the consignment in kilograms.
- **Categorical Nominal Predictors:**
  - `route`: Specific transportation corridor (e.g., North-South, Urban Express, Coastway, Mountain, Midland Link).
  - `carrier`: Commercial courier partner (SwiftLog, ApexDelivery, MetroFreight, EcoHaul).
  - `weather`: Meteorological condition at dispatch (Clear, Rainy, Foggy, Snowy, Stormy).
  - `traffic_level`: Traffic congestion tier (Low, Medium, High, Severe).

### Target Leakage Prevention & Excluded Variables
The following columns are strictly excluded from the predictor matrix:
1. `delivery_id`: Arbitrary transaction identifier.
2. `delivery_time`: Exact delivery timestamp (directly reveals target $y = \text{delivery\_time} - \text{dispatch\_time}$).
3. `delayed`: Binary boolean indicator derived directly from `delivery_time_hours > expected_sla` (fatal target leakage).
4. `order_date` and `dispatch_time`: Excluded from static regression to prevent memorizing day-specific identifiers without physical causal justification.

---

## 3. Preprocessing & Validation Methodology

### Featurization Ordering
To comply with academic data science rigor, the dataset is partitioned into an **80% training split (960 samples)** and a **20% held-out test split (240 samples)** using a deterministic seed (`random_state=42`) **before** any transformation is fitted.

### Scikit-Learn Pipeline Design
All feature transformations are encapsulated inside a `ColumnTransformer` embedded in unified `Pipeline` objects:
- **Numerical Sub-pipeline:** `SimpleImputer(strategy='median')` $\rightarrow$ `StandardScaler()`.
- **Categorical Sub-pipeline:** `SimpleImputer(strategy='most_frequent')` $\rightarrow$ `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`.
- **Cross-Fold Isolation:** Because preprocessing is wrapped inside the Pipeline, transformations are independently fitted on in-fold training data during 5-fold cross-validation, guaranteeing zero leakage across folds.

---

## 4. Cross-Validation Benchmarking Results

Cross-validation was conducted with 5 folds on `X_train`. Metrics are reported with their mean and standard deviation:

| Model | CV MAE (hours) | CV RMSE (hours) | CV MSE ($\text{hours}^2$) | CV $R^2$ |
|:---|:---:|:---:|:---:|:---:|
| **HistGradientBoosting** | **0.557 $\pm$ 0.021** | **0.874 $\pm$ 0.021** | **0.765 $\pm$ 0.036** | **0.954 $\pm$ 0.004** |
| **Random Forest** | **0.685 $\pm$ 0.031** | **1.070 $\pm$ 0.029** | **1.146 $\pm$ 0.062** | **0.931 $\pm$ 0.007** |
| **Linear Regression** | 0.926 $\pm$ 0.043 | 1.321 $\pm$ 0.051 | 1.747 $\pm$ 0.134 | 0.895 $\pm$ 0.012 |
| **Dummy Baseline** | 3.218 $\pm$ 0.146 | 4.083 $\pm$ 0.135 | 16.687 $\pm$ 1.123 | -0.000 $\pm$ 0.000 |

### Metric Explanations in Logistics Terms
- **MAE (Mean Absolute Error):** Direct measure of average deviation in hours. Random Forest achieves an average absolute error of 41 minutes.
- **RMSE (Root Mean Squared Error):** Expressed in hours, penalizing large outliers. Because RMSE ($1.07$ hrs) moderately exceeds MAE ($0.68$ hrs), the error distribution has heavier tails than a pure Gaussian curve.
- **MSE (Mean Squared Error):** Variance of errors in squared hours ($\text{hrs}^2$).
- **$R^2$ (Coefficient of Determination):** Explains that over 93% of the variability in delivery times is captured by the 6 operational features.

---

## 5. Held-Out Test Set Generalization

Evaluating the selected **Random Forest Regressor** on the completely untouched 20% test partition confirms absence of overfitting:

| Metric | Test Set Score | Operational Meaning |
|:---|:---:|:---|
| **Test MAE** | **0.730 hours** | Average prediction deviation of **43.8 minutes** |
| **Test RMSE** | **1.136 hours** | Root mean square error of **68.2 minutes** |
| **Test MSE** | **1.290 $\text{hours}^2$** | Squared dispersion error |
| **Test $R^2$** | **0.933** | **93.3% of duration variance explained** on unseen shipments |

---

## 6. Comprehensive Residual Diagnostics

Residuals are calculated strictly as:
$$\text{Residual} = \text{Actual Delivery Duration} - \text{Predicted Delivery Duration}$$

- **Positive Residual:** Actual duration exceeded prediction (**Underprediction / Delivery Delay**).
- **Negative Residual:** Actual duration was less than prediction (**Overprediction / Early Arrival**).

### Summary of Diagnostic Findings

1. **Actual vs. Predicted Plot (`actual_vs_predicted.png`):**
   - High correlation along the ideal $y=x$ reference line across 1 to 14 hours.
   - Minor broadening of dispersion for trips $>8$ hours.
2. **Residuals vs. Fitted Values (`residuals_vs_predicted.png`):**
   - Centered around zero with no structural polynomial curvature.
   - Funnel-shaped heteroscedasticity: longer journeys exhibit wider variance due to cumulative en-route variance.
3. **Residual Distribution & Q-Q Plot (`residual_distribution.png`, `residual_qq_plot.png`):**
   - Unimodal, bell-shaped distribution with mild positive skewness and moderate leptokurtosis (fat right tail).
   - The right tail represents rare acute disruptions (severe accidents, breakdowns).
   - **Outlier Policy:** In accordance with academic best practice, **extreme positive residuals are NOT discarded**. Erasing them would hide critical operational supply chain risks from distributors.
4. **Slice-Based Residual Behavior:**
   - **Route Breakdown:** `Route_D_Mountain` has an average overrun of $+0.92$ hours and MAE of $1.10$ hours due to elevation slowdowns. `Route_B_UrbanExpress` has an MAE of $0.77$ hours due to urban signal density.
   - **Weather Breakdown:** `Stormy` weather increases average delivery delay by $+1.65$ hours ($\text{MAE} = 1.96$ hrs). `Snowy` weather increases delay by $+0.71$ hours ($\text{MAE} = 1.01$ hrs).
   - **Carrier Breakdown:** Residual means for all carriers remain near zero ($-0.19$ to $+0.60$ hrs). Differences in dispersion stem from route difficulty and weather exposure rather than carrier negligence.
   - **Parcel Weight Breakdown:** Parcel weight exhibits relatively stable error distributions, confirming that weight affects handling time additively without destabilizing transit predictions.

---

## 7. Feature Importance & Non-Causality Warning

### Feature Hierarchy
1. **`distance_km` (54.2%):** Primary physical determinant of transit duration.
2. **`traffic_level` (22.8%):** Primary speed modulator.
3. **`route` (11.4%):** Captures corridor topography, infrastructure, and speed limits.
4. **`weather` (7.6%):** Atmospheric friction and hazard factor.
5. **`parcel_weight_kg` (2.4%):** Depot loading and unloading overhead.
6. **`carrier` (1.6%):** Vendor dispatch and scheduling variance.

### Academic Distinction: Correlation vs. Causation
High feature importance confirms strong predictive association, but **does NOT prove causation**:
- A carrier exhibiting higher average duration or variance is frequently assigned to difficult corridors (mountainous routes or storm-prone zones).
- Attributing delays to carrier incompetence based solely on regression coefficients would be statistically invalid.

---

## 8. Operational Recommendations for Distributor Management

1. **Dynamic ETA Quoting:** Replace static distance-based SLA matrices with real-time inference from `predict_delivery_time()`.
2. **Automated Weather Buffers:** Apply a mandatory $+45$ to $+60$ minute contingency buffer to customer arrival windows when dispatching into `Snowy` or `Stormy` conditions.
3. **Mountain Route Dispatch Rules:** Increase base transit time allocations by $25\%$ on `Route_D_Mountain` to prevent routine SLA breaches.
4. **Outlier Anomaly Auditing:** Establish an automated monitoring threshold where deliveries with residuals $> +2.0$ hours trigger supply chain incident investigations.

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
│   ├── test_modeling.py                          # Automated unit test suite (11 tests, all passing)
│   ├── create_test_fixture.py                    # Contract-compliant test data fixture generator
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
    │   ├── feature_importance_mdi.csv
    │   └── feature_importance_permutation.csv
    └── reports/
        └── predictive_modeling_summary.md
```
