# RouteWise: Predictive Modeling & Residual Diagnostics Report
**Component:** Component 4 — Machine Learning & Predictive Analytics  
**Author / Contributor:** Person 2  
**Project:** RouteWise: Supply Chain Delivery Analytics & Prediction  
**Repository Branch:** `person-2-modeling`  
**Execution Timestamp:** 2026-11-06  

---

## 1. Executive Summary & Synthetic Data Disclosure

A regional freight distributor commissioned this exploratory data science initiative to determine the feasibility of predicting shipment delivery durations from pre-dispatch and environmental attributes. 

> **Synthetic Dataset Disclosure:** This project is developed and evaluated on a synthetic benchmark dataset generated for an exploratory data science curriculum. While the statistical workflows, cross-validation isolation, and residual evaluations follow rigorous academic standards, all findings demonstrate **prototype feasibility** rather than real-world logistics accuracy.

The primary analytical question addressed is:
> **"How accurately can delivery time be predicted using route, carrier, weather, parcel weight, distance, and traffic conditions?"**

### Core Findings
1. **Predictive Accuracy:** The prototype machine learning pipelines explain **approximately 96% of delivery duration variance ($R^2 \approx 0.960$)** under 5-fold cross-validation on training data, and achieve consistent performance ($R^2 = 0.960$) on an untouched 20% test partition ($N = 1,428$).
2. **Operational Error Bounds on Held-Out Test Set:**
   - **Mean Absolute Error (MAE):** **0.221 hours (~13.3 minutes)**.
   - **Root Mean Squared Error (RMSE):** **0.353 hours (~21.2 minutes)**.
   - **95th Percentile Absolute Error (P95):** **~0.638 hours (~38.3 minutes)** across the full test distribution.
3. **Algorithm Hierarchy & Experimental Selection:**
   - **HistGradientBoosting Regressor** achieved top cross-validated performance ($R^2 = 0.960 \pm 0.003$, $\text{MAE} = 0.234 \pm 0.006$ hrs, $\text{RMSE} = 0.358 \pm 0.018$ hrs) and was selected as the prototype candidate.
   - **Random Forest Regressor** demonstrated strong competitive performance ($R^2 = 0.948 \pm 0.003$, $\text{MAE} = 0.265 \pm 0.004$ hrs, $\text{RMSE} = 0.405 \pm 0.018$ hrs).
   - **Linear Regression** served as an interpretable baseline ($R^2 = 0.912 \pm 0.002$, $\text{MAE} = 0.365 \pm 0.005$ hrs), displaying structural bias due to unmodeled non-linear weather/traffic interaction terms.
   - **Dummy Baseline** failed completely ($R^2 = -0.001 \pm 0.001$, $\text{MAE} = 1.366 \pm 0.018$ hrs), validating the necessity of machine learning over central-tendency quoting.
4. **Key Diagnostic Insights:**
   - **Severe Weather Events:** `Stormy` conditions produce the highest prediction dispersion (Count: 76, $\text{MAE} = 0.442$ hrs, $\text{RMSE} = 0.652$ hrs, $\text{P95} = 1.453$ hrs). Deliveries with unrecorded weather (`Missing`, Count: 46) were imputed at the fold level and show $\text{MAE} = 0.500$ hrs.
   - **Corridor Dispersion:** Longer-distance routes (`Route_East` $\text{MAE} = 0.414$ hrs, `Route_West` $\text{MAE} = 0.275$ hrs, `Route_North` $\text{MAE} = 0.251$ hrs) display wider residual dispersion than compact urban routes (`Route_Central` $\text{MAE} = 0.101$ hrs).
   - **Carrier Reporting:** Residual means cluster tightly near zero ($-0.014$ to $+0.026$ hrs) with MAEs between 0.202h and 0.264h. These metrics do not establish operational neutrality or vendor competence, as carrier assignments correlate with assigned route distances.

---

## 2. Problem Formulation & Feature Architecture

### Target Variable
- **`delivery_time_hours`**: Continuous elapsed duration from dispatch to customer receipt (range: 0.43 to 13.17 hours, mean: 3.05 hours).

### Predictor Set
- **Continuous Numerical Predictors:**
  - `distance_km`: Journey distance in kilometers.
  - `parcel_weight_kg`: Weight of the consignment in kilograms (contains 178 intentional missing values in processed dataset; min = 0.2 kg, max = 40.0 kg).
- **Categorical Nominal Predictors:**
  - `route`: Transportation corridor (`Route_Central`, `Route_North`, `Route_South`, `Route_East`, `Route_West`).
  - `carrier`: Commercial courier partner (`ApexDelivery`, `FastShip`, `NationalExpress`, `QuickLogistics`).
  - `weather`: Meteorological condition at dispatch (`Clear`, `Rainy`, `Foggy`, `Snowy`, `Stormy` — contains 200 intentional missing values in processed dataset).
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
The primary dataset (`data/processed/modelling_delivery_data.csv`, 7,136 rows) is partitioned into an **80% training split (5,708 samples)** and a **20% held-out test split (1,428 samples)** using a deterministic seed (`random_state=42`) **before** any transformations or imputations are fitted.

### Scikit-Learn Pipeline Design & Leakage Prevention
All feature transformations are encapsulated inside a `ColumnTransformer` embedded in unified `Pipeline` objects:
- **Numerical Sub-pipeline:** `SimpleImputer(strategy='median')` $\rightarrow$ `StandardScaler()`.
- **Categorical Sub-pipeline:** `SimpleImputer(strategy='most_frequent')` $\rightarrow$ `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`.
- **In-Fold Imputation Isolation:** Because imputation is performed inside the pipeline prior to model fitting, every cross-validation fold fits its own imputation statistics strictly on in-fold training data. (Note: HistGradientBoosting does not utilize its native missing-value routing in this architecture, as missing values are handled explicitly by the pipeline's imputation stage).

---

## 4. Cross-Validation Benchmarking Results

Cross-validation was conducted with 5 folds on `X_train` (5,708 samples). Metrics are reported with their mean and standard deviation:

| Model | CV MAE (hours) | CV RMSE (hours) | CV MSE ($\text{hours}^2$) | CV $R^2$ |
|:---|:---:|:---:|:---:|:---:|
| **HistGradientBoosting** | **0.234 $\pm$ 0.006** | **0.358 $\pm$ 0.018** | **0.129 $\pm$ 0.013** | **0.960 $\pm$ 0.003** |
| **Random Forest** | 0.265 $\pm$ 0.004 | 0.405 $\pm$ 0.018 | 0.164 $\pm$ 0.015 | 0.948 $\pm$ 0.003 |
| **Linear Regression** | 0.365 $\pm$ 0.005 | 0.528 $\pm$ 0.014 | 0.279 $\pm$ 0.015 | 0.912 $\pm$ 0.002 |
| **Dummy Baseline** | 1.366 $\pm$ 0.018 | 1.785 $\pm$ 0.054 | 3.188 $\pm$ 0.192 | -0.001 $\pm$ 0.001 |

### Model Selection Decision
HistGradientBoosting achieves lower cross-validated MAE and higher $R^2$ on this synthetic benchmark. Both ensemble pipelines rely on identical preprocessing and fold isolation. HistGradientBoosting is advanced as the prototype model for test-set evaluation.

---

## 5. Held-Out Test Set Generalization

Evaluating the selected **HistGradientBoosting Regressor** on the completely untouched 20% test partition (1,428 shipments) shows high consistency with cross-validation:

| Metric | Test Set Score | Operational Meaning |
|:---|:---:|:---|
| **Test MAE** | **0.221 hours** | Average prediction deviation of **13.3 minutes** |
| **Test RMSE** | **0.353 hours** | Root mean square error of **21.2 minutes** |
| **Test MSE** | **0.125 $\text{hours}^2$** | Squared dispersion error |
| **Test $R^2$** | **0.960** | **96.0% of duration variance explained** on unseen synthetic shipments |

The close agreement between CV and test metrics suggests consistent generalization across partitions without evidence of fold leakage.

---

## 6. Comprehensive Residual Diagnostics

Residuals are calculated strictly as:
$$\text{Residual}_i = y_i - \hat{y}_i = \text{Actual Delivery Duration}_i - \text{Predicted Delivery Duration}_i$$

- **Positive Residual ($e_i > 0$):** Actual duration exceeded prediction (**Underprediction / Delivery Delay**).
- **Negative Residual ($e_i < 0$):** Actual duration was less than prediction (**Overprediction / Early Arrival**).

### Reconciled Slice Residual Diagnostics Summary ($N = 1,428$)

All slice breakdowns account for 100% of the held-out test observations without dropping missing values:

#### 1. Route Breakdown (Total = 1,428)
| route | Count | Mean_Residual | Std_Residual | MAE | RMSE | P95_Absolute_Error | Max_Overrun | Max_Underrun |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| `Route_East` | 227 | -0.016 | 0.598 | 0.414 | 0.597 | 1.254 | +2.862 | -1.843 |
| `Route_West` | 187 | +0.039 | 0.421 | 0.275 | 0.422 | 0.738 | +3.168 | -0.989 |
| `Route_North` | 340 | +0.005 | 0.363 | 0.251 | 0.362 | 0.642 | +2.507 | -1.866 |
| `Route_South` | 276 | +0.003 | 0.211 | 0.160 | 0.211 | 0.415 | +0.643 | -0.898 |
| `Route_Central` | 398 | -0.008 | 0.131 | 0.101 | 0.131 | 0.268 | +0.467 | -0.366 |

#### 2. Carrier Breakdown (Total = 1,428)
| carrier | Count | Mean_Residual | Std_Residual | MAE | RMSE | P95_Absolute_Error | Max_Overrun | Max_Underrun |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| `NationalExpress` | 302 | +0.026 | 0.440 | 0.264 | 0.440 | 0.844 | +3.168 | -1.205 |
| `QuickLogistics` | 252 | +0.021 | 0.355 | 0.223 | 0.355 | 0.631 | +2.089 | -1.866 |
| `ApexDelivery` | 462 | -0.014 | 0.322 | 0.208 | 0.322 | 0.619 | +1.940 | -1.843 |
| `FastShip` | 412 | -0.009 | 0.311 | 0.202 | 0.311 | 0.581 | +2.862 | -1.338 |

#### 3. Weather Breakdown (Total = 1,428)
| weather | Count | Mean_Residual | Std_Residual | MAE | RMSE | P95_Absolute_Error | Max_Overrun | Max_Underrun |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| `Missing` | 46 | +0.420 | 0.775 | 0.500 | 0.875 | 2.291 | +3.168 | -0.518 |
| `Stormy` | 76 | +0.023 | 0.656 | 0.442 | 0.652 | 1.453 | +2.298 | -1.668 |
| `Snowy` | 113 | +0.015 | 0.349 | 0.250 | 0.348 | 0.786 | +0.856 | -1.427 |
| `Foggy` | 149 | +0.007 | 0.318 | 0.216 | 0.317 | 0.556 | +0.845 | -1.843 |
| `Rainy` | 299 | -0.001 | 0.314 | 0.202 | 0.313 | 0.589 | +2.089 | -1.338 |
| `Clear` | 745 | -0.028 | 0.264 | 0.185 | 0.265 | 0.540 | +1.156 | -1.866 |

#### 4. Parcel Weight Category Breakdown (Total = 1,428)
| parcel_weight_category | Count | Mean_Residual | Std_Residual | MAE | RMSE | P95_Absolute_Error | Max_Overrun | Max_Underrun |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| `Medium (10-25kg)` | 180 | +0.014 | 0.395 | 0.248 | 0.394 | 0.761 | +2.089 | -1.427 |
| `Heavy (>25kg)` | 15 | +0.036 | 0.316 | 0.243 | 0.307 | 0.574 | +0.563 | -0.600 |
| `Light (<10kg)` | 1,191 | -0.001 | 0.349 | 0.217 | 0.349 | 0.629 | +3.168 | -1.866 |
| `Missing` | 42 | +0.036 | 0.286 | 0.210 | 0.285 | 0.513 | +0.845 | -0.504 |

---

## 7. Feature Importance & Non-Causality Warning

Feature importance was extracted using **Permutation Importance** on the held-out test partition ($N=1,428$, 10 repeats):

| Feature | Permutation Importance ($R^2$ Mean Drop) | Standard Deviation |
|:---|:---:|:---:|
| `distance_km` | **1.693** | $\pm$ 0.049 |
| `traffic_level` | **0.232** | $\pm$ 0.010 |
| `weather` | **0.160** | $\pm$ 0.009 |
| `route` | **0.017** | $\pm$ 0.001 |
| `carrier` | **0.014** | $\pm$ 0.002 |
| `parcel_weight_kg` | **0.005** | $\pm$ 0.001 |

### Interpretation & Methodological Context
- **Exploratory Tool:** Permutation feature importance is used here strictly as an exploratory diagnostic, not as a criteria for further model hyperparameter tuning.
- **Score Drop vs. Variance Share:** Permutation values represent the **mean reduction in test $R^2$** when the feature is randomly permuted, **not** percentage shares or additive proportions of explained variance.
- **Correlation vs. Causation Caution:** High predictive importance indicates statistical dependence within this synthetic dataset, but **does NOT prove causation**. For instance, carrier or route importances reflect logistical assignments rather than courier negligence or road defect.

---

## 8. Operational Inference & Unseen Category Handling

The inference function [`predict_delivery_time()`](file:///c:/Users/sbambore/Desktop/GitHub/EDS-project/src/modeling.py) provides safe prototype estimation:
- **Known Categories:** Computes fast point predictions without warnings.
- **Missing Values:** Gracefully handled via the pipeline's pre-fitted `SimpleImputer`.
- **Unseen Categories:** When encountering previously unobserved categories (e.g., a newly onboarded carrier), the pipeline zero-fills one-hot indicators via `handle_unknown='ignore'`. However, `predict_delivery_time()` explicitly issues a `UserWarning` alerting the caller that predictions with unobserved levels are unvalidated extrapolations and should not be assumed reliable solely because the encoder accepted them without error.

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
│   ├── test_modeling.py                          # Automated unit test suite (13 tests, all passing)
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
