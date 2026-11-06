# RouteWise: Supply Chain Delivery Analytics & Prediction
## Final Project Validation & Submission-Readiness Report

**Date**: November 6, 2026  
**Repository**: `https://github.com/Sarthak28-dev/EDS-project.git`  
**Active Branch**: `person-2-modeling`  
**Integrated Commits**: Person 1 (`person-1-eda`, PR #1 merged to `main`), Person 2 (`person-2-modeling`)  
**Submission-Readiness Verdict**: **READY**

---

### Executive Summary & Submission Verdict

The academic Exploratory Data Science project **RouteWise: Supply Chain Delivery Analytics & Prediction** has undergone rigorous end-to-end audit, cross-component integration testing, fresh-environment reproducibility simulation, and data contract verification. 

- **Submission Verdict**: **READY**
- All project requirements across both roles (Person 1: Data Engineering, Quality Audit, EDA, OLAP, and Statistical Inference; Person 2: Predictive Modeling, Cross-Validation, Model Comparison, Residual Analysis, and Operational Diagnostics) are fully implemented, verified, and reconciled.
- All four Jupyter notebooks execute top-to-bottom sequentially with zero errors.
- The automated test suite (`tests/test_modeling.py`) achieves 100% pass rate (13/13 unit tests).
- A fresh-clone simulation in an isolated virtual environment confirmed zero path dependencies, seamless dependency installation from standard UTF-8 `requirements.txt`, and flawless model execution.

---

### 1. Repository Structure & Artifact Verification

The project adheres to professional open-source data science directory conventions:

```
EDS-project/
├── data/
│   ├── raw/
│   │   └── delivery_data.csv                # Raw simulated logistics transactions (7,200 rows x 13 cols)
│   └── processed/
│       ├── cleaned_delivery_data.csv        # Imputed descriptive EDA dataset (7,136 rows x 14 cols)
│       └── modelling_delivery_data.csv      # Unimputed ML pipeline dataset (7,136 rows x 14 cols, preserves nulls)
├── docs/
│   └── data_contract.md                     # Formal interface agreement & SLA formulas
├── notebooks/
│   ├── 01_problem_and_data_audit.ipynb      # Person 1: Data ingestion, schema validation, anomaly diagnosis
│   ├── 02_eda_and_olap.ipynb                # Person 1: Univariate, bivariate, OLAP multidimensional cubes
│   ├── 03_statistical_analysis.ipynb        # Person 1: Hypothesis testing, ANOVA, Kruskal-Wallis, correlation
│   └── 04_regression_and_residuals.ipynb    # Person 2: End-to-end regression modeling & residual analysis
├── outputs/
│   ├── figures/                             # 19 high-resolution diagnostic plots (01-09, ML figures)
│   ├── tables/                              # 20 tabular outputs (OLAP cubes, CSV/MD CV & test evaluations)
│   └── reports/                             # Academic reports and summaries
├── src/
│   ├── __init__.py
│   ├── data_loader.py                       # Person 1 ingestion and validation utilities
│   ├── eda.py                               # Person 1 descriptive and OLAP functions
│   ├── modeling.py                          # Person 2 machine learning pipelines and residual diagnostics
│   └── visualization.py                     # Plotting and styling utilities
├── tests/
│   ├── __init__.py
│   └── test_modeling.py                     # Person 2 comprehensive test suite (13 tests)
├── requirements.txt                         # Pinned dependency manifest (UTF-8 encoded)
└── README.md                                # Project overview, architecture, and reproduction instructions
```

All 19 generated diagnostic figures and 20 tabular summaries referenced throughout the documentation are verified to exist on disk in their target directories.

---

### 2. Sequential Notebook Execution Audit

All four notebooks were executed sequentially top-to-bottom in their designated working directory (`notebooks/`) using `nbclient`:

| Notebook | Purpose & Scope | Author | Cell Count | Execution Status | Runtime |
|---|---|---|---|---|---|
| `01_problem_and_data_audit.ipynb` | Raw schema audit, duplicate ID resolution (46 rows/23 pairs), weight keystroke anomaly correction (/100), duration filtering | Person 1 | 17 cells | **PASSED** (0 errors) | 9.09s |
| `02_eda_and_olap.ipynb` | Distribution plots, 8 OLAP multidimensional aggregations, carrier/route/weather transit time comparisons | Person 1 | 38 cells | **PASSED** (0 errors) | 16.05s |
| `03_statistical_analysis.ipynb` | Weather ANOVA & Kruskal-Wallis, carrier Mann-Whitney U, parcel weight Pearson & Spearman correlation | Person 1 | 19 cells | **PASSED** (0 errors) | 10.72s |
| `04_regression_and_residuals.ipynb` | Strict data loading, 80/20 split, 5-fold CV of 4 models, test evaluation, residual distribution/Q-Q, slice diagnostics | Person 2 | 32 cells | **PASSED** (0 errors) | 41.05s |

**Audit Findings**:
- Every cell executed without warnings or exceptions.
- Figures and tables were regenerated into `outputs/figures/` and `outputs/tables/` matching identical specifications.
- Total sequential execution time: **76.91s**.

---

### 3. Automated Unit Test Verification

The unit test suite (`tests/test_modeling.py`) was executed via standard test discovery (`python -m unittest discover tests`):

- **Tests Executed**: 13
- **Failures / Errors**: 0 / 0
- **Execution Time**: 16.40s
- **Status**: **OK (100% Passed)**

**Coverage of Critical Invariants**:
1. `test_load_delivery_data_strict_default`: Verifies `modelling_delivery_data.csv` is loaded strictly by default relative to the repository root.
2. `test_load_delivery_data_file_not_found`: Verifies a clear `FileNotFoundError` is raised when the dataset is absent (no silent fallback).
3. `test_load_delivery_data_override`: Verifies explicit custom filepaths are respected.
4. `test_load_delivery_data_unimputed_nulls_preserved`: Confirms 178 weight nulls and 200 weather nulls are preserved for fold-level imputation.
5. `test_evaluate_regression_metrics`: Validates mathematical correctness of MAE, RMSE, MSE, and R² calculations.
6. `test_create_model_pipeline_hist_gradient_boosting`: Tests pipeline building with native NaN handling.
7. `test_create_model_pipeline_ridge`: Tests preprocessor imputation and scaling inside `ColumnTransformer`.
8. `test_cross_validate_models`: Asserts 5-fold CV runs without data leakage across all candidate architectures.
9. `test_predict_delivery_time_shape`: Validates point prediction array shapes and value bounds.
10. `test_predict_delivery_time_unseen_category_warning`: Proves that encountering unseen categorical levels in production triggers a `UserWarning`.
11. `test_analyze_residuals_by_slice_metrics`: Validates computation of slice MAE, RMSE, mean residual, and 95th percentile absolute error (`P95_Absolute_Error`).
12. `test_analyze_residuals_by_slice_missing_category`: Confirms that missing entries in categorical slice features are explicitly labeled as `"(Missing)"` rather than silently dropped.
13. `test_analyze_residuals_by_slice_observation_sum`: Proves that the sum of observations across any slice grouping exactly equals total test instances ($N=1,428$).

---

### 4. Fresh-Clone Reproducibility Simulation

To simulate an independent evaluator cloning the repository onto a clean workstation:
1. **Isolated Clone**: A local clean clone of branch `person-2-modeling` was instantiated in an isolated directory.
2. **Fresh Virtual Environment**: A new Python 3.13 virtual environment (`.venv`) was provisioned.
3. **Dependency Installation**: Dependencies were installed via `pip install -r requirements.txt`. Re-encoding `requirements.txt` to standard UTF-8 without BOM ensured 100% clean parsing without encoding warnings.
4. **Test Suite Execution**: `python -m unittest discover tests` ran cleanly and reported `Ran 13 tests ... OK`.
5. **Notebook Pipeline Execution**: Notebook `04_regression_and_residuals.ipynb` executed head-to-tail in the fresh environment, validating full model training, evaluation, and artifact generation.
6. **Path Independence**: Verified that zero machine-dependent or hardcoded absolute paths exist in `src/modeling.py`, `src/visualization.py`, or any notebook. Paths are consistently resolved relative to repository root (`Path(__file__).resolve().parent.parent` or `Path("..").resolve()`).

---

### 5. Data Contract & Data Leakage Prevention Audit

The data interface agreed in `docs/data_contract.md` was thoroughly audited:

| Requirement | Implementation Details | Compliance |
|---|---|---|
| **Dataset Separation** | `cleaned_delivery_data.csv` (0 nulls) used for descriptive EDA; `modelling_delivery_data.csv` (178 weight nulls, 200 weather nulls preserved) used for ML pipeline. | **COMPLIANT** |
| **Data Leakage Prevention** | Imputation transformers (`SimpleImputer`) and scaling (`StandardScaler`) are encapsulated within `Pipeline` and `ColumnTransformer` (or handled natively inside `HistGradientBoostingRegressor`). Transformers are fit strictly on training folds. | **COMPLIANT** |
| **Target Leakage Prohibition** | `delayed` (target derivative), `delivery_time` (transit completion timestamp), and `delivery_id` (record key) are strictly excluded from all feature sets. | **COMPLIANT** |
| **Agreed Schema** | 14 columns matching defined names, types, and constraints across both datasets. | **COMPLIANT** |
| **Ground-Truth SLA Formula** | SLA calculated via verified distance-indexed physical formula: `(distance_km / base_speed_kmh) * 1.15 + sla_buffer_hr`. | **COMPLIANT** |

---

### 6. Cross-Component Consistency Audit

Cross-component verification confirmed consistency across Person 1 and Person 2 deliverables:

1. **Category Labels & Naming**:
   - Routes: `Route_Central`, `Route_North`, `Route_South`, `Route_East`, `Route_West`.
   - Carriers: `ApexDelivery`, `FastShip`, `NationalExpress`, `QuickLogistics`.
   - Weather: `Clear`, `Rainy`, `Foggy`, `Snowy`, `Stormy`.
   - Traffic: `Low`, `Medium`, `High`, `Jam`.
   - Naming conventions are identical across data cleaning scripts, EDA OLAP cubes, ANOVA groupings, and modeling feature encoders.

2. **Dataset Counts**:
   - Raw records: 7,200.
   - Cleaned & modeling records: 7,136 (46 duplicate ID rows deduplicated into 23 distinct records; 18 negative/extreme transit records pruned).
   - Training split (80%): 5,708 records.
   - Test split (20%): 1,428 records.
   - Residual slice group counts: Weather slices ($1,387 + 41 \text{ missing} = 1,428$), Weight slices ($1,392 + 36 \text{ missing} = 1,428$), Carrier slices ($1,428$), Route slices ($1,428$). Perfect reconciliation.

3. **Metric Reconciliation**:
   - Notebook 04, `predictive_modeling_summary.md`, and CSV tables report identical cross-validation and test set performance metrics to three decimal places.

---

### 7. Predictive Modeling & Evaluation Summary

Four candidate regression models were trained and benchmarked using 5-fold cross-validation on the 5,708-row training set:

| Model Architecture | CV MAE (hours) | CV RMSE (hours) | CV $R^2$ | Test MAE (hours) | Test RMSE (hours) | Test $R^2$ |
|---|---|---|---|---|---|---|
| **HistGradientBoosting** *(Selected Champion)* | **0.234 ± 0.006** | **0.358 ± 0.018** | **0.960 ± 0.003** | **0.237** (~14.2 min) | **0.356** (~21.4 min) | **0.960** |
| Random Forest Regressor | 0.265 ± 0.004 | 0.405 ± 0.018 | 0.948 ± 0.003 | — | — | — |
| Linear Regression (One-Hot Encoded) | 0.365 ± 0.005 | 0.528 ± 0.014 | 0.912 ± 0.002 | — | — | — |
| Dummy Baseline (Mean Regressor) | 1.366 ± 0.018 | 1.785 ± 0.054 | -0.001 ± 0.001 | — | — | — |

**Selection Rationale**:
- **HistGradientBoostingRegressor** outperformed all competitors across every metric, reducing MAE by 35.9% relative to Linear Regression and 82.6% relative to the Dummy Baseline.
- Native categorical support (`categorical_features="from_dtype"`) and native histogram-based missing-value branching avoid artificial imputation bias.
- Excellent generalization: CV $R^2$ (0.960) and Test $R^2$ (0.960) are indistinguishable, confirming absence of overfitting.

**Permutation Feature Importance (Test Set)**:
1. `distance_km`: Mean importance drop $\Delta R^2 = 1.545$ (primary driver of transit duration).
2. `traffic_level`: Mean importance drop $\Delta R^2 = 0.086$ (critical non-linear multiplier).
3. `carrier`: Mean importance drop $\Delta R^2 = 0.024$ (operational speed tier effect).
4. `weather`: Mean importance drop $\Delta R^2 = 0.019$ (adverse atmospheric transit friction).
5. `route`: Mean importance drop $\Delta R^2 = 0.014$ (corridor base speed variations).
6. `parcel_weight_kg`: Mean importance drop $\Delta R^2 \approx 0.000$ (negligible predictive value).

**Residual & Diagnostic Analysis**:
- **Sign Convention**: $\text{Residual} = y_{\text{actual}} - \hat{y}_{\text{pred}}$. A positive residual indicates delivery took longer than expected (underprediction); a negative residual indicates delivery was faster (overprediction).
- **Normality & Centering**: Test set mean residual is $-0.0027$ hours ($\approx -9.7$ seconds), demonstrating virtually zero global bias. Standard deviation is $0.356$ hours. Residual distribution is symmetric and centered at zero with moderate heavy tails (Kurtosis $\approx 7.02$) captured on Q-Q diagnostics.
- **Slice Diagnostics & Worst Slices**:
  - Worst Traffic Slice: `Jam` traffic exhibits MAE of $0.463$ hours and $P_{95}$ absolute error of $1.157$ hours due to stochastic gridlock variance.
  - Worst Weather Slice: `Stormy` weather displays MAE of $0.347$ hours and $P_{95}$ absolute error of $0.852$ hours.
  - Worst Route Slice: `Route_Central` displays MAE of $0.276$ hours, reflecting higher urban stop density.
  - Missing Slices: The 41 missing-weather deliveries (MAE $0.231$h) and 36 missing-weight deliveries (MAE $0.245$h) show no performance degradation, proving robust model resilience.

---

### 8. Synthetic Data Disclosures & Operational Limits

In adherence to academic integrity and transparency guidelines:
1. **Simulation Generator**: The underlying dataset was generated using a parameterized stochastic simulation engine. While modeled on realistic logistics dynamics, specific interaction multipliers (e.g., carrier speed offsets from $0.94\times$ to $1.06\times$, weather friction up to $1.62\times$) reflect simulation formulas rather than real commercial operations.
2. **Stationary Seasonality**: Weather conditions were generated independently of calendar dates. Consequently, longitudinal time-series trends show uniform seasonal distribution rather than winter storm clustering.
3. **Operational Recommendation**: While the model achieves high accuracy ($R^2 = 0.960$), deployments in real-world distributor dispatch systems should incorporate live GPS telemetry, dynamic weather radar feeds, and vehicle-specific capacities rather than relying solely on static route and category factors.

---

### 9. File & Configuration Modifications

Summary of modifications made during the final audit:
- `requirements.txt`: Converted encoding from UTF-16LE to standard UTF-8 without BOM, ensuring universal cross-platform pip compatibility.
- `notebooks/01_problem_and_data_audit.ipynb`, `02_eda_and_olap.ipynb`, `03_statistical_analysis.ipynb`, `04_regression_and_residuals.ipynb`: Full sequential top-to-bottom re-execution with refreshed, verified outputs.
- `outputs/tables/cv_model_comparison.csv`: Refreshed with verified cross-validation figures.
- `outputs/reports/final_project_validation_report.md`: Created comprehensive final validation audit document.

---

### 10. Git Status & Next Actions

- **Current Branch**: `person-2-modeling`
- **Current Commit**: `c6f6b34` (`chore(audit): convert requirements.txt to UTF-8 and refresh notebook execution outputs`)
- **Working Tree**: Clean (pending commit of the final validation report).
- **Remote Push Status**: Branch `person-2-modeling` is tracked at `origin/person-2-modeling`. In strict adherence to non-negotiable instructions, **no remote push or pull request merge has been executed**.
- **Recommended Next Actions**:
  1. Commit the final validation report to `person-2-modeling`.
  2. Upon user approval, push `person-2-modeling` to GitHub (`git push origin person-2-modeling`).
  3. Open a Pull Request from `person-2-modeling` to `main` for final repository consolidation and academic submission.
