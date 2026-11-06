"""
RouteWise: Supply Chain Delivery Analytics & Prediction
Person 2: Predictive Analytics & Machine Learning Module

This module implements regression problem formulation, feature preprocessing,
model pipelines, 5-fold cross-validation, test set evaluation, residual calculations,
feature importance extraction, and safe operational inference for delivery time prediction.
"""

from typing import Dict, List, Tuple, Any, Optional
import os
import warnings
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Feature Contract Definitions
# ---------------------------------------------------------------------------
TARGET_VARIABLE: str = "delivery_time_hours"

NUMERICAL_FEATURES: List[str] = [
    "parcel_weight_kg",
    "distance_km",
]

CATEGORICAL_FEATURES: List[str] = [
    "route",
    "carrier",
    "weather",
    "traffic_level",
]

EXCLUDED_COLUMNS: List[str] = [
    "delivery_id",
    "order_date",
    "dispatch_time",
    "delivery_time",
    "delayed",
    "origin",
    "destination",
]


def load_delivery_data(
    filepath: Optional[str] = None,
    primary_path: str = "data/processed/modelling_delivery_data.csv"
) -> pd.DataFrame:
    """
    Load the primary delivery modeling dataset strictly.

    Default: 'data/processed/modelling_delivery_data.csv' resolved relative to repository root.
    Preserves explicit 'filepath' overrides.
    Silent fallback to 'cleaned_delivery_data.csv' is strictly prohibited.
    Raises FileNotFoundError if the file does not exist.
    """
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    if filepath is not None:
        target_path = filepath if os.path.isabs(filepath) else os.path.abspath(os.path.join(repo_root, filepath))
    else:
        target_path = os.path.abspath(os.path.join(repo_root, primary_path))

    if not os.path.exists(target_path):
        raise FileNotFoundError(
            f"Primary modelling dataset not found at: '{target_path}'. "
            "Silent fallback to 'cleaned_delivery_data.csv' is disabled to prevent accidental evaluation on descriptive data."
        )

    df = pd.read_csv(target_path)
    print(f"[load_delivery_data] Successfully loaded {len(df)} records from: {target_path}")
    return df


def prepare_features(
    df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Extract predictor matrix X and target vector y according to the dataset contract.

    Excludes target leakage variables (delivery_time, delayed, etc.) and
    identifiers (delivery_id).
    """
    if TARGET_VARIABLE not in df.columns:
        raise KeyError(f"Target column '{TARGET_VARIABLE}' not found in DataFrame.")

    # Validate presence of expected features
    missing_num = [f for f in NUMERICAL_FEATURES if f not in df.columns]
    missing_cat = [f for f in CATEGORICAL_FEATURES if f not in df.columns]
    if missing_num or missing_cat:
        raise KeyError(
            f"Missing required predictors in dataset. Numerical missing: {missing_num}, "
            f"Categorical missing: {missing_cat}"
        )

    # Clean rows with missing target values (if any)
    valid_mask = df[TARGET_VARIABLE].notna()
    if (~valid_mask).sum() > 0:
        print(f"[prepare_features] Dropping {(~valid_mask).sum()} rows with missing target values.")
        df = df[valid_mask].copy()

    X = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES].copy()
    y = df[TARGET_VARIABLE].copy()

    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Perform an 80/20 train/test split with fixed random seed for reproducibility.
    """
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )


def build_preprocessor() -> ColumnTransformer:
    """
    Build a robust ColumnTransformer that:
    - Imputes and scales numerical features (StandardScaler).
    - Imputes and one-hot encodes categorical features (OneHotEncoder with handle_unknown='ignore').
    """
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, NUMERICAL_FEATURES),
            ("cat", cat_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return preprocessor


def create_model_pipeline(
    regressor: BaseEstimator,
    preprocessor: Optional[ColumnTransformer] = None
) -> Pipeline:
    """
    Wrap the preprocessor and estimator inside a Scikit-Learn Pipeline
    to strictly prevent data leakage across cross-validation folds.
    """
    if preprocessor is None:
        preprocessor = build_preprocessor()

    return Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", regressor),
    ])


def get_model_candidates(
    random_state: int = 42
) -> Dict[str, Pipeline]:
    """
    Instantiate the 4 candidate model pipelines:
    1. DummyRegressor (Baseline)
    2. Linear Regression (Interpretable baseline)
    3. Random Forest Regressor (Nonlinear ensemble)
    4. HistGradientBoostingRegressor (Gradient boosting comparison)
    """
    preprocessor = build_preprocessor()

    candidates = {
        "Dummy Baseline": create_model_pipeline(
            DummyRegressor(strategy="mean"),
            preprocessor=preprocessor
        ),
        "Linear Regression": create_model_pipeline(
            LinearRegression(),
            preprocessor=preprocessor
        ),
        "Random Forest": create_model_pipeline(
            RandomForestRegressor(
                n_estimators=100,
                max_depth=12,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=random_state,
                n_jobs=-1
            ),
            preprocessor=preprocessor
        ),
        "HistGradientBoosting": create_model_pipeline(
            HistGradientBoostingRegressor(
                max_iter=100,
                max_depth=6,
                learning_rate=0.08,
                random_state=random_state
            ),
            preprocessor=preprocessor
        ),
    }
    return candidates


def evaluate_cv(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_folds: int = 5,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Perform 5-fold cross-validation on the training dataset.

    Computes MAE, MSE, RMSE, and R2 across all folds.
    Returns individual fold scores, mean scores, and standard deviations.
    """
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    scoring = {
        "mae": "neg_mean_absolute_error",
        "mse": "neg_mean_squared_error",
        "r2": "r2",
    }

    scores = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        return_train_score=False,
        n_jobs=-1
    )

    mae_scores = -scores["test_mae"]
    mse_scores = -scores["test_mse"]
    rmse_scores = np.sqrt(mse_scores)
    r2_scores = scores["test_r2"]

    return {
        "mae_scores": mae_scores,
        "mse_scores": mse_scores,
        "rmse_scores": rmse_scores,
        "r2_scores": r2_scores,
        "mae_mean": float(np.mean(mae_scores)),
        "mae_std": float(np.std(mae_scores)),
        "mse_mean": float(np.mean(mse_scores)),
        "mse_std": float(np.std(mse_scores)),
        "rmse_mean": float(np.mean(rmse_scores)),
        "rmse_std": float(np.std(rmse_scores)),
        "r2_mean": float(np.mean(r2_scores)),
        "r2_std": float(np.std(r2_scores)),
    }


def compare_models_cv(
    candidates: Dict[str, Pipeline],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_folds: int = 5,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Run 5-fold cross-validation on all candidate models and compile
    a formatted comparison DataFrame with mean and std.
    """
    records = []
    for name, pipeline in candidates.items():
        res = evaluate_cv(pipeline, X_train, y_train, cv_folds=cv_folds, random_state=random_state)
        records.append({
            "Model": name,
            "CV MAE (hours)": f"{res['mae_mean']:.3f} +/- {res['mae_std']:.3f}",
            "CV RMSE (hours)": f"{res['rmse_mean']:.3f} +/- {res['rmse_std']:.3f}",
            "CV MSE (hours^2)": f"{res['mse_mean']:.3f} +/- {res['mse_std']:.3f}",
            "CV R2": f"{res['r2_mean']:.3f} +/- {res['r2_std']:.3f}",
            "raw_mae_mean": res["mae_mean"],
            "raw_rmse_mean": res["rmse_mean"],
            "raw_mse_mean": res["mse_mean"],
            "raw_r2_mean": res["r2_mean"],
            "raw_mae_std": res["mae_std"],
            "raw_rmse_std": res["rmse_std"],
            "raw_r2_std": res["r2_std"],
        })

    df_comp = pd.DataFrame(records)
    # Sort descending by R2, then ascending by MAE
    df_comp = df_comp.sort_values(by=["raw_r2_mean", "raw_mae_mean"], ascending=[False, True]).reset_index(drop=True)
    return df_comp


def evaluate_test_set(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Tuple[Dict[str, float], np.ndarray]:
    """
    Fit the pipeline on full training set and evaluate on untouched test set.

    Returns dictionary of test metrics (MAE, MSE, RMSE, R2) and predicted values.
    """
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    mae = float(mean_absolute_error(y_test, y_pred))
    mse = float(mean_squared_error(y_test, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_test, y_pred))

    metrics = {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
    }
    return metrics, y_pred


def compute_residuals(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray
) -> pd.Series:
    """
    Calculate residuals:
        Residual = Actual Delivery Time - Predicted Delivery Time

    Operational interpretation:
    - Positive residual (y_true > y_pred): actual delivery took longer than predicted (underpredicted).
    - Negative residual (y_true < y_pred): actual delivery took less time than predicted (overpredicted).
    """
    y_true_arr = np.asarray(y_true)
    residuals = y_true_arr - y_pred
    return pd.Series(residuals, index=getattr(y_true, "index", None), name="residual")


def analyze_residuals_by_slice(
    df_eval: pd.DataFrame,
    slice_col: str,
    residual_col: str = "residual"
) -> pd.DataFrame:
    """
    Compute residual summary statistics across categorical slices (e.g., route, carrier, weather).

    Residual Definition and Direction:
        Residual = Actual Delivery Time - Predicted Delivery Time (e_i = y_i - y_hat_i)
        - Positive Residual (e_i > 0): Actual duration exceeded prediction (Underprediction / Delivery Delay).
        - Negative Residual (e_i < 0): Actual duration was less than prediction (Overprediction / Early Arrival).

    Computed Statistics:
        - Count: Number of observations in this categorical group.
        - Mean_Residual: Signed mean residual (bias). Values near 0 indicate lack of systematic bias.
        - Std_Residual: Sample standard deviation of residuals (dispersion / uncertainty).
        - MAE: Mean Absolute Error (average magnitude of prediction error in hours).
        - RMSE: Root Mean Squared Error (penalizes larger errors).
        - P95_Absolute_Error: 95th percentile absolute error (practical upper bound for 95% of predictions).
        - Max_Overrun: Maximum positive residual (longest unexpected delay in hours).
        - Max_Underrun: Minimum negative residual (largest overprediction / earliest arrival in hours).
    """
    if slice_col not in df_eval.columns:
        raise KeyError(f"Slice column '{slice_col}' not found in evaluation DataFrame.")

    grouped = df_eval.groupby(slice_col, observed=False)[residual_col]
    summary = grouped.agg(
        Count="count",
        Mean_Residual=lambda r: float(np.mean(r)),
        Std_Residual=lambda r: float(np.std(r, ddof=1)) if len(r) > 1 else 0.0,
        MAE=lambda r: float(np.mean(np.abs(r))),
        RMSE=lambda r: float(np.sqrt(np.mean(r**2))),
        P95_Absolute_Error=lambda r: float(np.percentile(np.abs(r), 95)) if len(r) > 0 else 0.0,
        Max_Overrun=lambda r: float(np.max(r)) if len(r) > 0 else 0.0,
        Max_Underrun=lambda r: float(np.min(r)) if len(r) > 0 else 0.0,
    ).reset_index()

    return summary.sort_values(by="MAE", ascending=False).reset_index(drop=True)


def extract_feature_importance(
    fitted_pipeline: Pipeline,
    X_test: Optional[pd.DataFrame] = None,
    y_test: Optional[pd.Series] = None
) -> Dict[str, pd.DataFrame]:
    """
    Extract feature importance from a fitted pipeline (specifically RandomForestRegressor).
    Provides:
    1. Raw transformed feature importances (MDI).
    2. Grouped feature importances aggregated by original feature name.
    3. Permutation feature importance on test set (if X_test and y_test provided).
    """
    regressor = fitted_pipeline.named_steps["regressor"]
    preprocessor = fitted_pipeline.named_steps["preprocessor"]

    feature_names = preprocessor.get_feature_names_out()
    results = {}

    if hasattr(regressor, "feature_importances_"):
        mdi_importances = regressor.feature_importances_
        df_raw = pd.DataFrame({
            "feature": feature_names,
            "importance": mdi_importances
        }).sort_values(by="importance", ascending=False).reset_index(drop=True)
        results["raw_mdi"] = df_raw

        # Group one-hot encoded features back to original feature groups
        grouped_dict: Dict[str, float] = {}
        for feat, imp in zip(feature_names, mdi_importances):
            # Format: 'cat__route_Route 1' -> 'route', 'num__distance_km' -> 'distance_km'
            clean_name = feat.split("__")[-1]
            base_col = None
            for col in NUMERICAL_FEATURES + CATEGORICAL_FEATURES:
                if clean_name == col or clean_name.startswith(f"{col}_"):
                    base_col = col
                    break
            if base_col is None:
                base_col = clean_name
            grouped_dict[base_col] = grouped_dict.get(base_col, 0.0) + float(imp)

        df_grouped = pd.DataFrame([
            {"feature_group": k, "importance": v} for k, v in grouped_dict.items()
        ]).sort_values(by="importance", ascending=False).reset_index(drop=True)
        results["grouped_mdi"] = df_grouped

    if X_test is not None and y_test is not None:
        perm = permutation_importance(
            fitted_pipeline, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
        )
        df_perm = pd.DataFrame({
            "feature": X_test.columns,
            "importance_mean": perm.importances_mean,
            "importance_std": perm.importances_std,
        }).sort_values(by="importance_mean", ascending=False).reset_index(drop=True)
        results["permutation"] = df_perm

    # Define primary importance dataframe (permutation preferred for general estimators like HistGradientBoosting)
    if "permutation" in results:
        results["primary"] = results["permutation"]
    elif "grouped_mdi" in results:
        results["primary"] = results["grouped_mdi"]

    return results


def predict_delivery_time(
    pipeline: Pipeline,
    input_data: Dict[str, Any] | pd.DataFrame
) -> float | np.ndarray:
    """
    Operational inference function. Accepts a dictionary or DataFrame with:
    - route (str)
    - carrier (str)
    - weather (str)
    - parcel_weight_kg (float)
    - distance_km (float)
    - traffic_level (str)

    Safely handles unseen categorical levels using the pipeline's handle_unknown='ignore' configuration,
    while explicitly detecting and warning about unobserved categories.
    Returns predicted delivery duration in hours.
    """
    if isinstance(input_data, dict):
        df_input = pd.DataFrame([input_data])
        single_prediction = True
    else:
        df_input = input_data.copy()
        single_prediction = False

    # Validate required columns
    required = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    missing = [col for col in required if col not in df_input.columns]
    if missing:
        raise ValueError(f"Input is missing required features: {missing}")

    # Inspect fitted encoder to detect unseen categories
    unseen_warnings = {}
    try:
        preprocessor = pipeline.named_steps.get("preprocessor")
        if preprocessor is not None and hasattr(preprocessor, "named_transformers_"):
            cat_transformer = preprocessor.named_transformers_.get("cat")
            if cat_transformer is not None and hasattr(cat_transformer, "named_steps"):
                onehot = cat_transformer.named_steps.get("onehot")
                if onehot is not None and hasattr(onehot, "categories_"):
                    for i, col in enumerate(CATEGORICAL_FEATURES):
                        if col in df_input.columns:
                            known_cats = set(onehot.categories_[i])
                            vals = df_input[col].dropna().unique()
                            unseen = [v for v in vals if v not in known_cats]
                            if unseen:
                                unseen_warnings[col] = unseen
    except Exception:
        pass

    if unseen_warnings:
        warnings.warn(
            f"Input features contain unobserved categorical level(s): {unseen_warnings}. "
            "The OneHotEncoder step will zero-fill these indicators via handle_unknown='ignore'. "
            "Caution: Predictions with unseen categories are unvalidated extrapolations "
            "and should not be assumed reliable solely because the encoder processed them without error.",
            UserWarning,
            stacklevel=2,
        )

    preds = pipeline.predict(df_input[required])
    if single_prediction:
        return float(preds[0])
    return preds
