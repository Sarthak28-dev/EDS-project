"""
Unit and integration tests for RouteWise modeling module (Person 2).
Audited and updated to validate against Person 1's authentic modelling_delivery_data.csv.
"""

import unittest
import warnings
import numpy as np
import pandas as pd
from src.modeling import (
    load_delivery_data,
    prepare_features,
    split_data,
    build_preprocessor,
    create_model_pipeline,
    get_model_candidates,
    evaluate_cv,
    compare_models_cv,
    evaluate_test_set,
    compute_residuals,
    analyze_residuals_by_slice,
    extract_feature_importance,
    predict_delivery_time,
    TARGET_VARIABLE,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    EXCLUDED_COLUMNS,
)

class TestRouteWiseModeling(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load authentic unimputed modelling dataset (7,136 records)
        cls.df = load_delivery_data(primary_path="data/processed/modelling_delivery_data.csv")
        cls.X, cls.y = prepare_features(cls.df)
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = split_data(
            cls.X, cls.y, test_size=0.2, random_state=42
        )

    def test_strict_dataset_loading_no_silent_fallback(self):
        # Calling with missing primary path must raise FileNotFoundError, NOT fallback to cleaned data
        with self.assertRaises(FileNotFoundError):
            load_delivery_data(primary_path="data/processed/nonexistent_modelling_file.csv")

        # Default call must load authentic modelling dataset with 7,136 records and intentional missingness
        df_default = load_delivery_data()
        self.assertEqual(len(df_default), 7136)
        self.assertGreater(df_default["parcel_weight_kg"].isna().sum(), 0)
        self.assertGreater(df_default["weather"].isna().sum(), 0)

    def test_dataset_loaded_and_schema_contract(self):
        # Verify row count matches Person 1's 7,136 records
        self.assertEqual(len(self.df), 7136)
        self.assertIn(TARGET_VARIABLE, self.df.columns)
        for num_col in NUMERICAL_FEATURES:
            self.assertIn(num_col, self.df.columns)
        for cat_col in CATEGORICAL_FEATURES:
            self.assertIn(cat_col, self.df.columns)

        # Verify intentional missingness in unimputed modelling dataset
        self.assertGreater(self.df["parcel_weight_kg"].isna().sum(), 0)
        self.assertGreater(self.df["weather"].isna().sum(), 0)

    def test_prepare_features_prevents_leakage(self):
        expected_cols = sorted(NUMERICAL_FEATURES + CATEGORICAL_FEATURES)
        self.assertListEqual(sorted(list(self.X.columns)), expected_cols)
        self.assertEqual(len(self.y), len(self.X))

        # Check that excluded leakage variables and identifiers are strictly absent from X
        for col in EXCLUDED_COLUMNS:
            self.assertNotIn(col, self.X.columns)
        self.assertNotIn(TARGET_VARIABLE, self.X.columns)

    def test_train_test_split_ratio(self):
        n_total = len(self.X)
        n_test = len(self.X_test)
        self.assertAlmostEqual(n_test / n_total, 0.2, delta=0.01)
        self.assertEqual(len(self.X_train), len(self.y_train))
        self.assertEqual(len(self.X_test), len(self.y_test))

    def test_pipeline_imputation_handles_missing_data(self):
        # Verify pipeline fits and predicts seamlessly despite nulls in X_train and X_test
        candidates = get_model_candidates(random_state=42)
        hgb_pipeline = candidates["HistGradientBoosting"]
        hgb_pipeline.fit(self.X_train, self.y_train)
        preds = hgb_pipeline.predict(self.X_test)
        self.assertEqual(len(preds), len(self.X_test))
        self.assertFalse(np.isnan(preds).any(), "Predictions contain NaN despite imputer in pipeline!")

    def test_candidate_models_instantiation(self):
        candidates = get_model_candidates(random_state=42)
        self.assertEqual(len(candidates), 4)
        self.assertIn("Dummy Baseline", candidates)
        self.assertIn("Linear Regression", candidates)
        self.assertIn("Random Forest", candidates)
        self.assertIn("HistGradientBoosting", candidates)

    def test_5fold_cross_validation(self):
        candidates = get_model_candidates(random_state=42)
        res = evaluate_cv(candidates["HistGradientBoosting"], self.X_train, self.y_train, cv_folds=5)

        self.assertEqual(len(res["mae_scores"]), 5)
        self.assertEqual(len(res["rmse_scores"]), 5)
        self.assertEqual(len(res["mse_scores"]), 5)
        self.assertEqual(len(res["r2_scores"]), 5)
        self.assertGreater(res["mae_mean"], 0.0)
        self.assertGreater(res["rmse_mean"], 0.0)
        self.assertGreater(res["r2_mean"], 0.90)

    def test_compare_models_cv_and_selection(self):
        candidates = get_model_candidates(random_state=42)
        comp_df = compare_models_cv(candidates, self.X_train, self.y_train, cv_folds=5)

        self.assertEqual(len(comp_df), 4)
        self.assertIn("Model", comp_df.columns)
        self.assertIn("CV MAE (hours)", comp_df.columns)
        self.assertIn("CV RMSE (hours)", comp_df.columns)
        self.assertIn("CV MSE (hours^2)", comp_df.columns)
        self.assertIn("CV R2", comp_df.columns)

        # Confirm HistGradientBoosting achieves superior CV performance
        hgb_r2 = comp_df.loc[comp_df["Model"] == "HistGradientBoosting", "raw_r2_mean"].values[0]
        rf_r2 = comp_df.loc[comp_df["Model"] == "Random Forest", "raw_r2_mean"].values[0]
        dummy_r2 = comp_df.loc[comp_df["Model"] == "Dummy Baseline", "raw_r2_mean"].values[0]

        self.assertGreater(hgb_r2, rf_r2)
        self.assertGreater(rf_r2, dummy_r2)

    def test_test_set_evaluation(self):
        candidates = get_model_candidates(random_state=42)
        metrics, y_pred = evaluate_test_set(
            candidates["HistGradientBoosting"],
            self.X_train, self.y_train,
            self.X_test, self.y_test
        )

        self.assertIn("MAE", metrics)
        self.assertIn("MSE", metrics)
        self.assertIn("RMSE", metrics)
        self.assertIn("R2", metrics)
        self.assertEqual(len(y_pred), len(self.y_test))
        self.assertAlmostEqual(metrics["RMSE"], np.sqrt(metrics["MSE"]), places=4)
        self.assertGreater(metrics["R2"], 0.95)
        self.assertLess(metrics["MAE"], 0.30)

    def test_residual_calculation_and_signs(self):
        y_true = np.array([5.0, 3.0, 8.0])
        y_pred = np.array([4.0, 3.5, 8.0])
        res = compute_residuals(y_true, y_pred)

        # Residual = Actual - Predicted
        # 5.0 - 4.0 = +1.0 (underpredicted, took longer than predicted)
        # 3.0 - 3.5 = -0.5 (overpredicted, took less time than predicted)
        # 8.0 - 8.0 = 0.0
        self.assertAlmostEqual(res.iloc[0], 1.0)
        self.assertAlmostEqual(res.iloc[1], -0.5)
        self.assertAlmostEqual(res.iloc[2], 0.0)

    def test_analyze_residuals_by_slice(self):
        candidates = get_model_candidates(random_state=42)
        metrics, y_pred = evaluate_test_set(
            candidates["HistGradientBoosting"],
            self.X_train, self.y_train,
            self.X_test, self.y_test
        )
        df_eval = self.X_test.copy()
        df_eval["residual"] = compute_residuals(self.y_test, y_pred)

        route_slice = analyze_residuals_by_slice(df_eval, "route")
        self.assertIn("route", route_slice.columns)
        self.assertIn("Count", route_slice.columns)
        self.assertIn("Mean_Residual", route_slice.columns)
        self.assertIn("Std_Residual", route_slice.columns)
        self.assertIn("MAE", route_slice.columns)
        self.assertIn("RMSE", route_slice.columns)
        self.assertIn("P95_Absolute_Error", route_slice.columns)
        self.assertIn("Max_Overrun", route_slice.columns)
        self.assertIn("Max_Underrun", route_slice.columns)
        self.assertGreaterEqual(route_slice["P95_Absolute_Error"].min(), 0.0)

    def test_feature_importance_extraction(self):
        hgb_pipeline = get_model_candidates(random_state=42)["HistGradientBoosting"]
        hgb_pipeline.fit(self.X_train, self.y_train)

        importances = extract_feature_importance(hgb_pipeline, self.X_test, self.y_test)
        self.assertIn("permutation", importances)
        self.assertIn("primary", importances)

        # Check distance is the top predictor
        top_feature = importances["permutation"].iloc[0]["feature"]
        self.assertEqual(top_feature, "distance_km")

    def test_safe_prediction_function(self):
        pipeline = get_model_candidates(random_state=42)["HistGradientBoosting"]
        pipeline.fit(self.X_train, self.y_train)

        # 1. Normal prediction input using authentic known categories should NOT emit warning
        sample_input = {
            "route": "Route_Central",
            "carrier": "ApexDelivery",
            "weather": "Clear",
            "parcel_weight_kg": 12.5,
            "distance_km": 85.0,
            "traffic_level": "Jam"
        }
        with warnings.catch_warnings(record=True) as recorded_warnings:
            warnings.simplefilter("always")
            pred = predict_delivery_time(pipeline, sample_input)
            self.assertEqual(len(recorded_warnings), 0, "Known categories unexpectedly triggered a warning!")
        self.assertIsInstance(pred, float)
        self.assertGreater(pred, 0.0)

        # 2. Edge case: novel unseen categorical levels should emit a UserWarning
        unseen_input = {
            "route": "Route_Galaxy",
            "carrier": "QuantumDrone",
            "weather": "Blizzard",
            "parcel_weight_kg": 15.0,
            "distance_km": 50.0,
            "traffic_level": "Gridlock"
        }
        with warnings.catch_warnings(record=True) as recorded_warnings:
            warnings.simplefilter("always")
            pred_unseen = predict_delivery_time(pipeline, unseen_input)
            self.assertGreaterEqual(len(recorded_warnings), 1)
            self.assertTrue(any(issubclass(w.category, UserWarning) for w in recorded_warnings))
            warning_msg = str(recorded_warnings[0].message)
            self.assertIn("unobserved categorical level", warning_msg)
        self.assertIsInstance(pred_unseen, float)
        self.assertGreater(pred_unseen, 0.0)

        # 3. Missing values in input dictionary should be gracefully imputed by pipeline without unseen warning
        missing_input = {
            "route": "Route_North",
            "carrier": "FastShip",
            "weather": np.nan,
            "parcel_weight_kg": np.nan,
            "distance_km": 110.0,
            "traffic_level": "High"
        }
        with warnings.catch_warnings(record=True) as recorded_warnings:
            warnings.simplefilter("always")
            pred_missing = predict_delivery_time(pipeline, missing_input)
            self.assertEqual(len(recorded_warnings), 0, "Missing values triggered an unintended unseen warning!")
        self.assertIsInstance(pred_missing, float)
        self.assertGreater(pred_missing, 0.0)

if __name__ == "__main__":
    unittest.main()
