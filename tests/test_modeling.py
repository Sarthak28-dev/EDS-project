"""
Unit and integration tests for RouteWise modeling module (Person 2).
"""

import unittest
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
        cls.df = load_delivery_data(fallback_fixture_path="data/processed/temp_delivery_data_fixture.csv")
        cls.X, cls.y = prepare_features(cls.df)
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = split_data(
            cls.X, cls.y, test_size=0.2, random_state=42
        )

    def test_dataset_loaded_and_schema_contract(self):
        self.assertGreater(len(self.df), 100)
        self.assertIn(TARGET_VARIABLE, self.df.columns)
        for num_col in NUMERICAL_FEATURES:
            self.assertIn(num_col, self.df.columns)
        for cat_col in CATEGORICAL_FEATURES:
            self.assertIn(cat_col, self.df.columns)

    def test_prepare_features_prevents_leakage(self):
        expected_cols = sorted(NUMERICAL_FEATURES + CATEGORICAL_FEATURES)
        self.assertListEqual(sorted(list(self.X.columns)), expected_cols)
        self.assertEqual(len(self.y), len(self.X))

        # Check that excluded leakage variables are strictly NOT in X
        for col in EXCLUDED_COLUMNS:
            self.assertNotIn(col, self.X.columns)
        self.assertNotIn(TARGET_VARIABLE, self.X.columns)

    def test_train_test_split_ratio(self):
        n_total = len(self.X)
        n_test = len(self.X_test)
        self.assertAlmostEqual(n_test / n_total, 0.2, delta=0.02)
        self.assertEqual(len(self.X_train), len(self.y_train))
        self.assertEqual(len(self.X_test), len(self.y_test))

    def test_candidate_models_instantiation(self):
        candidates = get_model_candidates(random_state=42)
        self.assertEqual(len(candidates), 4)
        self.assertIn("Dummy Baseline", candidates)
        self.assertIn("Linear Regression", candidates)
        self.assertIn("Random Forest", candidates)
        self.assertIn("HistGradientBoosting", candidates)

    def test_5fold_cross_validation(self):
        candidates = get_model_candidates(random_state=42)
        res = evaluate_cv(candidates["Linear Regression"], self.X_train, self.y_train, cv_folds=5)

        self.assertEqual(len(res["mae_scores"]), 5)
        self.assertEqual(len(res["rmse_scores"]), 5)
        self.assertGreater(res["mae_mean"], 0.0)
        self.assertGreater(res["rmse_mean"], 0.0)
        self.assertLessEqual(res["r2_mean"], 1.0)

    def test_compare_models_cv(self):
        candidates = get_model_candidates(random_state=42)
        comp_df = compare_models_cv(candidates, self.X_train, self.y_train, cv_folds=5)

        self.assertEqual(len(comp_df), 4)
        self.assertIn("Model", comp_df.columns)
        self.assertIn("CV MAE (hours)", comp_df.columns)
        self.assertIn("CV RMSE (hours)", comp_df.columns)
        self.assertIn("CV R2", comp_df.columns)

        # Baseline should have lower R2 than actual ML models
        dummy_r2 = comp_df.loc[comp_df["Model"] == "Dummy Baseline", "raw_r2_mean"].values[0]
        rf_r2 = comp_df.loc[comp_df["Model"] == "Random Forest", "raw_r2_mean"].values[0]
        self.assertGreater(rf_r2, dummy_r2)

    def test_test_set_evaluation(self):
        candidates = get_model_candidates(random_state=42)
        metrics, y_pred = evaluate_test_set(
            candidates["Random Forest"],
            self.X_train, self.y_train,
            self.X_test, self.y_test
        )

        self.assertIn("MAE", metrics)
        self.assertIn("MSE", metrics)
        self.assertIn("RMSE", metrics)
        self.assertIn("R2", metrics)
        self.assertEqual(len(y_pred), len(self.y_test))
        self.assertAlmostEqual(metrics["RMSE"], np.sqrt(metrics["MSE"]), places=4)

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
            candidates["Linear Regression"],
            self.X_train, self.y_train,
            self.X_test, self.y_test
        )
        df_eval = self.X_test.copy()
        df_eval["residual"] = compute_residuals(self.y_test, y_pred)

        route_slice = analyze_residuals_by_slice(df_eval, "route")
        self.assertIn("route", route_slice.columns)
        self.assertIn("MAE", route_slice.columns)
        self.assertIn("RMSE", route_slice.columns)
        self.assertIn("Mean_Residual", route_slice.columns)

    def test_feature_importance_extraction(self):
        rf_pipeline = get_model_candidates(random_state=42)["Random Forest"]
        rf_pipeline.fit(self.X_train, self.y_train)

        importances = extract_feature_importance(rf_pipeline, self.X_test, self.y_test)
        self.assertIn("raw_mdi", importances)
        self.assertIn("grouped_mdi", importances)
        self.assertIn("permutation", importances)

        # Sum of MDI importances should equal approximately 1.0
        self.assertAlmostEqual(float(importances["grouped_mdi"]["importance"].sum()), 1.0, places=3)

    def test_safe_prediction_function(self):
        pipeline = get_model_candidates(random_state=42)["Random Forest"]
        pipeline.fit(self.X_train, self.y_train)

        # Normal prediction input
        sample_input = {
            "route": "Route_A_NorthSouth",
            "carrier": "SwiftLog",
            "weather": "Clear",
            "parcel_weight_kg": 12.5,
            "distance_km": 120.0,
            "traffic_level": "Medium"
        }
        pred = predict_delivery_time(pipeline, sample_input)
        self.assertIsInstance(pred, float)
        self.assertGreater(pred, 0.0)

        # Edge case: novel unseen categorical levels should NOT throw an error
        unseen_input = {
            "route": "Completely_Unknown_Route",
            "carrier": "FutureCarrierX",
            "weather": "Tornado",
            "parcel_weight_kg": 15.0,
            "distance_km": 80.0,
            "traffic_level": "Gridlock"
        }
        pred_unseen = predict_delivery_time(pipeline, unseen_input)
        self.assertIsInstance(pred_unseen, float)
        self.assertGreater(pred_unseen, 0.0)

if __name__ == "__main__":
    unittest.main()
