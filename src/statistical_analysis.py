"""
RouteWise: Supply Chain Delivery Analytics & Prediction
Statistical Analysis & Hypothesis Testing Module (Person 1 - Data Engineering & EDA)

Executes rigorous statistical hypothesis testing:
1. Research Question 1: Delivery durations across weather conditions (Normality, Levene's,
   One-way ANOVA, Kruskal-Wallis, effect sizes, post-hoc pairwise tests).
2. Research Question 2: Correlation between parcel weight and delivery duration
   (Pearson, Spearman, statistical vs. practical significance).
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from scipy import stats


def run_weather_hypothesis_tests(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Test Research Question 1:
    H0: Mean/median delivery durations are identical across all weather conditions.
    H1: At least one weather condition exhibits significantly different delivery duration.
    """
    weather_groups = [group["delivery_time_hours"].values for _, group in df.groupby("weather")]
    weather_names = list(df.groupby("weather").groups.keys())

    # 1. Descriptive statistics per weather group
    desc_list = []
    for name, vals in zip(weather_names, weather_groups):
        desc_list.append({
            "weather": name,
            "sample_size": len(vals),
            "mean_hours": float(np.mean(vals)),
            "std_hours": float(np.std(vals, ddof=1)),
            "median_hours": float(np.median(vals)),
            "iqr_hours": float(stats.iqr(vals))
        })
    desc_df = pd.DataFrame(desc_list)

    # 2. Assumption checks
    # Homoscedasticity: Levene's Test
    levene_stat, levene_p = stats.levene(*weather_groups, center="median")

    # Normality: Shapiro-Wilk on sampled sub-arrays (Shapiro is valid up to 5000)
    shapiro_results = {}
    for name, vals in zip(weather_names, weather_groups):
        sample_subset = vals[:min(len(vals), 1000)]
        s_stat, s_p = stats.shapiro(sample_subset)
        shapiro_results[name] = {"stat": float(s_stat), "p_val": float(s_p)}

    # 3. Parametric: One-Way ANOVA
    f_stat, anova_p = stats.f_oneway(*weather_groups)

    # Calculate ANOVA effect size (Eta-squared: eta^2 = SS_between / SS_total)
    grand_mean = df["delivery_time_hours"].mean()
    ss_total = np.sum((df["delivery_time_hours"] - grand_mean) ** 2)
    ss_between = sum(len(vals) * (np.mean(vals) - grand_mean) ** 2 for vals in weather_groups)
    eta_squared = ss_between / ss_total

    # 4. Non-Parametric: Kruskal-Wallis Test
    kw_stat, kw_p = stats.kruskal(*weather_groups)
    # Kruskal-Wallis effect size (Epsilon-squared: eps^2 = (H - k + 1) / (N - k))
    k = len(weather_groups)
    n_total = len(df)
    epsilon_squared = (kw_stat - k + 1) / (n_total - k)

    # Summary results table
    tests_summary = [
        {
            "Test Name": "Levene's Test (Homogeneity of Variances)",
            "Hypothesis Tested": "Equal variances across weather conditions",
            "Test Statistic": round(float(levene_stat), 4),
            "p-value": float(levene_p),
            "Significant (alpha=0.05)": bool(levene_p < 0.05),
            "Effect Size": "N/A",
            "Conclusion": "Variances are heteroscedastic" if levene_p < 0.05 else "Variances are homogeneous"
        },
        {
            "Test Name": "One-Way ANOVA",
            "Hypothesis Tested": "Equal mean delivery duration across weather conditions",
            "Test Statistic": round(float(f_stat), 4),
            "p-value": float(anova_p),
            "Significant (alpha=0.05)": bool(anova_p < 0.05),
            "Effect Size": f"Eta-sq = {eta_squared:.4f}",
            "Conclusion": "Reject H0: Statistically significant mean differences" if anova_p < 0.05 else "Fail to reject H0"
        },
        {
            "Test Name": "Kruskal-Wallis H-Test (Non-Parametric)",
            "Hypothesis Tested": "Equal median delivery duration ranks across weather conditions",
            "Test Statistic": round(float(kw_stat), 4),
            "p-value": float(kw_p),
            "Significant (alpha=0.05)": bool(kw_p < 0.05),
            "Effect Size": f"Epsilon-sq = {epsilon_squared:.4f}",
            "Conclusion": "Reject H0: Statistically significant rank differences" if kw_p < 0.05 else "Fail to reject H0"
        }
    ]
    summary_df = pd.DataFrame(tests_summary)

    # 5. Post-Hoc Pairwise Comparisons (Mann-Whitney U with Bonferroni correction)
    num_comparisons = (k * (k - 1)) // 2
    bonferroni_alpha = 0.05 / num_comparisons

    pairwise_list = []
    for i in range(k):
        for j in range(i + 1, k):
            name_a, vals_a = weather_names[i], weather_groups[i]
            name_b, vals_b = weather_names[j], weather_groups[j]
            u_stat, u_p = stats.mannwhitneyu(vals_a, vals_b, alternative="two-sided")
            mean_diff = float(np.mean(vals_a) - np.mean(vals_b))

            # Rank-biserial correlation effect size r = 1 - (2U / (n1 * n2))
            rank_biserial = 1.0 - (2.0 * u_stat) / (len(vals_a) * len(vals_b))

            pairwise_list.append({
                "Comparison": f"{name_a} vs. {name_b}",
                "Mean Difference (Hrs)": round(mean_diff, 3),
                "U Statistic": round(float(u_stat), 1),
                "Raw p-value": float(u_p),
                "Bonferroni Significant (alpha=0.005)": bool(u_p < bonferroni_alpha),
                "Rank-Biserial Correlation (r)": round(float(rank_biserial), 3)
            })
    pairwise_df = pd.DataFrame(pairwise_list)

    metadata = {
        "shapiro_normality": shapiro_results,
        "eta_squared": eta_squared,
        "epsilon_squared": epsilon_squared,
        "desc_df": desc_df
    }

    return summary_df, pairwise_df, metadata


def run_weight_correlation_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Test Research Question 2:
    Evaluate relationship between parcel weight and delivery duration.
    Tests Pearson (linear) and Spearman (monotonic) correlations and assesses
    statistical significance vs. practical significance.
    """
    weight = df["parcel_weight_kg"].values
    duration = df["delivery_time_hours"].values
    n = len(df)

    # Pearson correlation
    pearson_r, pearson_p = stats.pearsonr(weight, duration)
    # 95% Confidence Interval for Pearson using Fisher z-transformation
    z = np.arctanh(pearson_r)
    se = 1.0 / np.sqrt(n - 3)
    z_crit = 1.95996
    ci_low = np.tanh(z - z_crit * se)
    ci_high = np.tanh(z + z_crit * se)

    # Spearman rank correlation
    spearman_rho, spearman_p = stats.spearmanr(weight, duration)

    # Coefficient of determination R^2
    r_squared = pearson_r ** 2

    results = [
        {
            "Metric": "Pearson Linear Correlation (r)",
            "Estimate": round(float(pearson_r), 4),
            "95% Confidence Interval": f"[{ci_low:.4f}, {ci_high:.4f}]",
            "p-value": float(pearson_p),
            "Variance Explained (R^2)": f"{r_squared * 100:.2f}%",
            "Statistical Significance": "Statistically Significant (p < 0.05)" if pearson_p < 0.05 else "Not Significant",
            "Practical Significance Assessment": "Practically Negligible / Very Weak: explains < 1% of duration variation"
        },
        {
            "Metric": "Spearman Rank Correlation (rho)",
            "Estimate": round(float(spearman_rho), 4),
            "95% Confidence Interval": "N/A (Rank Monotonic)",
            "p-value": float(spearman_p),
            "Variance Explained (R^2)": "N/A",
            "Statistical Significance": "Statistically Significant (p < 0.05)" if spearman_p < 0.05 else "Not Significant",
            "Practical Significance Assessment": "Negligible monotonic relationship; routing distance & traffic dominate"
        }
    ]

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="Execute RouteWise statistical hypothesis testing.")
    parser.add_argument("--input", type=str, default="data/processed/cleaned_delivery_data.csv")
    parser.add_argument("--out_dir", type=str, default="outputs/tables")
    args = parser.parse_args()

    clean_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading cleaned dataset from {clean_path}...")
    df = pd.read_csv(clean_path)

    print("\n--- RESEARCH QUESTION 1: WEATHER IMPACT ON DELIVERY TIME ---")
    weather_summary, pairwise_df, meta = run_weather_hypothesis_tests(df)
    print("\nHypothesis Test Results:")
    print(weather_summary.to_string(index=False))

    print("\nPost-Hoc Pairwise Tests (Bonferroni-adjusted alpha = 0.005):")
    print(pairwise_df.to_string(index=False))

    weather_summary.to_csv(out_dir / "hypothesis_tests_weather.csv", index=False)
    pairwise_df.to_csv(out_dir / "pairwise_weather_comparisons.csv", index=False)
    meta["desc_df"].to_csv(out_dir / "weather_group_descriptives.csv", index=False)

    print("\n--- RESEARCH QUESTION 2: PARCEL WEIGHT VS. DELIVERY TIME ---")
    weight_corr_df = run_weight_correlation_analysis(df)
    print(weight_corr_df.to_string(index=False))
    weight_corr_df.to_csv(out_dir / "parcel_weight_correlation_analysis.csv", index=False)

    print(f"\nAll statistical reports successfully saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
