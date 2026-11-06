"""
Builds and executes notebooks/03_statistical_analysis.ipynb using NotebookExecutor.
"""

from pathlib import Path
import nbformat as nbf
from runner import NotebookExecutor


def main():
    executor = NotebookExecutor()
    nb = nbf.v4.new_notebook()

    # Title & Metadata
    executor.add_markdown(nb, """# RouteWise: Supply Chain Delivery Analytics & Prediction
## Notebook 03: Statistical Analysis & Hypothesis Testing
**Academic Exploratory Data Science Project**  
**Role: Person 1 — Data Engineering, Data Cleaning, EDA & Statistical Analysis**  
**Branch:** `person-1-eda` | **Integration Contract:** `docs/data_contract.md`

---

### Notebook Objectives:
1. **Research Question 1 (Weather Impact on Transit Duration):**
   - Formulate formal null ($H_0$) and alternative ($H_1$) hypotheses.
   - Inspect normality (Shapiro-Wilk test, Q-Q distributions) and homoscedasticity (Levene's test).
   - Execute One-Way ANOVA (parametric) and Kruskal-Wallis H-Test (non-parametric).
   - Quantify effect sizes (Eta-squared $\\eta^2$, Epsilon-squared $\\epsilon^2$).
   - Conduct post-hoc pairwise comparisons with Bonferroni correction ($\alpha_{\\text{adj}} = 0.005$) and rank-biserial effect sizes.
2. **Research Question 2 (Parcel Weight Sensitivity):**
   - Formulate correlation hypotheses.
   - Compute Pearson linear correlation ($r$) with $95\\%$ Fisher-z confidence intervals, and Spearman rank correlation ($\\rho$).
   - Rigorously differentiate **statistical significance** from **practical significance** in large-sample logistics datasets.
3. **Implications for Person 2's Regression Modeling:** Synthesize empirical statistical conclusions into actionable feature selection guidance.""")

    # Setup Code
    executor.add_markdown(nb, """---
## 1. Setup, Environment & Cleaned Data Ingestion""")

    code_setup = """import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Ensure project root is in sys.path
PROJECT_ROOT = Path("..").resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.statistical_analysis import run_weather_hypothesis_tests, run_weight_correlation_analysis

# Seaborn & Matplotlib styling
sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams.update({
    "figure.autolayout": True,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150
})

data_path = PROJECT_ROOT / "data" / "processed" / "cleaned_delivery_data.csv"
print(f"Loading cleaned dataset from: {data_path}")
df = pd.read_csv(data_path)
print(f"Loaded Clean Records: {len(df):,}")
df.head()"""
    executor.add_code(nb, code_setup)

    # RQ1 Section
    executor.add_markdown(nb, """---
## 2. Research Question 1: Do Delivery Times Differ Significantly Across Weather Conditions?

### 2.1 Hypothesis Formulation
- **Null Hypothesis ($H_0$):** Delivery durations do not differ across weather conditions; the population means (or rank medians) across all five weather groups are identical:
  $$\\mu_{\\text{Clear}} = \\mu_{\\text{Rainy}} = \\mu_{\\text{Foggy}} = \\mu_{\\text{Snowy}} = \\mu_{\\text{Stormy}}$$
- **Alternative Hypothesis ($H_1$):** At least one weather condition exhibits a statistically significant difference in delivery duration:
  $$\\exists \\, i, j \\quad \\text{such that} \\quad \\mu_i \\neq \\mu_j$$
- **Significance Level:** $\\alpha = 0.05$ (two-tailed).""")

    executor.add_markdown(nb, """### 2.2 Descriptive Group Statistics by Weather Condition""")
    code_desc = """weather_groups = [group["delivery_time_hours"].values for _, group in df.groupby("weather")]
weather_names = list(df.groupby("weather").groups.keys())

desc_list = []
for name, vals in zip(weather_names, weather_groups):
    desc_list.append({
        "Weather Condition": name,
        "Sample Size (N)": len(vals),
        "Mean Duration (Hrs)": round(float(np.mean(vals)), 2),
        "Std Dev (Hrs)": round(float(np.std(vals, ddof=1)), 2),
        "Median Duration (Hrs)": round(float(np.median(vals)), 2),
        "IQR (Hrs)": round(float(stats.iqr(vals)), 2),
        "Skewness": round(float(stats.skew(vals)), 2)
    })
desc_df = pd.DataFrame(desc_list)
desc_df"""
    executor.add_code(nb, code_desc)

    executor.add_markdown(nb, """### 2.3 Statistical Assumption Diagnostics: Normality & Homoscedasticity

Before selecting the primary hypothesis test, we verify the core assumptions:
1. **Normality:** Assessed via Shapiro-Wilk test on representative group subsets and visual Q-Q plots.
2. **Homogeneity of Variance (Homoscedasticity):** Assessed via Levene's Test (centered at median).""")

    code_assumptions = """# 1. Levene's Test for Equality of Variances
levene_stat, levene_p = stats.levene(*weather_groups, center="median")
print(f"Levene's Test Statistic: {levene_stat:.4f}, p-value: {levene_p:.4e}")
if levene_p < 0.05:
    print("Conclusion: Reject null hypothesis of equal variances. Data is heteroscedastic.")
else:
    print("Conclusion: Fail to reject equal variances. Data is homoscedastic.")

# 2. Shapiro-Wilk Test for Normality
print("\\nShapiro-Wilk Normality Tests:")
for name, vals in zip(weather_names, weather_groups):
    sub = vals[:min(len(vals), 1000)]
    s_stat, s_p = stats.shapiro(sub)
    print(f" - {name:10s}: W = {s_stat:.4f}, p = {s_p:.4e} -> {'Non-Normal' if s_p < 0.05 else 'Normal'}")

# 3. Visual Q-Q Plots
fig, axes = plt.subplots(1, 5, figsize=(16, 3.5), sharey=True)
for ax, name, vals in zip(axes, weather_names, weather_groups):
    stats.probplot(vals, dist="norm", plot=ax)
    ax.set_title(f"Q-Q: {name}")
    ax.grid(True, linestyle=":", alpha=0.6)
plt.show()"""
    executor.add_code(nb, code_assumptions)

    executor.add_markdown(nb, """**Diagnostics Evaluation:**  
- **Homoscedasticity:** Levene's test yields $W = 58.75, p = 7.17 \\times 10^{-49} < 0.05$, confirming that variances across weather groups are significantly unequal (heteroscedastic).
- **Normality:** Shapiro-Wilk tests ($p < 0.001$) and Q-Q plots demonstrate moderate departures from normality, driven by multi-modal corridor distances and positive right-tail skewness.
- **Methodological Selection:** Because normality and homoscedasticity are violated, the non-parametric **Kruskal-Wallis H-test** is the methodologically superior test. However, given our large sample size ($N = 7,136$), One-Way ANOVA is also evaluated for comparison due to the Central Limit Theorem.""")

    executor.add_markdown(nb, """### 2.4 Omninus Hypothesis Tests & Effect Sizes (ANOVA & Kruskal-Wallis)""")
    code_tests = """weather_summary, pairwise_df, meta = run_weather_hypothesis_tests(df)
weather_summary"""
    executor.add_code(nb, code_tests)

    executor.add_markdown(nb, """### 2.5 Post-Hoc Pairwise Comparisons with Bonferroni Correction

With $k = 5$ weather groups, there are $m = \\frac{5 \\times 4}{2} = 10$ unique pairwise comparisons. To control the family-wise error rate (FWER) at $\\alpha = 0.05$, we apply the Bonferroni adjustment:
$$\\alpha_{\\text{adj}} = \\frac{0.05}{10} = 0.005$$
We report Mann-Whitney U test statistics, adjusted significance, and Rank-Biserial correlation effect sizes ($r = 1 - \\frac{2U}{n_1 n_2}$).""")

    code_pairwise = """pairwise_df"""
    executor.add_code(nb, code_pairwise)

    executor.add_markdown(nb, """**Interpretation of Research Question 1:**  
1. **Omnibus Rejection of $H_0$:** Both One-Way ANOVA ($F = 149.77, p = 3.37 \\times 10^{-123}$) and Kruskal-Wallis ($H = 429.95, p = 9.37 \\times 10^{-92}$) overwhelmingly reject the null hypothesis at $\\alpha = 0.05$.
2. **Substantial Effect Size:** $\\eta^2 = 0.0775$ and $\\epsilon^2 = 0.0597$ indicate a medium-to-large effect size in human-operated freight operations: weather condition alone accounts for approximately $6-8\\%$ of the global variance in delivery durations.
3. **Pairwise Disparities:** All 10 pairwise comparisons maintain statistical significance post-Bonferroni correction ($p < 0.005$). The largest operational delta exists between `Clear` and `Stormy` weather, where deliveries take an average of $1.71$ hours longer ($r = 0.473$), representing a massive $59\\%$ transit inflation.""")

    # RQ2 Section
    executor.add_markdown(nb, """---
## 3. Research Question 2: Relationship Between Parcel Weight and Delivery Time

### 3.1 Hypothesis Formulation
- **Null Hypothesis ($H_0$):** There is no correlation between parcel weight and delivery duration:
  $$\\rho_{\\text{weight, time}} = 0$$
- **Alternative Hypothesis ($H_1$):** There is a non-zero correlation between parcel weight and delivery duration:
  $$\\rho_{\\text{weight, time}} \\neq 0$$
- **Significance Level:** $\\alpha = 0.05$ (two-tailed).""")

    executor.add_markdown(nb, """### 3.2 Correlation Testing & Confidence Interval Estimation""")
    code_weight = """weight_corr_df = run_weight_correlation_analysis(df)
weight_corr_df"""
    executor.add_code(nb, code_weight)

    # Statistical vs Practical Significance Markdown
    executor.add_markdown(nb, """### 3.3 Statistical Significance vs. Practical Significance

A central tenet of rigorous Exploratory Data Science is distinguishing between **statistical significance** (the probability that an observed effect is not zero due to random sampling) and **practical significance** (the real-world magnitude and operational usefulness of that effect).

| Dimension | Observation in RouteWise Dataset | Scientific Interpretation |
|---|---|---|
| **Sample Size ($N$)** | $N = 7,136$ observations | Enormous statistical power; standard error of correlation is exceptionally small: $SE = \\frac{1}{\\sqrt{N-3}} \\approx 0.0118$. |
| **P-Value** | $p = 0.000203 < 0.05$ | **Statistically Significant:** We confidently reject $H_0: \\rho = 0$. The relationship is real and not a sampling artifact. |
| **Correlation Coefficient ($r$)** | $r = 0.0440$, $95\\%$ CI $[0.0208, 0.0671]$ | **Practically Negligible:** A correlation of $0.044$ is negligible according to standard behavioral and engineering benchmarks ($r < 0.10$). |
| **Coefficient of Determination ($R^2$)** | $R^2 = 0.0019$ ($0.19\\%$) | **No Explanatory Value:** Parcel weight explains less than $0.2\\%$ of the variance in delivery duration. Over $99.8\\%$ of variance is governed by distance, road traffic, and weather. |

#### Operational Supply Chain Conclusion:
While warehouse loading crews expend slightly more physical effort handling heavy freight parcels, this handling delay adds mere minutes to a multi-hour transit journey. Operations management should **not** restructure delivery schedules around parcel weights; dispatch optimization algorithms must focus almost exclusively on distance routing, carrier velocity, and weather/traffic avoidance.""")

    # Person 2 Handoff Markdown
    executor.add_markdown(nb, """---
## 4. Synthesis & Modeling Guidance for Person 2

Based on our exploratory, OLAP, and statistical testing findings, we provide the following recommendations to Person 2 for regression modeling:

1. **Primary Predictor Priority:**
   - **Critical Continuous Predictor:** `distance_km` ($r = 0.882$, accounts for $>77\\%$ direct variance).
   - **Key Categorical Predictors:** `traffic_level` (monotonic ordinal penalty), `weather` ($\eta^2 = 0.078$), `carrier` ($12-15\\%$ performance spread), and `route`.
   - **Low-Utility Predictor:** `parcel_weight_kg` ($R^2 = 0.19\\%$). Include for physical completeness, but anticipate small regression coefficients.
2. **Interaction Candidates:**
   - $\\text{distance\\_km} \\times \\text{traffic\\_level}$: The slope of delivery duration over distance steepens markedly under severe traffic.
   - $\\text{weather} \\times \\text{traffic\\_level}$: Multiplicative compound delays emerge during storm events on congested routes.
3. **Data Leakage Safeguards:**
   - `delayed` must be excluded from regression predictor sets (it is a deterministic target derivative).
   - `delivery_time` timestamp must be excluded (directly yields duration).
   - All categorical encodings and scalers must be fitted strictly inside training folds using `scikit-learn Pipeline`.""")

    target_path = Path("notebooks/03_statistical_analysis.ipynb")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Notebook 03 successfully generated and saved at: {target_path}")


if __name__ == "__main__":
    main()
