"""
RouteWise: Supply Chain Delivery Analytics & Prediction
Person 2: Visualization and Model Diagnostics Module

This module provides diagnostic plotting routines for regression evaluation,
residual analysis, slice-based error inspections, cross-validation benchmarking,
and feature importance interpretation.
"""

from typing import Optional, List
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns

# Set professional styling for academic presentation
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
})


def _ensure_dir(filepath: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)


def plot_actual_vs_predicted(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Selected Model",
    save_path: Optional[str] = "outputs/figures/actual_vs_predicted.png"
) -> plt.Figure:
    """
    Plot actual vs. predicted delivery duration with an ideal y=x reference line.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(y_true, y_pred, alpha=0.55, color="#1f77b4", edgecolor="none", s=35, label="Deliveries")

    # Ideal identity line y = x
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], color="#d62728", linestyle="--", linewidth=2, label="Ideal (y = x)")

    r2 = 1 - np.sum((np.asarray(y_true) - y_pred) ** 2) / np.sum((np.asarray(y_true) - np.mean(y_true)) ** 2)
    rmse = np.sqrt(np.mean((np.asarray(y_true) - y_pred) ** 2))
    mae = np.mean(np.abs(np.asarray(y_true) - y_pred))

    ax.text(
        0.05, 0.92,
        f"$R^2 = {r2:.3f}$\n$RMSE = {rmse:.2f}$ hrs\n$MAE = {mae:.2f}$ hrs",
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#cccccc", alpha=0.9)
    )

    ax.set_xlabel("Actual Delivery Time (hours)")
    ax.set_ylabel("Predicted Delivery Time (hours)")
    ax.set_title(f"Actual vs. Predicted Delivery Time — {model_name}")
    ax.legend(loc="lower right")
    plt.tight_layout()

    if save_path:
        _ensure_dir(save_path)
        fig.savefig(save_path, dpi=300)
        print(f"[plot_actual_vs_predicted] Saved figure to {save_path}")

    return fig


def plot_residuals_vs_predicted(
    y_pred: np.ndarray,
    residuals: pd.Series | np.ndarray,
    model_name: str = "Selected Model",
    save_path: Optional[str] = "outputs/figures/residuals_vs_predicted.png"
) -> plt.Figure:
    """
    Plot residuals against predicted values to evaluate homoscedasticity and linearity.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    res_arr = np.asarray(residuals)
    ax.scatter(y_pred, res_arr, alpha=0.55, color="#2ca02c", edgecolor="none", s=35, label="Residuals")
    ax.axhline(0, color="#d62728", linestyle="--", linewidth=2, label="Zero Error Line")

    # Add smooth polynomial trend line to reveal any systematic non-linearity
    try:
        sort_idx = np.argsort(y_pred)
        x_sorted = y_pred[sort_idx]
        y_sorted = res_arr[sort_idx]
        poly_coeffs = np.polyfit(x_sorted, y_sorted, deg=2)
        poly_func = np.poly1d(poly_coeffs)
        x_trend = np.linspace(np.min(x_sorted), np.max(x_sorted), 100)
        ax.plot(x_trend, poly_func(x_trend), color="#ff7f0e", linewidth=2.2, label="Polynomial Trend")
    except Exception:
        pass

    ax.set_xlabel("Predicted Delivery Time (hours)")
    ax.set_ylabel("Residual: Actual - Predicted (hours)")
    ax.set_title(f"Residuals vs. Fitted Values — {model_name}")
    ax.legend(loc="upper right")
    plt.tight_layout()

    if save_path:
        _ensure_dir(save_path)
        fig.savefig(save_path, dpi=300)
        print(f"[plot_residuals_vs_predicted] Saved figure to {save_path}")

    return fig


def plot_residual_distribution(
    residuals: pd.Series | np.ndarray,
    model_name: str = "Selected Model",
    save_path: Optional[str] = "outputs/figures/residual_distribution.png"
) -> plt.Figure:
    """
    Plot residual frequency distribution with KDE and theoretical Gaussian curve overlay.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    res_arr = np.asarray(residuals)
    sns.histplot(res_arr, kde=True, stat="density", color="#3b528b", bins=30, ax=ax, label="Observed Residual KDE")

    # Overlay theoretical normal distribution
    mean_res = np.mean(res_arr)
    std_res = np.std(res_arr)
    x_axis = np.linspace(np.min(res_arr), np.max(res_arr), 200)
    normal_pdf = stats.norm.pdf(x_axis, mean_res, std_res)
    ax.plot(x_axis, normal_pdf, color="#d62728", linestyle="--", linewidth=2, label=rf"Normal Curve ($\mu={mean_res:.2f}, \sigma={std_res:.2f}$)")

    skew = stats.skew(res_arr)
    kurt = stats.kurtosis(res_arr)

    ax.text(
        0.05, 0.92,
        f"Mean: {mean_res:.3f} hrs\nStd Dev: {std_res:.3f} hrs\nSkewness: {skew:.3f}\nKurtosis: {kurt:.3f}",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#cccccc", alpha=0.9)
    )

    ax.axvline(0, color="black", linestyle=":", linewidth=1.5, alpha=0.7)
    ax.set_xlabel("Residual: Actual - Predicted (hours)")
    ax.set_ylabel("Density")
    ax.set_title(f"Residual Distribution & Normality — {model_name}")
    ax.legend(loc="upper right")
    plt.tight_layout()

    if save_path:
        _ensure_dir(save_path)
        fig.savefig(save_path, dpi=300)
        print(f"[plot_residual_distribution] Saved figure to {save_path}")

    return fig


def plot_residual_qq(
    residuals: pd.Series | np.ndarray,
    model_name: str = "Selected Model",
    save_path: Optional[str] = "outputs/figures/residual_qq_plot.png"
) -> plt.Figure:
    """
    Normal Q-Q (quantile-quantile) plot to inspect normality and heavy tails.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    res_arr = np.asarray(residuals)
    stats.probplot(res_arr, dist="norm", plot=ax)

    # Style plot
    ax.get_lines()[0].set_markerfacecolor("#1f77b4")
    ax.get_lines()[0].set_markeredgecolor("none")
    ax.get_lines()[0].set_markersize(5.0)
    ax.get_lines()[0].set_alpha(0.6)
    ax.get_lines()[1].set_color("#d62728")
    ax.get_lines()[1].set_linewidth(2)

    ax.set_title(f"Normal Q-Q Plot of Residuals — {model_name}")
    ax.set_xlabel("Theoretical Quantiles")
    ax.set_ylabel("Sample Quantiles (Residuals in hours)")
    plt.tight_layout()

    if save_path:
        _ensure_dir(save_path)
        fig.savefig(save_path, dpi=300)
        print(f"[plot_residual_qq] Saved figure to {save_path}")

    return fig


def plot_residuals_by_slice(
    df_eval: pd.DataFrame,
    slice_col: str,
    residual_col: str = "residual",
    model_name: str = "Selected Model",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Box plot of residuals stratified by operational slice (route, carrier, weather, weight).
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Calculate ordering by median residual or alphabetical
    order = sorted(df_eval[slice_col].dropna().unique())

    sns.boxplot(
        data=df_eval,
        x=slice_col,
        y=residual_col,
        order=order,
        palette="vlag",
        ax=ax,
        showmeans=True,
        meanprops={"marker": "D", "markerfacecolor": "red", "markeredgecolor": "red", "markersize": 5}
    )

    ax.axhline(0, color="#d62728", linestyle="--", linewidth=1.5, alpha=0.8)
    ax.set_title(f"Residual Distribution by {slice_col.replace('_', ' ').title()} — {model_name}")
    ax.set_xlabel(slice_col.replace("_", " ").title())
    ax.set_ylabel("Residual: Actual - Predicted (hours)")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()

    if save_path:
        _ensure_dir(save_path)
        fig.savefig(save_path, dpi=300)
        print(f"[plot_residuals_by_slice] Saved figure to {save_path}")

    return fig


def plot_cv_comparison(
    cv_comparison_df: pd.DataFrame,
    save_path: Optional[str] = "outputs/figures/model_cv_comparison.png"
) -> plt.Figure:
    """
    Plot cross-validation benchmark comparison across candidate models (MAE and RMSE).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    models = cv_comparison_df["Model"].tolist()
    mae_means = cv_comparison_df["raw_mae_mean"].tolist()
    mae_stds = cv_comparison_df["raw_mae_std"].tolist()
    rmse_means = cv_comparison_df["raw_rmse_mean"].tolist()
    rmse_stds = cv_comparison_df["raw_rmse_std"].tolist()
    r2_means = cv_comparison_df["raw_r2_mean"].tolist()

    colors = ["#7f7f7f", "#1f77b4", "#2ca02c", "#ff7f0e"][:len(models)]

    # Subplot 1: MAE & RMSE
    x = np.arange(len(models))
    width = 0.35

    ax1.bar(x - width/2, mae_means, width, yerr=mae_stds, label="CV MAE (hrs)", capsize=4, color="#4c72b0")
    ax1.bar(x + width/2, rmse_means, width, yerr=rmse_stds, label="CV RMSE (hrs)", capsize=4, color="#dd8452")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=20, ha="right")
    ax1.set_ylabel("Error (hours)")
    ax1.set_title("Cross-Validation Prediction Error (Lower is Better)")
    ax1.legend()

    # Subplot 2: R^2
    bars = ax2.bar(models, r2_means, color=colors, edgecolor="black", linewidth=0.5)
    ax2.set_ylabel("Coefficient of Determination ($R^2$)")
    ax2.set_title("Cross-Validation $R^2$ Score (Higher is Better)")
    ax2.set_xticklabels(models, rotation=20, ha="right")
    ax2.set_ylim([min(-0.05, min(r2_means) - 0.05), 1.05])

    for bar, val in zip(bars, r2_means):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            max(val + 0.02, 0.02),
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold"
        )

    plt.tight_layout()

    if save_path:
        _ensure_dir(save_path)
        fig.savefig(save_path, dpi=300)
        print(f"[plot_cv_comparison] Saved figure to {save_path}")

    return fig


def plot_feature_importance(
    importance_df: pd.DataFrame,
    title: str = "Feature Importance",
    save_path: Optional[str] = "outputs/figures/feature_importance.png"
) -> plt.Figure:
    """
    Horizontal bar chart for feature importances.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    feat_col = "feature_group" if "feature_group" in importance_df.columns else "feature"
    val_col = "importance" if "importance" in importance_df.columns else "importance_mean"

    df_sorted = importance_df.sort_values(by=val_col, ascending=True)

    bars = ax.barh(df_sorted[feat_col], df_sorted[val_col], color="#2b5c8f", edgecolor="none", height=0.6)
    ax.set_xlabel("Relative Importance Score")
    ax.set_title(title)

    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.005, bar.get_y() + bar.get_height() / 2, f"{width:.3f}", va="center", fontsize=9)

    plt.tight_layout()

    if save_path:
        _ensure_dir(save_path)
        fig.savefig(save_path, dpi=300)
        print(f"[plot_feature_importance] Saved figure to {save_path}")

    return fig
