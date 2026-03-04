"""
Publication-Quality Visualizations for BINN-HPV.

Generates all figures: performance curves, SHAP plots, PPI networks,
and the biological cascade showing how gene expression → PPIs → Reactome
pathways → HPV status prediction.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import warnings
from typing import Any

# Some notebook runtimes export MPLBACKEND=module://matplotlib_inline.backend_inline.
# In non-notebook/headless jobs this backend may be unavailable, which breaks import.
if os.environ.get("MPLBACKEND", "").strip() == "module://matplotlib_inline.backend_inline":
    os.environ["MPLBACKEND"] = "Agg"

import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.path import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

try:
    from adjustText import adjust_text
    _ADJUST_TEXT = True
except ImportError:
    _ADJUST_TEXT = False

# ── Global constants ───────────────────────────────────────────────────────────

MODEL_DISPLAY = {
    "binn": "BINN",
    "svm_rbf": "SVM-RBF",
    "knn": "KNN",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}
MODEL_ORDER = ["binn", "svm_rbf", "knn", "random_forest", "xgboost"]

MODEL_COLORS = {
    "binn":          "#2E86AB",
    "svm_rbf":       "#A23B72",
    "knn":           "#F18F01",
    "random_forest": "#C73E1D",
    "xgboost":       "#3B1F2B",
}

# Diverging red-blue: high SHAP (HPV+) → red; low/negative (HPV−) → blue
SHAP_CMAP = LinearSegmentedColormap.from_list(
    "shap_rdb", ["#2166AC", "#F7F7F7", "#D6604D"], N=256
)


# ── Style ─────────────────────────────────────────────────────────────────────

def setup_plot_style() -> None:
    """Set publication-quality matplotlib defaults."""
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "figure.figsize": (8, 6),
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
        "axes.linewidth": 1.2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
    })


setup_plot_style()


# ── I/O helpers ───────────────────────────────────────────────────────────────

def _save_fig(fig: plt.Figure, base_path: str, formats: tuple = ("png", "pdf")) -> None:
    """Save figure to base_path.{fmt} for each format. Creates parent dir."""
    os.makedirs(os.path.dirname(os.path.abspath(base_path)), exist_ok=True)
    for fmt in formats:
        fig.savefig(f"{base_path}.{fmt}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved → {base_path}.{{{'|'.join(formats)}}}")


def _disp(name: str) -> str:
    return MODEL_DISPLAY.get(name, name)


def _color(name: str) -> str:
    return MODEL_COLORS.get(name, "#555555")


def _trunc(s: str, n: int = 45) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


# ── Geometry helpers ───────────────────────────────────────────────────────────

def _normalise_pos(
    pos: dict[str, tuple[float, float]],
    pad: float = 0.08,
) -> dict[str, tuple[float, float]]:
    """Normalise arbitrary node coordinates into [pad, 1-pad]^2."""
    if not pos:
        return {}
    xs = np.array([p[0] for p in pos.values()], dtype=float)
    ys = np.array([p[1] for p in pos.values()], dtype=float)
    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())
    x_rng = max(x_max - x_min, 1e-9)
    y_rng = max(y_max - y_min, 1e-9)
    out: dict[str, tuple[float, float]] = {}
    for node, (x, y) in pos.items():
        xn = pad + (1 - 2 * pad) * ((float(x) - x_min) / x_rng)
        yn = pad + (1 - 2 * pad) * ((float(y) - y_min) / y_rng)
        out[node] = (xn, yn)
    return out


def _resolve_node_overlaps(
    pos: dict[str, tuple[float, float]],
    radii: dict[str, float],
    bounds: tuple[float, float, float, float],
    *,
    gap: float = 0.001,
    iterations: int = 400,
    anchor_strength: float = 0.05,
) -> dict[str, tuple[float, float]]:
    """
    Iteratively separate circular nodes until they no longer overlap.
    Bounds are `(x_min, x_max, y_min, y_max)` in the same coordinate system.
    """
    if len(pos) <= 1:
        return dict(pos)

    nodes = list(pos.keys())
    arr = np.array([pos[n] for n in nodes], dtype=float)
    anchor = arr.copy()
    rad = np.array([max(float(radii.get(n, 0.0)), 1e-9) for n in nodes], dtype=float)

    x_min, x_max, y_min, y_max = bounds
    for _ in range(max(iterations, 1)):
        total_push = 0.0
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                dx = arr[j, 0] - arr[i, 0]
                dy = arr[j, 1] - arr[i, 1]
                dist = float(np.hypot(dx, dy))
                min_dist = float(rad[i] + rad[j] + gap)
                if dist >= min_dist:
                    continue
                if dist < 1e-9:
                    theta = ((i * 137 + j * 91) % 360) * np.pi / 180.0
                    ux, uy = float(np.cos(theta)), float(np.sin(theta))
                else:
                    ux, uy = dx / dist, dy / dist
                shift = 0.5 * (min_dist - dist)
                arr[i, 0] -= ux * shift
                arr[i, 1] -= uy * shift
                arr[j, 0] += ux * shift
                arr[j, 1] += uy * shift
                total_push += shift

        # Keep close to original layout while preserving separation.
        arr += (anchor - arr) * anchor_strength

        # Clamp inside bounds while respecting node radius.
        for k in range(len(nodes)):
            rk = rad[k]
            arr[k, 0] = float(np.clip(arr[k, 0], x_min + rk, x_max - rk))
            arr[k, 1] = float(np.clip(arr[k, 1], y_min + rk, y_max - rk))

        if total_push < 1e-8:
            break

    return {n: (float(arr[i, 0]), float(arr[i, 1])) for i, n in enumerate(nodes)}


def _count_node_overlaps(
    pos: dict[str, tuple[float, float]],
    radii: dict[str, float],
    *,
    gap: float = 0.0,
) -> int:
    """Return number of overlapping node pairs."""
    nodes = list(pos.keys())
    n_overlaps = 0
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            ni, nj = nodes[i], nodes[j]
            xi, yi = pos[ni]
            xj, yj = pos[nj]
            d = float(np.hypot(xj - xi, yj - yi))
            min_d = float(radii.get(ni, 0.0) + radii.get(nj, 0.0) + gap)
            if d < min_d:
                n_overlaps += 1
    return n_overlaps


def _resolve_ppi_schema(
    ppi_df: pd.DataFrame,
) -> tuple[str, str, str, str | None, str | None]:
    """
    Return `(gene_a_col, gene_b_col, additive_col, shap_a_col, shap_b_col)`.
    Raises ValueError if required columns are missing.
    """
    if {"gene_1", "gene_2"}.issubset(ppi_df.columns):
        g1_col, g2_col = "gene_1", "gene_2"
    elif {"gene_a", "gene_b"}.issubset(ppi_df.columns):
        g1_col, g2_col = "gene_a", "gene_b"
    else:
        raise ValueError("PPI DataFrame missing gene endpoint columns")

    if "ppi_importance_add" in ppi_df.columns:
        imp_col = "ppi_importance_add"
    elif "additive" in ppi_df.columns:
        imp_col = "additive"
    else:
        raise ValueError("PPI DataFrame missing additive importance column")

    if {"shap_gene_1", "shap_gene_2"}.issubset(ppi_df.columns):
        s1_col, s2_col = "shap_gene_1", "shap_gene_2"
    elif {"shap_a", "shap_b"}.issubset(ppi_df.columns):
        s1_col, s2_col = "shap_a", "shap_b"
    else:
        s1_col = s2_col = None
    return g1_col, g2_col, imp_col, s1_col, s2_col


# ── Curve helpers ──────────────────────────────────────────────────────────────

def _interp(x_raw, y_raw, grid):
    """Interpolate (x_raw, y_raw) onto `grid`; sorts by x first."""
    order = np.argsort(x_raw)
    return np.interp(grid, np.array(x_raw)[order], np.array(y_raw)[order])


def _mean_std_curves(metrics_list, x_key, y_key, n_pts=200):
    """
    Interpolate per-fold curves onto a common grid and return mean ± std.

    Returns
    -------
    grid, mean_y, std_y — all shape (n_pts,)
    """
    grid = np.linspace(0, 1, n_pts)
    curves = [_interp(m[x_key], m[y_key], grid) for m in metrics_list
              if x_key in m and y_key in m]
    if not curves:
        return grid, np.zeros(n_pts), np.zeros(n_pts)
    arr = np.stack(curves, axis=0)
    return grid, arr.mean(0), arr.std(0)


def _fold_metrics_for(all_metrics: list[dict], model_name: str) -> list[dict]:
    return [m for m in all_metrics if m.get("model") == model_name]


# ── Shared subplot builder for curve figures ──────────────────────────────────

def _curve_figure_2pan(
    all_fold_metrics: list[dict],
    x_key: str, y_key: str,
    xlabel: str, ylabel: str,
    auc_key: str,
    title_left: str,
    title_right: str,
    diagonal: bool = False,
    invert_x: bool = False,
    no_skill_y: float | None = None,
    save_path: str | None = None,
) -> None:
    """
    Generic 2-panel curve figure (left: BINN per-fold, right: model comparison).
    """
    if not all_fold_metrics:
        log.warning("No fold metrics available — skipping curve plot.")
        return

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 6), sharey=False)

    binn_metrics = _fold_metrics_for(all_fold_metrics, "binn")
    grid, mean_y, std_y = _mean_std_curves(binn_metrics, x_key, y_key)
    mean_auc = np.mean([m[auc_key] for m in binn_metrics]) if binn_metrics else 0.0
    std_auc  = np.std([m[auc_key] for m in binn_metrics]) if binn_metrics else 0.0

    # ── Left panel: BINN per-fold ─────────────────────────────────────────────
    c = _color("binn")
    for m in binn_metrics:
        ax_l.plot(m[x_key], m[y_key], color=c, lw=0.8, alpha=0.3)
    ax_l.plot(grid, mean_y, color=c, lw=2.5,
              label=f"BINN (mean={mean_auc:.3f}±{std_auc:.3f})")
    ax_l.fill_between(grid,
                      np.clip(mean_y - std_y, 0, 1),
                      np.clip(mean_y + std_y, 0, 1),
                      color=c, alpha=0.15)
    if diagonal:
        ax_l.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
    if no_skill_y is not None:
        ax_l.axhline(no_skill_y, color="gray", lw=0.8, ls=":", label=f"No skill ({no_skill_y:.2f})")

    ax_l.set_xlabel(xlabel); ax_l.set_ylabel(ylabel)
    ax_l.set_xlim(0, 1);     ax_l.set_ylim(0, 1)
    ax_l.set_title(title_left)
    handles_l, _ = ax_l.get_legend_handles_labels()
    if handles_l:
        ax_l.legend(loc="lower right" if not invert_x else "lower left")
    ax_l.grid(alpha=0.2)
    if invert_x:
        ax_l.invert_xaxis()

    # ── Right panel: all models ───────────────────────────────────────────────
    for mname in MODEL_ORDER:
        mets = _fold_metrics_for(all_fold_metrics, mname)
        if not mets:
            continue
        g, my, sy = _mean_std_curves(mets, x_key, y_key)
        auc_vals = [m[auc_key] for m in mets if auc_key in m]
        mu  = float(np.mean(auc_vals)) if auc_vals else 0.0
        std = float(np.std(auc_vals))  if auc_vals else 0.0
        col = _color(mname)
        ax_r.plot(g, my, color=col, lw=2.2,
                  label=f"{_disp(mname)} ({mu:.3f}±{std:.3f})")
        ax_r.fill_between(g, np.clip(my - sy, 0, 1), np.clip(my + sy, 0, 1),
                          color=col, alpha=0.08)

    if diagonal:
        ax_r.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5, label="Random")
    if no_skill_y is not None:
        ax_r.axhline(no_skill_y, color="gray", lw=0.8, ls=":", label="No skill")

    ax_r.set_xlabel(xlabel); ax_r.set_ylabel(ylabel)
    ax_r.set_xlim(0, 1);     ax_r.set_ylim(0, 1)
    ax_r.set_title(title_right)
    handles_r, _ = ax_r.get_legend_handles_labels()
    if handles_r:
        ax_r.legend(loc="lower right" if not invert_x else "lower left", fontsize=9)
    ax_r.grid(alpha=0.2)
    if invert_x:
        ax_r.invert_xaxis()

    fig.tight_layout()
    if save_path:
        _save_fig(fig, save_path)
    else:
        plt.show()


# ── 1. ROC curves ─────────────────────────────────────────────────────────────

def plot_roc_curves(all_fold_metrics: list[dict], save_path: str | None = None) -> None:
    _curve_figure_2pan(
        all_fold_metrics,
        x_key="fpr_array", y_key="tpr_array",
        xlabel="False Positive Rate", ylabel="True Positive Rate (Sensitivity)",
        auc_key="auroc",
        title_left="ROC — BINN per fold",
        title_right="ROC — Model Comparison",
        diagonal=True,
        save_path=save_path,
    )


# ── 2. Precision-Recall curves ────────────────────────────────────────────────

def plot_precision_recall_curves(
    all_fold_metrics: list[dict],
    save_path: str | None = None,
) -> None:
    # Positive-class prevalence from first fold
    first = next((m for m in all_fold_metrics if "y_true" in m), None)
    no_skill = float(np.mean(first["y_true"])) if first else None

    _curve_figure_2pan(
        all_fold_metrics,
        x_key="recall_array", y_key="precision_array",
        xlabel="Recall", ylabel="Precision",
        auc_key="auprc",
        title_left="PR Curve — BINN per fold",
        title_right="PR Curve — Model Comparison",
        diagonal=False,
        no_skill_y=no_skill,
        save_path=save_path,
    )


# ── 3. Specificity-Sensitivity curves ─────────────────────────────────────────

def plot_specificity_sensitivity_curves(
    all_fold_metrics: list[dict],
    save_path: str | None = None,
) -> None:
    _curve_figure_2pan(
        all_fold_metrics,
        x_key="specificity_array", y_key="sensitivity_array",
        xlabel="Specificity (TNR)", ylabel="Sensitivity (TPR)",
        auc_key="auroc",
        title_left="Spec-Sens — BINN per fold",
        title_right="Spec-Sens — Model Comparison",
        diagonal=True,
        invert_x=True,
        save_path=save_path,
    )


# ── 4. Confusion matrices ─────────────────────────────────────────────────────

def _draw_cm(ax, cm_arr, title, acc=None, sens=None, spec=None):
    labels = ["HPV−", "HPV+"]
    sns.heatmap(
        cm_arr, annot=True, fmt=".0f", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        linewidths=0.5, linecolor="gray",
        ax=ax, cbar=False,
    )
    ax.set_title(title, fontsize=11, pad=6)
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("Actual", fontsize=10)
    if acc is not None:
        ax.text(0.5, -0.20,
                f"Acc={acc:.2f}  Sens={sens:.2f}  Spec={spec:.2f}",
                ha="center", va="top", transform=ax.transAxes, fontsize=9)


def plot_confusion_matrices(
    all_fold_metrics: list[dict],
    save_path: str | None = None,
) -> None:
    if not all_fold_metrics:
        log.warning("No fold metrics available — skipping confusion matrices.")
        return
    models = [m for m in MODEL_ORDER if any(x["model"] == m for x in all_fold_metrics)]
    n = len(models)
    if n == 0:
        log.warning("No known models found in metrics — skipping confusion matrices.")
        return
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 4.5))
    if n == 1:
        axes = [axes]

    for ax, mname in zip(axes, models):
        mets = _fold_metrics_for(all_fold_metrics, mname)
        cm_total = np.zeros((2, 2), dtype=float)
        for m in mets:
            cm_total += np.array(m["confusion_matrix"], dtype=float)
        acc  = np.mean([m["accuracy"]    for m in mets])
        sens = np.mean([m["recall"]      for m in mets])
        spec = np.mean([m["specificity"] for m in mets])
        _draw_cm(ax, cm_total, _disp(mname), acc, sens, spec)

    fig.suptitle("Confusion Matrices (Summed Across Folds)", fontsize=13, y=1.02)
    fig.tight_layout()
    if save_path:
        _save_fig(fig, save_path)
    else:
        plt.show()

    # Also per-fold BINN confusion matrices
    binn = _fold_metrics_for(all_fold_metrics, "binn")
    if not binn:
        return
    ncols = len(binn)
    fig2, axes2 = plt.subplots(1, ncols, figsize=(3.2 * ncols, 4.5))
    if ncols == 1:
        axes2 = [axes2]
    for ax, m in zip(axes2, binn):
        cm = np.array(m["confusion_matrix"], dtype=float)
        _draw_cm(ax, cm, f"BINN fold {m['fold']}",
                 m["accuracy"], m["recall"], m["specificity"])
    fig2.suptitle("BINN Confusion Matrix — Per Fold", fontsize=13, y=1.02)
    fig2.tight_layout()
    if save_path:
        _save_fig(fig2, save_path.replace("confusion_matrices", "binn_confusion_per_fold"))
    else:
        plt.show()


# ── 5. SHAP beeswarm ──────────────────────────────────────────────────────────

def plot_shap_beeswarm(
    shap_values: np.ndarray,
    gene_names: list[str],
    X_test: np.ndarray | None = None,
    top_n: int = 30,
    save_path: str | None = None,
) -> None:
    if shap_values is None:
        log.warning("SHAP values missing — skipping beeswarm plot.")
        return
    shap_values = np.asarray(shap_values)
    if shap_values.ndim != 2 or shap_values.size == 0:
        log.warning("SHAP values must be a non-empty 2D array — skipping beeswarm plot.")
        return
    _, n_features = shap_values.shape
    if n_features == 0:
        log.warning("SHAP values have zero features — skipping beeswarm plot.")
        return
    mean_abs = np.abs(shap_values).mean(0)
    top_n = max(1, min(top_n, n_features))
    top_idx  = np.argsort(mean_abs)[::-1][:top_n]
    # Bottom of chart = most important
    plot_order = top_idx[::-1]
    X_test_arr = np.asarray(X_test) if X_test is not None else None

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35 + 2)))
    rng = np.random.default_rng(0)

    for rank, gi in enumerate(plot_order):
        sv = shap_values[:, gi]
        jitter = rng.uniform(-0.28, 0.28, len(sv))
        y = rank + jitter

        if X_test_arr is not None and X_test_arr.ndim == 2 and X_test_arr.shape[1] > gi:
            fv = X_test_arr[:, gi]
            mn, mx = fv.min(), fv.max()
            norm = (fv - mn) / (mx - mn + 1e-9)
            c = SHAP_CMAP(norm)
        else:
            c = _color("binn")

        ax.scatter(sv, y, c=c, s=14, alpha=0.75, zorder=5, linewidths=0)

    ax.axvline(0, color="black", lw=0.9, ls="--", alpha=0.6)
    ax.set_yticks(range(len(plot_order)))
    labels = []
    for i in plot_order:
        if 0 <= i < len(gene_names):
            labels.append(gene_names[i])
        else:
            labels.append(f"feature_{i}")
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("SHAP Value (impact on model output)")
    ax.set_title("Gene Contributions to HPV Status Prediction")
    ax.grid(axis="x", alpha=0.2)

    sm = plt.cm.ScalarMappable(cmap=SHAP_CMAP, norm=Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.015, pad=0.02)
    cbar.set_label("Feature value (expression)", fontsize=10)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["Low", "High"])

    fig.tight_layout()
    if save_path:
        _save_fig(fig, save_path)
    else:
        plt.show()


# ── 6. SHAP bar chart ─────────────────────────────────────────────────────────

def plot_shap_bar(
    gene_shap_df: pd.DataFrame,
    top_n: int = 30,
    save_path: str | None = None,
) -> None:
    """Horizontal bar chart of gene importance with cross-fold std error bars."""
    if gene_shap_df is None or gene_shap_df.empty:
        log.warning("Gene SHAP DataFrame is empty — skipping SHAP bar plot.")
        return
    if "mean_abs_shap" not in gene_shap_df.columns:
        log.warning("Gene SHAP DataFrame missing 'mean_abs_shap' — skipping SHAP bar plot.")
        return
    df = gene_shap_df.sort_values("mean_abs_shap", ascending=False).head(top_n).iloc[::-1]
    n = len(df)
    if n == 0:
        log.warning("No SHAP rows available after filtering — skipping SHAP bar plot.")
        return

    fig, ax = plt.subplots(figsize=(9, max(5, n * 0.33 + 2)))

    # Color by stability (low std = saturated, high std = lighter)
    if "shap_std" in df.columns:
        max_std = df["shap_std"].max() + 1e-9
        alphas = 1.0 - 0.55 * (df["shap_std"].values / max_std)
    else:
        alphas = np.ones(n)

    bars = ax.barh(range(n), df["mean_abs_shap"].values,
                   color=[plt.cm.Blues(0.55 + 0.35 * a) for a in alphas],
                   edgecolor="white", linewidth=0.4)

    if "shap_std" in df.columns:
        ax.errorbar(
            df["mean_abs_shap"].values, range(n),
            xerr=df["shap_std"].values,
            fmt="none", color="#333333", lw=1.2, capsize=3,
        )

    ax.set_yticks(range(n))
    ax.set_yticklabels(df["gene"].values if "gene" in df.columns else df.index, fontsize=10)
    ax.set_xlabel("Mean |SHAP| (across folds)")
    ax.set_title("Gene Feature Importance (SHAP)")
    ax.grid(axis="x", alpha=0.25)
    ax.set_xlim(left=0)

    fig.tight_layout()
    if save_path:
        _save_fig(fig, save_path)
    else:
        plt.show()


# ── 7. Pathway importance ─────────────────────────────────────────────────────

def plot_pathway_importance(
    pathway_df: pd.DataFrame,
    top_n: int = 25,
    save_path: str | None = None,
) -> None:
    if pathway_df is None or pathway_df.empty:
        log.warning("Pathway DataFrame is empty — skipping plot.")
        return
    if "mean_abs_shap" not in pathway_df.columns:
        log.warning("Pathway DataFrame missing 'mean_abs_shap' — skipping plot.")
        return
    df = pathway_df.sort_values("mean_abs_shap", ascending=False).head(top_n).iloc[::-1]
    n = len(df)
    if n == 0:
        log.warning("Pathway DataFrame is empty — skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(11, max(5, n * 0.38 + 2)))

    # Color: darker = higher importance (more specific/leaf)
    vals = df["mean_abs_shap"].values
    norm = (vals - vals.min()) / (vals.max() - vals.min() + 1e-9)
    colors = [plt.cm.YlOrRd(0.3 + 0.65 * v) for v in norm]

    bars = ax.barh(range(n), vals, color=colors, edgecolor="white", lw=0.4)

    if "n_genes" in df.columns:
        for i, (val, ng) in enumerate(zip(vals, df["n_genes"].values)):
            ax.text(val + vals.max() * 0.01, i, f"n={ng}",
                    va="center", fontsize=8, color="#444444")

    labels = [_trunc(str(nm)) for nm in (
        df["pathway_name"].values if "pathway_name" in df.columns else df.index)]
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Mean |SHAP| Pathway Score")
    ax.set_title("Reactome Pathway Importance")
    ax.grid(axis="x", alpha=0.2)
    ax.set_xlim(left=0)

    sm = plt.cm.ScalarMappable(cmap=plt.cm.YlOrRd, norm=Normalize(vals.min(), vals.max()))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.015, pad=0.02)
    cbar.set_label("Importance", fontsize=10)

    fig.tight_layout()
    if save_path:
        _save_fig(fig, save_path)
    else:
        plt.show()


# ── 8. PPI network ────────────────────────────────────────────────────────────

def plot_ppi_importance(
    ppi_df: pd.DataFrame,
    top_n: int = 20,
    save_path: str | None = None,
) -> None:
    if ppi_df.empty:
        log.warning("PPI DataFrame is empty — skipping plot.")
        return

    try:
        g1_col, g2_col, imp_col, s1_col, s2_col = _resolve_ppi_schema(ppi_df)
    except ValueError as exc:
        log.warning("%s — skipping plot.", exc)
        return

    df = ppi_df.sort_values(imp_col, ascending=False).head(top_n)
    has_gene_shap = s1_col is not None and s2_col is not None

    G = nx.Graph()
    for _, row in df.iterrows():
        g1, g2 = row[g1_col], row[g2_col]
        edge_w = _safe_num(row.get(imp_col), 0.0)
        s1 = _safe_num(row.get(s1_col), edge_w / 2.0) if has_gene_shap else edge_w / 2.0
        s2 = _safe_num(row.get(s2_col), edge_w / 2.0) if has_gene_shap else edge_w / 2.0

        if g1 not in G:
            G.add_node(g1, shap=0.0)
        if g2 not in G:
            G.add_node(g2, shap=0.0)
        G.nodes[g1]["shap"] = max(_safe_num(G.nodes[g1].get("shap"), 0.0), s1)
        G.nodes[g2]["shap"] = max(_safe_num(G.nodes[g2].get("shap"), 0.0), s2)
        G.add_edge(g1, g2, weight=edge_w)

    if G.number_of_nodes() == 0:
        return

    # ── Collision-free node layout ───────────────────────────────────────────
    try:
        base_pos = nx.kamada_kawai_layout(G, weight="weight")
    except Exception:
        base_pos = nx.spring_layout(G, seed=42, k=1.2, iterations=300, weight="weight")
    pos = _normalise_pos(base_pos, pad=0.10)

    node_list = list(G.nodes())
    shap_vals = np.array([_safe_num(G.nodes[n].get("shap"), 0.0) for n in node_list], dtype=float)
    sv_abs = np.abs(shap_vals)
    sv_norm = sv_abs / (sv_abs.max() + 1e-9)
    radii = {
        n: float(0.020 + 0.022 * sv_norm[i]) for i, n in enumerate(node_list)
    }
    pos = _resolve_node_overlaps(
        pos,
        radii,
        bounds=(0.03, 0.97, 0.05, 0.95),
        gap=0.004,
        iterations=700,
        anchor_strength=0.03,
    )

    fig, ax = plt.subplots(figsize=(11, 9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    edges = list(G.edges(data=True))
    if edges:
        edge_weights = np.array([_safe_num(d.get("weight"), 0.0) for _, _, d in edges], dtype=float)
        ew_norm = (edge_weights - edge_weights.min()) / (edge_weights.max() - edge_weights.min() + 1e-9)
        for (u, v, d), en in zip(edges, ew_norm):
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            ax.plot(
                [x0, x1], [y0, y1],
                color="#555555",
                linewidth=0.8 + 3.8 * float(en),
                alpha=0.30 + 0.45 * float(en),
                zorder=1,
            )

    node_colors = SHAP_CMAP(
        (shap_vals - shap_vals.min()) / (shap_vals.max() - shap_vals.min() + 1e-9)
    )
    for i, n in enumerate(node_list):
        x, y = pos[n]
        r = radii[n]
        circ = mpatches.Circle(
            (x, y), r,
            facecolor=node_colors[i],
            edgecolor="#333333",
            linewidth=0.9,
            alpha=0.95,
            zorder=3,
        )
        ax.add_patch(circ)
        ax.text(
            x, y,
            str(n),
            fontsize=8,
            ha="center",
            va="center",
            color="#111111",
            zorder=4,
            clip_on=True,
        )

    sm = plt.cm.ScalarMappable(cmap=SHAP_CMAP,
                               norm=Normalize(shap_vals.min(), shap_vals.max()))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Gene contribution score", fontsize=10)

    ax.set_title("Protein-Protein Interaction Network\n"
                 "(node size & color = SHAP importance, edge width = PPI score)",
                 fontsize=12)
    ax.axis("off")
    fig.tight_layout()
    if save_path:
        _save_fig(fig, save_path)
    else:
        plt.show()


def plot_full_ppi_map(
    ppi_df: pd.DataFrame,
    save_path: str | None = None,
) -> None:
    """
    Exhaustive companion view: draw the full PPI table (all edges) with a
    collision-resolved node layout.
    """
    if ppi_df is None or ppi_df.empty:
        log.warning("PPI DataFrame is empty — skipping full PPI map.")
        return

    try:
        g1_col, g2_col, imp_col, s1_col, s2_col = _resolve_ppi_schema(ppi_df)
    except ValueError as exc:
        log.warning("%s — skipping full PPI map.", exc)
        return

    G = nx.Graph()
    for row in ppi_df.itertuples(index=False):
        ra = getattr(row, g1_col)
        rb = getattr(row, g2_col)
        ga, gb = str(ra), str(rb)
        if not ga or not gb or ga == gb:
            continue
        w = _safe_num(getattr(row, imp_col), 0.0)
        if w <= 0:
            continue
        sa = _safe_num(getattr(row, s1_col), w / 2.0) if s1_col else w / 2.0
        sb = _safe_num(getattr(row, s2_col), w / 2.0) if s2_col else w / 2.0
        if ga not in G:
            G.add_node(ga, shap=0.0)
        if gb not in G:
            G.add_node(gb, shap=0.0)
        G.nodes[ga]["shap"] = max(_safe_num(G.nodes[ga].get("shap"), 0.0), sa)
        G.nodes[gb]["shap"] = max(_safe_num(G.nodes[gb].get("shap"), 0.0), sb)
        if G.has_edge(ga, gb):
            G[ga][gb]["weight"] = max(_safe_num(G[ga][gb].get("weight"), 0.0), w)
        else:
            G.add_edge(ga, gb, weight=w)

    if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
        log.warning("No valid PPI edges available — skipping full PPI map.")
        return

    n_nodes = G.number_of_nodes()
    try:
        k = max(0.25, min(2.4, 2.0 / np.sqrt(max(n_nodes, 1))))
        base_pos = nx.spring_layout(
            G,
            seed=42,
            iterations=450 if n_nodes <= 500 else 250,
            k=k,
            weight="weight",
        )
    except Exception:
        base_pos = {n: (np.cos(i), np.sin(i)) for i, n in enumerate(G.nodes())}
    pos = _normalise_pos(base_pos, pad=0.08)

    node_list = list(G.nodes())
    shap_vals = np.array([_safe_num(G.nodes[n].get("shap"), 0.0) for n in node_list], dtype=float)
    sv_abs = np.abs(shap_vals)
    sv_norm = sv_abs / (sv_abs.max() + 1e-9)

    # Adaptive radii keep dense graphs readable while enforcing no overlaps.
    base_r = float(np.clip(0.24 / np.sqrt(max(n_nodes, 1)), 0.003, 0.013))
    radii = {n: float(base_r * (0.85 + 1.15 * sv_norm[i])) for i, n in enumerate(node_list)}
    pos = _resolve_node_overlaps(
        pos,
        radii,
        bounds=(0.02, 0.98, 0.04, 0.96),
        gap=max(0.001, base_r * 0.20),
        iterations=900 if n_nodes <= 350 else 500,
        anchor_strength=0.02,
    )

    # If any residual overlaps remain, shrink radii slightly and re-resolve.
    overlap_gap = max(0.0008, base_r * 0.15)
    retry = 0
    while _count_node_overlaps(pos, radii, gap=overlap_gap) > 0 and retry < 5:
        radii = {k: v * 0.92 for k, v in radii.items()}
        pos = _resolve_node_overlaps(
            pos,
            radii,
            bounds=(0.02, 0.98, 0.04, 0.96),
            gap=overlap_gap,
            iterations=500,
            anchor_strength=0.02,
        )
        retry += 1

    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    edges = list(G.edges(data=True))
    ew = np.array([_safe_num(d.get("weight"), 0.0) for _, _, d in edges], dtype=float)
    ew_norm = (ew - ew.min()) / (ew.max() - ew.min() + 1e-9)
    for (u, v, d), en in zip(edges, ew_norm):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        ax.plot(
            [x0, x1], [y0, y1],
            color="#546E7A",
            linewidth=0.18 + 1.75 * float(en),
            alpha=0.10 + 0.30 * float(en),
            zorder=1,
        )

    cvals = SHAP_CMAP((shap_vals - shap_vals.min()) / (shap_vals.max() - shap_vals.min() + 1e-9))
    for i, n in enumerate(node_list):
        x, y = pos[n]
        r = radii[n]
        ax.add_patch(
            mpatches.Circle(
                (x, y),
                r,
                facecolor=cvals[i],
                edgecolor="#1f2937",
                linewidth=0.35,
                alpha=0.95,
                zorder=3,
            )
        )

    # Label only top contributors to preserve readability.
    top_label_n = min(60, n_nodes)
    rank_nodes = sorted(node_list, key=lambda n: _safe_num(G.nodes[n].get("shap"), 0.0), reverse=True)
    for n in rank_nodes[:top_label_n]:
        x, y = pos[n]
        ax.text(
            x,
            y + radii[n] + 0.003,
            str(n),
            fontsize=7,
            ha="center",
            va="bottom",
            color="#0f172a",
            zorder=4,
            clip_on=True,
        )

    sm = plt.cm.ScalarMappable(cmap=SHAP_CMAP, norm=Normalize(shap_vals.min(), shap_vals.max()))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.02, pad=0.01)
    cbar.set_label("Gene contribution score", fontsize=10)

    ax.set_title(
        f"Full PPI Interaction Map ({G.number_of_nodes()} proteins, "
        f"{G.number_of_edges()} interactions)",
        fontsize=13,
        pad=12,
    )
    ax.axis("off")
    fig.tight_layout()
    if save_path:
        _save_fig(fig, save_path)
    else:
        plt.show()


# ── 9. Biological Cascade ─────────────────────────────────────────────────────

def _safe_num(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


# Diverging cascade colormap: blue (HPV−) → near-white → red (HPV+)
_CASCADE_CMAP = LinearSegmentedColormap.from_list(
    "cascade_div", ["#3B82F6", "#F9FAFB", "#DC2626"], N=256
)


def plot_biological_cascade(
    cascade: dict,
    gene_shap_df: pd.DataFrame | None = None,
    pathway_df: pd.DataFrame | None = None,
    ppi_df: pd.DataFrame | None = None,
    p2p_df: pd.DataFrame | None = None,
    save_path: str | None = None,
    compact: bool = False,
) -> None:
    """
    Four-column biological cascade visualization (white background, single figure).

    Columns (x in axes data coords [0, 1]):
        0.03–0.10   Gene nodes (SHAP-colored rounded rectangles)
        0.18–0.56   Pathway boxes with horizontal protein rows + PPI arcs
        0.66–0.78   Pathway hierarchy nodes
        0.86–0.94   HPV-status output node
    """
    # ── Limits by mode ────────────────────────────────────────────────────────
    n_genes = 6  if compact else 10
    n_boxes = 3  if compact else 5
    n_cross = 3  if compact else 5
    n_hier  = 2  if compact else 5
    figw, figh = (14, 7) if compact else (20, 10)

    # ── Data extraction ───────────────────────────────────────────────────────
    top_genes = sorted(
        cascade.get("top_genes", []),
        key=lambda g: _safe_num(g.get("shap"), 0.0),
        reverse=True,
    )[:n_genes]
    top_leaf      = cascade.get("top_leaf_pathways", [])[:n_boxes]
    top_inter     = cascade.get("top_intermediate_pathways", [])
    top_root      = cascade.get("top_root_pathways", [])
    cascade_edges = cascade.get("cascade_edges", [])

    if not top_genes or not top_leaf:
        log.warning("Insufficient cascade data — skipping cascade plot.")
        return

    # ── SHAP colormap ─────────────────────────────────────────────────────────
    shap_signed: dict[str, float] = {
        g["gene"]: _safe_num(g.get("shap"), 0.0) for g in top_genes
    }
    if gene_shap_df is not None and not gene_shap_df.empty and "gene" in gene_shap_df.columns:
        for _, row in gene_shap_df.iterrows():
            shap_signed.setdefault(str(row["gene"]),
                                   _safe_num(row.get("mean_abs_shap"), 0.0))

    shap_max = max((abs(v) for v in shap_signed.values()), default=1.0) or 1.0
    shap_norm = Normalize(vmin=-shap_max, vmax=shap_max)

    def _gcol(gene: str) -> tuple:
        return _CASCADE_CMAP(shap_norm(shap_signed.get(gene, 0.0)))

    # ── Gene → leaf-pathway membership ───────────────────────────────────────
    leaf_ids   = {p["id"] for p in top_leaf}
    leaf_info  = {p["id"]: p for p in top_leaf}
    gene_names = {g["gene"] for g in top_genes}

    gene_to_leaves: dict[str, list[str]] = {}
    for u, v, _w in cascade_edges:
        u, v = str(u), str(v)
        if u in gene_names and v in leaf_ids:
            if v not in gene_to_leaves.setdefault(u, []):
                gene_to_leaves[u].append(v)

    pathway_to_genes: dict[str, list[str]] = {}
    for gene, pids in gene_to_leaves.items():
        for pid in pids:
            pathway_to_genes.setdefault(pid, []).append(gene)

    # ── PPI within-pathway and cross-pathway edges ────────────────────────────
    ppi_within: dict[str, list[tuple[str, str, float]]] = {}
    cross_ppi_raw: list[tuple[str, str, str, str, float]] = []

    if ppi_df is not None and not ppi_df.empty:
        ga_col = "gene_a" if "gene_a" in ppi_df.columns else "gene_1"
        gb_col = "gene_b" if "gene_b" in ppi_df.columns else "gene_2"
        imp_col = next(
            (c for c in ("multiplicative", "ppi_importance_mult",
                         "ppi_importance_add", "additive")
             if c in ppi_df.columns),
            None,
        )
        for _, row in ppi_df.iterrows():
            ga = str(row.get(ga_col, ""))
            gb = str(row.get(gb_col, ""))
            if ga not in gene_names or gb not in gene_names:
                continue
            imp = _safe_num(row.get(imp_col), 0.0) if imp_col else 0.0
            if imp <= 0:
                continue
            la = set(gene_to_leaves.get(ga, [])) & leaf_ids
            lb = set(gene_to_leaves.get(gb, [])) & leaf_ids
            for pid in la & lb:
                ppi_within.setdefault(pid, []).append((ga, gb, imp))
            for pa in sorted(la - lb):
                for pb in sorted(lb - la):
                    cross_ppi_raw.append((ga, gb, pa, pb, imp))

    cross_ppi = sorted(cross_ppi_raw, key=lambda e: e[4], reverse=True)[:n_cross]

    # ── Hierarchy nodes (mix inter + root, top-N by attribution) ─────────────
    all_hier = [
        {
            "id": str(p.get("id", "")),
            "name": str(p.get("name", p.get("id", ""))),
            "attribution": _safe_num(p.get("attribution"), 0.0),
        }
        for p in list(top_inter) + list(top_root)
        if str(p.get("id", ""))
    ]
    all_hier.sort(key=lambda x: x["attribution"], reverse=True)
    seen_hier: set[str] = set()
    hier_nodes = []
    for h in all_hier:
        if h["id"] not in seen_hier:
            seen_hier.add(h["id"])
            hier_nodes.append(h)
            if len(hier_nodes) == n_hier:
                break

    # ── Figure and axes ───────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(figw, figh))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.94, bottom=0.04)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    # ── Layout constants ──────────────────────────────────────────────────────
    X_GENE  = 0.065
    GENE_W  = 0.065
    GENE_H  = 0.052
    X_BL    = 0.18
    X_BR    = 0.56
    BOX_W   = X_BR - X_BL
    X_HL    = 0.66
    X_HR    = 0.78
    HIER_W  = X_HR - X_HL
    HIER_H  = 0.042
    X_OUT   = 0.90
    OUT_W   = 0.075
    OUT_H   = 0.055
    PROT_R  = 0.010
    HDR_FRAC = 0.22

    # ── Gene y-positions (most positive SHAP at top) ──────────────────────────
    n_g   = len(top_genes)
    gene_ys = np.linspace(0.88, 0.10, n_g) if n_g > 1 else [0.50]
    gene_pos: dict[str, tuple[float, float]] = {}
    for i, ginfo in enumerate(top_genes):
        gene_pos[ginfo["gene"]] = (X_GENE, float(gene_ys[i]))

    # ── Pathway box geometry ──────────────────────────────────────────────────
    n_b     = len(top_leaf)
    MIN_BH  = 0.065
    MAX_BH  = 0.14
    gap     = 0.012
    total_avail = 0.84 - gap * max(n_b - 1, 0)
    avg_bh  = float(np.clip(total_avail / max(n_b, 1), MIN_BH, MAX_BH))

    box_pos: dict[str, tuple[float, float, float, float]] = {}
    protein_pos: dict[tuple[str, str], tuple[float, float]] = {}

    y_cur = 0.92
    for pinfo in top_leaf:
        pid = pinfo["id"]
        n_m = max(1, len(pathway_to_genes.get(pid, [])))
        bh  = float(np.clip(avg_bh * (0.7 + 0.3 * min(n_m / 4.0, 1.0)), MIN_BH, MAX_BH))
        bh  = min(bh, y_cur - 0.02)
        if bh <= 0.01:
            break
        yb = y_cur - bh
        box_pos[pid] = (X_BL, yb, BOX_W, bh)
        y_cur = yb - gap

    # ── Protein positions (horizontal row, centered below box header) ─────────
    for pid, (bx, by, bw, bh) in box_pos.items():
        genes_in = pathway_to_genes.get(pid, [])
        if not genes_in:
            continue
        n_p   = len(genes_in)
        hdr_h = bh * HDR_FRAC
        row_y = by + (bh - hdr_h) * 0.50
        pad_x = 0.10
        if n_p == 1:
            xs_rel = [0.50]
        else:
            xs_rel = [pad_x + i * (1.0 - 2 * pad_x) / (n_p - 1) for i in range(n_p)]
        for j, gene in enumerate(genes_in):
            protein_pos[(pid, gene)] = (bx + xs_rel[j] * bw, row_y)

    # ── Hierarchy y-positions ─────────────────────────────────────────────────
    n_h     = len(hier_nodes)
    hier_ys = np.linspace(0.86, 0.14, n_h) if n_h > 1 else [0.50]
    hier_cx = (X_HL + X_HR) / 2
    hier_pos_map: dict[str, tuple[float, float]] = {}
    for i, hinfo in enumerate(hier_nodes):
        hier_pos_map[hinfo["id"]] = (hier_cx, float(hier_ys[i]))

    hier_attr_max = max((h["attribution"] for h in hier_nodes), default=1.0) or 1.0
    leaf_attr_max = max(
        (_safe_num(p.get("attribution"), 0.0) for p in top_leaf), default=1.0
    ) or 1.0
    out_cy = 0.50

    # ═══════════════════════════════════════════════════════════════════════════
    # Draw in z-order
    # ═══════════════════════════════════════════════════════════════════════════

    # ── 1. Gene → pathway connection lines ───────────────────────────────────
    for gene, (gcx, gcy) in gene_pos.items():
        sv = shap_signed.get(gene, 0.0)
        lw = 0.8 + 2.2 * (abs(sv) / shap_max)
        ec = _gcol(gene)
        for pid in gene_to_leaves.get(gene, []):
            if pid not in box_pos:
                continue
            tx, ty = (
                protein_pos[(pid, gene)]
                if (pid, gene) in protein_pos
                else (box_pos[pid][0], box_pos[pid][1] + box_pos[pid][3] / 2)
            )
            ax.add_patch(mpatches.FancyArrowPatch(
                (gcx + GENE_W / 2, gcy), (tx, ty),
                connectionstyle="arc3,rad=0.0",
                arrowstyle="-",
                color=ec, linewidth=float(lw), alpha=0.30, zorder=1,
            ))

    # ── 2. Pathway → hierarchy connection lines ───────────────────────────────
    for pinfo in top_leaf:
        pid = pinfo["id"]
        if pid not in box_pos:
            continue
        bx, by, bw, bh = box_pos[pid]
        src_x = bx + bw
        src_y = by + bh / 2
        p_attr = _safe_num(pinfo.get("attribution"), 0.01)
        lw    = float(np.clip(1.0 + 3.0 * (p_attr / leaf_attr_max), 1.0, 4.0))
        alpha = float(np.clip(0.40 + 0.35 * (p_attr / leaf_attr_max), 0.40, 0.75))
        for hinfo in hier_nodes:
            hcx, hcy = hier_pos_map[hinfo["id"]]
            ax.plot(
                [src_x, hcx - HIER_W / 2], [src_y, hcy],
                color="#64748B", linewidth=lw, alpha=alpha, zorder=2,
            )

    # ── 3. Hierarchy → output (arrowheads only here) ──────────────────────────
    for hinfo in hier_nodes:
        hcx, hcy = hier_pos_map[hinfo["id"]]
        lw = float(np.clip(1.5 + 2.5 * (hinfo["attribution"] / hier_attr_max), 1.5, 4.0))
        ax.add_patch(mpatches.FancyArrowPatch(
            (hcx + HIER_W / 2, hcy),
            (X_OUT - OUT_W / 2, out_cy),
            connectionstyle="arc3,rad=0.0",
            arrowstyle="-|>,head_length=6,head_width=4",
            color="#334155",
            linewidth=lw, alpha=0.75, zorder=3,
            mutation_scale=6.0,
        ))

    # ── 4. Pathway box background fills ──────────────────────────────────────
    for pid, (bx, by, bw, bh) in box_pos.items():
        ax.add_patch(mpatches.FancyBboxPatch(
            (bx, by), bw, bh,
            boxstyle="round,pad=0.006",
            facecolor="#F8FAFC", edgecolor="none",
            linewidth=0, zorder=4,
        ))
        hdr_h = bh * HDR_FRAC
        ax.plot(
            [bx + 0.004, bx + bw - 0.004],
            [by + bh - hdr_h, by + bh - hdr_h],
            color="#E2E8F0", linewidth=0.8, zorder=4,
        )

    # ── 5. PPI arcs inside boxes ──────────────────────────────────────────────
    for pid in box_pos:
        edges_in = sorted(ppi_within.get(pid, []), key=lambda e: e[2])
        if not edges_in:
            continue
        imp_max_in = max(e[2] for e in edges_in) or 1.0
        for ga, gb, imp in edges_in:
            if (pid, ga) not in protein_pos or (pid, gb) not in protein_pos:
                continue
            ni    = float(imp / imp_max_in)
            lw    = 1.5 + 3.5 * (ni ** 0.6)
            alpha = 0.25 + 0.60 * ni
            ax.add_patch(mpatches.FancyArrowPatch(
                protein_pos[(pid, ga)],
                protein_pos[(pid, gb)],
                connectionstyle="arc3,rad=-0.3",
                arrowstyle="-",
                color="#1E293B",
                linewidth=float(lw), alpha=float(alpha), zorder=5,
            ))

    # ── 6. Cross-pathway PPI curves + midpoint labels ─────────────────────────
    for ga, gb, pa, pb, imp in cross_ppi:
        pos_a = protein_pos.get((pa, ga)) or protein_pos.get((pa, gb))
        pos_b = protein_pos.get((pb, gb)) or protein_pos.get((pb, ga))
        if pos_a is None or pos_b is None:
            continue
        ni    = float(min(imp / (shap_max + 1e-9), 1.0))
        lw    = 1.5 + 2.0 * ni
        alpha = 0.50 + 0.30 * ni
        ax.add_patch(mpatches.FancyArrowPatch(
            pos_a, pos_b,
            connectionstyle="arc3,rad=0.25",
            arrowstyle="-",
            color="#0D9488",
            linewidth=float(lw), alpha=float(alpha), zorder=6,
        ))
        mx = (pos_a[0] + pos_b[0]) / 2
        my = (pos_a[1] + pos_b[1]) / 2
        ax.text(
            mx, my, f"{ga}–{gb}",
            fontsize=5.5, ha="center", va="center", color="#0D9488", zorder=6,
            bbox=dict(facecolor="white", alpha=0.70, edgecolor="none", pad=1),
        )

    # ── 7. Protein circles ────────────────────────────────────────────────────
    for (pid, gene), (px, py) in protein_pos.items():
        ax.add_patch(mpatches.Circle(
            (px, py), PROT_R,
            facecolor=_gcol(gene), edgecolor="#475569", linewidth=0.8, zorder=7,
        ))

    # ── 8. Protein labels ─────────────────────────────────────────────────────
    for (pid, gene), (px, py) in protein_pos.items():
        ax.text(
            px + PROT_R + 0.013, py, gene,
            fontsize=6.5, ha="left", va="center", color="#334155",
            zorder=8, clip_on=True,
        )

    # ── 9. Pathway box borders (crisp, drawn on top of fill) ─────────────────
    for pid, (bx, by, bw, bh) in box_pos.items():
        ax.add_patch(mpatches.FancyBboxPatch(
            (bx, by), bw, bh,
            boxstyle="round,pad=0.006",
            facecolor="none", edgecolor="#CBD5E1",
            linewidth=1.0, zorder=9,
        ))

    # ── 10. Pathway header text ───────────────────────────────────────────────
    for pid, (bx, by, bw, bh) in box_pos.items():
        hdr_h = bh * HDR_FRAC
        ax.text(
            bx + bw / 2, by + bh - hdr_h / 2,
            _trunc(leaf_info[pid]["name"], 42),
            ha="center", va="center", fontsize=8, fontweight="bold",
            color="#334155", zorder=10, clip_on=True,
        )

    # ── 11. Gene nodes ────────────────────────────────────────────────────────
    for gene, (gcx, gcy) in gene_pos.items():
        ax.add_patch(mpatches.FancyBboxPatch(
            (gcx - GENE_W / 2, gcy - GENE_H / 2), GENE_W, GENE_H,
            boxstyle="round,pad=0.012",
            facecolor=_gcol(gene), edgecolor="#1E293B",
            linewidth=1.2, zorder=11,
        ))

    # ── 12. Gene labels ───────────────────────────────────────────────────────
    for ginfo in top_genes:
        gene = ginfo["gene"]
        gcx, gcy = gene_pos[gene]
        ax.text(gcx, gcy + 0.010, gene,
                ha="center", va="center", fontsize=9, fontweight="bold",
                color="#0F172A", zorder=12)
        ax.text(gcx, gcy - 0.012, _trunc(str(ginfo.get("protein", "")), 20),
                ha="center", va="center", fontsize=7, style="italic",
                color="#64748B", zorder=12)

    # ── 13. Hierarchy nodes ───────────────────────────────────────────────────
    for hinfo in hier_nodes:
        hcx, hcy = hier_pos_map[hinfo["id"]]
        ax.add_patch(mpatches.FancyBboxPatch(
            (hcx - HIER_W / 2, hcy - HIER_H / 2), HIER_W, HIER_H,
            boxstyle="round,pad=0.006",
            facecolor="#F1F5F9", edgecolor="#94A3B8",
            linewidth=1.0, zorder=13,
        ))

    # ── 14. Hierarchy labels ──────────────────────────────────────────────────
    for hinfo in hier_nodes:
        hcx, hcy = hier_pos_map[hinfo["id"]]
        ax.text(hcx, hcy, _trunc(hinfo["name"], 28),
                ha="center", va="center", fontsize=7.5, color="#334155",
                zorder=14, clip_on=True)

    # ── 15. Output node + labels ──────────────────────────────────────────────
    ax.add_patch(mpatches.FancyBboxPatch(
        (X_OUT - OUT_W / 2, out_cy - OUT_H / 2), OUT_W, OUT_H,
        boxstyle="round,pad=0.006",
        facecolor="#1E293B", edgecolor="#1E293B",
        linewidth=1.2, zorder=15,
    ))
    ax.text(X_OUT, out_cy + 0.010, "HPV Status",
            ha="center", va="center", fontsize=9.5, fontweight="bold",
            color="white", zorder=15)
    ax.text(X_OUT, out_cy - 0.014, "Prediction",
            ha="center", va="center", fontsize=7.0, color="#94A3B8", zorder=15)

    # ── 16. Title, column headers, legend ─────────────────────────────────────
    ax.text(0.50, 0.975,
            "Biological Cascade: How HPV Drives HNSCC Proliferation",
            ha="center", va="top", fontsize=14, fontweight="medium",
            color="#0F172A", zorder=16)
    for label, cx in [
        ("Genes",                          X_GENE),
        ("Protein Interactions by Pathway", (X_BL + X_BR) / 2),
        ("Pathway Hierarchy",              (X_HL + X_HR) / 2),
        ("Prediction",                     X_OUT),
    ]:
        ax.text(cx, 0.945, label,
                ha="center", va="center", fontsize=9, fontweight="medium",
                color="#64748B", zorder=16)

    # Legend box (data coords, bottom-right corner)
    LEG_X0, LEG_Y0, LEG_W, LEG_H = 0.60, 0.012, 0.39, 0.082
    ax.add_patch(mpatches.FancyBboxPatch(
        (LEG_X0, LEG_Y0), LEG_W, LEG_H,
        boxstyle="round,pad=0.008",
        facecolor="#F1F5F9", edgecolor="#E2E8F0",
        linewidth=0.8, zorder=16,
    ))

    # Colorbar gradient (inline in axes coords)
    cbar_grad = np.linspace(0, 1, 256).reshape(1, 256)
    ax.imshow(
        cbar_grad, aspect="auto", cmap=_CASCADE_CMAP,
        extent=[LEG_X0 + 0.005, LEG_X0 + 0.140,
                LEG_Y0 + 0.010, LEG_Y0 + 0.030],
        zorder=17,
    )
    ax.text(LEG_X0 + 0.072, LEG_Y0 + 0.038,
            "← HPV−  SHAP  HPV+ →",
            fontsize=6.5, ha="center", va="center", color="#334155", zorder=17)

    # Importance line samples
    lx0 = LEG_X0 + 0.155
    for i, (label, lw) in enumerate([
        ("Low importance",  1.0),
        ("Med importance",  2.5),
        ("High importance", 4.5),
    ]):
        ly = LEG_Y0 + LEG_H - 0.018 - i * 0.022
        ax.plot([lx0, lx0 + 0.030], [ly, ly],
                color="#64748B", linewidth=lw, alpha=0.75, zorder=17)
        ax.text(lx0 + 0.034, ly, label,
                fontsize=6, va="center", color="#64748B", zorder=17)

    # Cross-pathway PPI legend line
    ly_cross = LEG_Y0 + LEG_H - 0.018 - 3 * 0.022
    ax.plot([lx0, lx0 + 0.030], [ly_cross, ly_cross],
            color="#0D9488", linewidth=2.0, alpha=0.75, zorder=17)
    ax.text(lx0 + 0.034, ly_cross, "Cross-pathway PPI",
            fontsize=6, va="center", color="#64748B", zorder=17)

    # ── Save ─────────────────────────────────────────────────────────────────
    if save_path:
        _save_fig(fig, save_path)
    else:
        plt.show()

# ── 10. Model comparison radar ────────────────────────────────────────────────

_RADAR_METRICS = ["auroc", "auprc", "balanced_accuracy", "f1", "recall",
                  "specificity", "mcc"]
_RADAR_LABELS  = ["AUROC", "AUPRC", "Bal. Acc.", "F1", "Sensitivity",
                  "Specificity", "MCC"]


def plot_model_comparison_radar(
    all_fold_metrics: list[dict],
    save_path: str | None = None,
) -> None:
    # Compute mean of each metric per model
    model_means: dict[str, np.ndarray] = {}
    for mname in MODEL_ORDER:
        mets = _fold_metrics_for(all_fold_metrics, mname)
        if not mets:
            continue
        vals = []
        for key in _RADAR_METRICS:
            raw = np.mean([m.get(key, 0) for m in mets])
            # Normalise MCC from [-1,1] to [0,1]
            if key == "mcc":
                raw = (raw + 1) / 2
            vals.append(float(np.clip(raw, 0, 1)))
        model_means[mname] = np.array(vals)

    if not model_means:
        log.warning("No model data for radar chart.")
        return

    n = len(_RADAR_METRICS)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for mname, vals in model_means.items():
        v = vals.tolist() + [vals[0]]
        col = _color(mname)
        ax.plot(angles, v, color=col, lw=2.2, label=_disp(mname))
        ax.fill(angles, v, color=col, alpha=0.12)

    ax.set_thetagrids(np.degrees(angles[:-1]), _RADAR_LABELS, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=8, color="gray")
    ax.yaxis.grid(True, color="gray", alpha=0.3)
    ax.xaxis.grid(True, color="gray", alpha=0.3)
    ax.set_title("Model Comparison — Multi-Metric Radar\n"
                 "(MCC normalised to [0,1])", pad=20, fontsize=12)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=10)

    fig.tight_layout()
    if save_path:
        _save_fig(fig, save_path)
    else:
        plt.show()


# ── 11. Model comparison boxplot ──────────────────────────────────────────────

def plot_model_comparison_boxplot(
    all_fold_metrics: list[dict],
    stat_tests: pd.DataFrame | None = None,
    save_path: str | None = None,
) -> None:
    rows = []
    for m in all_fold_metrics:
        rows.append({"model": m["model"], "auroc": m.get("auroc", float("nan")),
                     "fold": m.get("fold", 0)})
    df = pd.DataFrame(rows)
    if df.empty:
        log.warning("Empty metrics for boxplot.")
        return

    # Order and display names
    present = [m for m in MODEL_ORDER if m in df["model"].values]
    df["display"] = df["model"].map(_disp)
    order = [_disp(m) for m in present]
    palette = {_disp(m): _color(m) for m in present}

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x="display", y="auroc", hue="display", order=order, palette=palette,
                width=0.5, linewidth=1.4, fliersize=0, ax=ax, legend=False)
    sns.stripplot(data=df, x="display", y="auroc", hue="display", order=order, palette=palette,
                  size=7, alpha=0.7, jitter=True, dodge=False, ax=ax, legend=False,
                  edgecolor="white", linewidth=0.5)

    # Significance brackets vs BINN
    if stat_tests is not None and not stat_tests.empty:
        binn_auroc = df[df["model"] == "binn"]["auroc"].values
        y_max = df["auroc"].max()
        step = 0.05
        for i, mname in enumerate([m for m in present if m != "binn"]):
            row = stat_tests[stat_tests["model"] == mname] if "model" in stat_tests else pd.DataFrame()
            if row.empty:
                continue
            p = float(row["t_pvalue"].iloc[0])
            star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            binn_x = order.index(_disp("binn"))
            cmp_x  = order.index(_disp(mname))
            y_br = y_max + step * (i + 1)
            ax.plot([binn_x, binn_x, cmp_x, cmp_x], [y_br - step*0.1, y_br, y_br, y_br - step*0.1],
                    lw=1.2, color="#333333")
            ax.text((binn_x + cmp_x) / 2, y_br + 0.005, star,
                    ha="center", va="bottom", fontsize=11, color="#333333")

    ax.set_xlabel("")
    ax.set_ylabel("AUROC")
    ax.set_title("Per-Fold AUROC Distribution by Model")
    ax.grid(axis="y", alpha=0.25)
    ax.set_ylim(bottom=max(0, df["auroc"].min() - 0.05))
    fig.tight_layout()
    if save_path:
        _save_fig(fig, save_path)
    else:
        plt.show()


# ── 12. Training curves ───────────────────────────────────────────────────────

def plot_training_curves(
    training_histories: list[dict],
    save_path: str | None = None,
) -> None:
    if not training_histories:
        log.warning("No training histories — skipping training curves.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    c = _color("binn")
    cmap_f = plt.cm.Blues

    n_folds = len(training_histories)
    for i, h in enumerate(training_histories):
        alpha = 0.35
        fold_col = cmap_f(0.4 + 0.55 * i / max(n_folds - 1, 1))
        tr_loss = h.get("train_loss", [])
        vl_loss = h.get("val_loss",   [])
        vl_auc  = h.get("val_auroc",  [])
        best_ep = h.get("best_epoch", len(tr_loss) - 1)

        epochs = range(len(tr_loss))
        ax1.plot(epochs, tr_loss, lw=0.8, alpha=alpha, color=fold_col, ls="--")
        ax1.plot(epochs, vl_loss, lw=0.8, alpha=alpha, color=fold_col)
        ax1.axvline(best_ep, lw=0.8, alpha=alpha, color=fold_col, ls=":")
        ax2.plot(range(len(vl_auc)), vl_auc, lw=0.9, alpha=alpha, color=fold_col)
        ax2.axvline(best_ep, lw=0.8, alpha=alpha, color=fold_col, ls=":")

    # Mean curves
    max_ep_loss = max(len(h.get("val_loss", [])) for h in training_histories)
    max_ep_auc  = max(len(h.get("val_auroc", [])) for h in training_histories)

    def _pad_mean(key, max_e):
        arrs = [h.get(key, []) for h in training_histories]
        padded = [np.pad(a, (0, max_e - len(a)), constant_values=np.nan) for a in arrs]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return np.nanmean(padded, axis=0)

    mean_tr = _pad_mean("train_loss", max_ep_loss)
    mean_vl = _pad_mean("val_loss",   max_ep_loss)
    mean_au = _pad_mean("val_auroc",  max_ep_auc)

    ax1.plot(mean_tr, lw=2.2, color="#C73E1D", ls="--", label="Train loss (mean)")
    ax1.plot(mean_vl, lw=2.2, color=c, label="Val loss (mean)")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("BCE Loss")
    ax1.set_title("BINN Training & Validation Loss")
    ax1.legend(fontsize=9); ax1.grid(alpha=0.2)

    ax2.plot(mean_au, lw=2.2, color=c, label="Val AUROC (mean)")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("AUROC")
    ax2.set_title("BINN Validation AUROC")
    ax2.set_ylim(0, 1); ax2.grid(alpha=0.2); ax2.legend(fontsize=9)

    from matplotlib.lines import Line2D
    legend_els = [
        Line2D([0],[0], color="gray", ls="--", lw=1.2, label="Train loss (fold)"),
        Line2D([0],[0], color="gray", lw=1.2, label="Val loss (fold)"),
        Line2D([0],[0], color="gray", ls=":", lw=1.2, label="Best epoch (fold)"),
    ]
    ax1.legend(handles=legend_els + [
        Line2D([0],[0], color="#C73E1D", ls="--", lw=2, label="Train loss (mean)"),
        Line2D([0],[0], color=c, lw=2, label="Val loss (mean)"),
    ], fontsize=9)

    fig.tight_layout()
    if save_path:
        _save_fig(fig, save_path)
    else:
        plt.show()


# ── 13. Network sparsity ──────────────────────────────────────────────────────

def plot_network_sparsity(
    connectivity_matrices: list,
    layer_names: list[str] | None = None,
    save_path: str | None = None,
) -> None:
    if not connectivity_matrices:
        log.warning("No connectivity matrices — skipping sparsity plot.")
        return

    import torch
    n = len(connectivity_matrices)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 3.5 * rows))
    axes = np.array(axes).flatten()

    for i, (C, ax) in enumerate(zip(connectivity_matrices, axes)):
        if isinstance(C, torch.Tensor):
            mat = C.cpu().numpy()
        else:
            mat = np.array(C)

        # Downsample if very large (strided sampling keeps this dependency-free).
        max_dim = 200
        if mat.shape[0] > max_dim or mat.shape[1] > max_dim:
            fy = max(1, int(np.ceil(mat.shape[0] / max_dim)))
            fx = max(1, int(np.ceil(mat.shape[1] / max_dim)))
            mat = mat[::fy, ::fx]

        nnz = int((mat > 0).sum())
        total = mat.size
        sparsity = 100 * (1 - nnz / total)

        im = ax.imshow(mat, cmap="Blues", aspect="auto",
                       interpolation="nearest", vmin=0, vmax=1)
        name = layer_names[i] if layer_names and i < len(layer_names) else f"Layer {i}"
        ax.set_title(f"{name}\n{mat.shape[0]}×{mat.shape[1]}, {sparsity:.1f}% sparse",
                     fontsize=9)
        ax.set_xlabel("Output nodes", fontsize=8)
        ax.set_ylabel("Input nodes", fontsize=8)
        ax.tick_params(labelsize=7)

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle("BINN Connectivity Matrices (Sparse Architecture)", fontsize=13, y=1.02)
    fig.tight_layout()
    if save_path:
        _save_fig(fig, save_path)
    else:
        plt.show()


# ── 14. Master generator ──────────────────────────────────────────────────────

def generate_all_figures(
    results_dir: str = config.METRIC_DIR,
    shap_dir: str    = config.SHAP_DIR,
    figure_dir: str  = config.FIGURE_DIR,
) -> None:
    """
    Load all saved results and generate every figure.
    Skips gracefully if a required file is missing.
    """
    os.makedirs(figure_dir, exist_ok=True)

    def _fp(d, name):
        return os.path.join(d, name)

    def _load_json(path):
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        log.warning(f"Missing: {path}")
        return None

    def _load_csv(path):
        if os.path.exists(path):
            return pd.read_csv(path)
        log.warning(f"Missing: {path}")
        return pd.DataFrame()

    print("Generating figures...")

    # ── Load data ─────────────────────────────────────────────────────────────
    fold_metrics   = _load_json(_fp(results_dir, "nested_cv_results.json")) or []
    stat_tests_df  = _load_csv(_fp(results_dir, "statistical_tests.csv"))
    histories      = _load_json(_fp(results_dir, "training_histories.json")) or []
    cascade        = _load_json(_fp(shap_dir, "hpv_cascade.json")) or {}
    gene_shap_df   = _load_csv(_fp(shap_dir, "gene_shap_values.csv"))
    pathway_df     = _load_csv(_fp(shap_dir, "pathway_importance.csv"))
    ppi_df         = _load_csv(_fp(shap_dir, "ppi_importance.csv"))
    p2p_df         = _load_csv(_fp(shap_dir, "pathway_to_pathway_importance.csv"))

    N = 12
    status = {}

    def _run(step: int, label: str, fn, *args, **kwargs):
        try:
            fn(*args, **kwargs)
            print(f"  [{step}/{N}] {label:<40s} ✓")
            status[label] = "ok"
        except Exception as exc:
            print(f"  [{step}/{N}] {label:<40s} ✗  ({exc})")
            status[label] = str(exc)

    if fold_metrics:
        # 1. PR curves
        _run(1, "Precision-Recall curves",
             plot_precision_recall_curves, fold_metrics,
             _fp(figure_dir, "precision_recall_curves"))

        # 2. Spec-Sens curves
        _run(2, "Specificity-Sensitivity curves",
             plot_specificity_sensitivity_curves, fold_metrics,
             _fp(figure_dir, "specificity_sensitivity_curves"))

        # 3. ROC curves
        _run(3, "ROC curves",
             plot_roc_curves, fold_metrics,
             _fp(figure_dir, "roc_curves"))

        # 4. Confusion matrices
        _run(4, "Confusion matrices",
             plot_confusion_matrices, fold_metrics,
             _fp(figure_dir, "confusion_matrices"))
    else:
        print(f"  [1/{N}] Precision-Recall curves                  – (nested_cv_results.json missing)")
        print(f"  [2/{N}] Specificity-Sensitivity curves           – (nested_cv_results.json missing)")
        print(f"  [3/{N}] ROC curves                               – (nested_cv_results.json missing)")
        print(f"  [4/{N}] Confusion matrices                       – (nested_cv_results.json missing)")

    # 5. SHAP beeswarm needs per-sample SHAP arrays; aggregated CSV is insufficient.
    print(
        f"  [5/{N}] SHAP beeswarm                           – "
        "(requires per-sample SHAP arrays; not saved in current artifacts)"
    )

    # 6. SHAP bar
    _run(6, "SHAP gene importance bar",
         plot_shap_bar, gene_shap_df, 30,
         _fp(figure_dir, "shap_gene_importance"))

    # 7. Pathway importance
    _run(7, "Pathway importance",
         plot_pathway_importance, pathway_df, 25,
         _fp(figure_dir, "pathway_importance"))

    # 8. PPI network
    _run(8, "PPI network",
         plot_ppi_importance, ppi_df, 20,
         _fp(figure_dir, "ppi_network"))
    _run(8, "PPI full map (all edges)",
         plot_full_ppi_map, ppi_df,
         _fp(figure_dir, "ppi_full_map"))

    # 9. Biological cascade — full (20×10 in) and compact (14×7 in)
    if cascade:
        _run(9, "Biological cascade (full)",
             plot_biological_cascade, cascade, gene_shap_df, pathway_df, ppi_df, p2p_df,
             _fp(figure_dir, "cascade_full"), False)
        _run(9, "Biological cascade (compact)",
             plot_biological_cascade, cascade, gene_shap_df, pathway_df, ppi_df, p2p_df,
             _fp(figure_dir, "cascade_compact"), True)
    else:
        print(f"  [9/{N}] Biological cascade                       – (hpv_cascade.json missing)")

    # 10. Model comparison (radar + boxplot + training)
    if fold_metrics:
        _run(10, "Model comparison radar",
             plot_model_comparison_radar, fold_metrics,
             _fp(figure_dir, "model_comparison_radar"))
        _run(10, "Model comparison boxplot",
             plot_model_comparison_boxplot, fold_metrics,
             stat_tests_df if not stat_tests_df.empty else None,
             _fp(figure_dir, "model_comparison_boxplot"))
    else:
        print(f"  [10/{N}] Model comparison radar                  – (nested_cv_results.json missing)")
        print(f"  [10/{N}] Model comparison boxplot                – (nested_cv_results.json missing)")
    if histories:
        _run(10, "Training curves",
             plot_training_curves, histories,
             _fp(figure_dir, "binn_training_curves"))
    else:
        print(f"  [10/{N}] Training curves                          – (training_histories.json missing)")

    # 12. Network sparsity (if fold network info available)
    import pickle as _pickle
    _fni_path = getattr(config, "FOLD_NETWORK_INFO_FILE", "")
    if _fni_path and os.path.exists(_fni_path):
        try:
            with open(_fni_path, "rb") as _f:
                _fni = _pickle.load(_f)
            if _fni and _fni[0] is not None:
                _conn = _fni[0].get("connectivity_matrices", [])
                _run(12, "Network sparsity",
                     plot_network_sparsity, _conn,
                     _fp(figure_dir, "network_sparsity"))
        except Exception:
            print(f"  [12/{N}] Network sparsity                         – (load failed)")
    else:
        print(f"  [12/{N}] Network sparsity                         – (fold_network_info.pkl missing)")

    print(f"\nAll figures saved to {figure_dir}/")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    generate_all_figures()
