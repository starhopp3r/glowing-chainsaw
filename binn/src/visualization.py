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
    models = [m for m in MODEL_ORDER if any(x["model"] == m for x in all_fold_metrics)]
    n = len(models)
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
    mean_abs = np.abs(shap_values).mean(0)
    top_idx  = np.argsort(mean_abs)[::-1][:top_n]
    # Bottom of chart = most important
    plot_order = top_idx[::-1]

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35 + 2)))
    rng = np.random.default_rng(0)

    for rank, gi in enumerate(plot_order):
        sv = shap_values[:, gi]
        jitter = rng.uniform(-0.28, 0.28, len(sv))
        y = rank + jitter

        if X_test is not None:
            fv = X_test[:, gi]
            mn, mx = fv.min(), fv.max()
            norm = (fv - mn) / (mx - mn + 1e-9)
            c = SHAP_CMAP(norm)
        else:
            c = _color("binn")

        ax.scatter(sv, y, c=c, s=14, alpha=0.75, zorder=5, linewidths=0)

    ax.axvline(0, color="black", lw=0.9, ls="--", alpha=0.6)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([gene_names[i] for i in plot_order], fontsize=10)
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
    df = gene_shap_df.sort_values("mean_abs_shap", ascending=False).head(top_n).iloc[::-1]
    n = len(df)

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

    if "ppi_importance_add" not in ppi_df.columns:
        log.warning("PPI DataFrame missing 'ppi_importance_add' — skipping plot.")
        return
    df = ppi_df.sort_values("ppi_importance_add", ascending=False).head(top_n)
    has_gene_shap = {"shap_gene_1", "shap_gene_2"}.issubset(df.columns)

    G = nx.Graph()
    for _, row in df.iterrows():
        g1, g2 = row["gene_1"], row["gene_2"]
        edge_w = _safe_num(row.get("ppi_importance_add"), 0.0)
        s1 = _safe_num(row.get("shap_gene_1"), edge_w / 2.0) if has_gene_shap else edge_w / 2.0
        s2 = _safe_num(row.get("shap_gene_2"), edge_w / 2.0) if has_gene_shap else edge_w / 2.0

        if g1 not in G:
            G.add_node(g1, shap=0.0)
        if g2 not in G:
            G.add_node(g2, shap=0.0)
        G.nodes[g1]["shap"] = max(_safe_num(G.nodes[g1].get("shap"), 0.0), s1)
        G.nodes[g2]["shap"] = max(_safe_num(G.nodes[g2].get("shap"), 0.0), s2)
        G.add_edge(g1, g2, weight=edge_w)

    if G.number_of_nodes() == 0:
        return

    try:
        pos = nx.kamada_kawai_layout(G)
    except Exception:
        pos = nx.spring_layout(G, seed=42, k=1.5)

    fig, ax = plt.subplots(figsize=(11, 9))

    edges = list(G.edges(data=True))
    if edges:
        edge_weights = np.array([d["weight"] for _, _, d in edges])
        ew_norm = (edge_weights - edge_weights.min()) / (edge_weights.max() - edge_weights.min() + 1e-9)
        for (u, v, d), en in zip(edges, ew_norm):
            nx.draw_networkx_edges(
                G, pos, edgelist=[(u, v)], ax=ax,
                width=1.0 + 5.0 * en, alpha=0.35 + 0.45 * en,
                edge_color="#555555",
            )

    shap_vals = np.array([G.nodes[n].get("shap", 0) for n in G.nodes()])
    sv_abs = np.abs(shap_vals)
    node_sizes = 300 + 1500 * (sv_abs / (sv_abs.max() + 1e-9))
    node_colors = SHAP_CMAP((shap_vals - shap_vals.min()) / (shap_vals.max() - shap_vals.min() + 1e-9))

    nx.draw_networkx_nodes(G, pos, ax=ax,
                           node_size=node_sizes, node_color=node_colors,
                           edgecolors="#333333", linewidths=0.8, alpha=0.9)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=9, font_color="#111111")

    sm = plt.cm.ScalarMappable(cmap=SHAP_CMAP,
                               norm=Normalize(shap_vals.min(), shap_vals.max()))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Mean |SHAP| (gene contribution)", fontsize=10)

    ax.set_title("Protein-Protein Interaction Network\n"
                 "(node size & color = SHAP importance, edge width = PPI score)",
                 fontsize=12)
    ax.axis("off")
    fig.tight_layout()
    if save_path:
        _save_fig(fig, save_path)
    else:
        plt.show()


# ── 9. Biological Cascade ─────────────────────────────────────────────────────

_DARK_BG   = "#0D1117"
_DARK_TEXT = "#E6EDF3"
_LIGHT_BG  = "white"
_LIGHT_TEXT = "#1a1a2a"

# SHAP → color for cascade nodes
_SHAP_POS  = "#D6604D"   # positive → HPV+
_SHAP_NEG  = "#4393C3"   # negative → HPV-
_PATHWAY_COLORS = ["#3A86FF", "#8338EC", "#FB5607", "#FF006E", "#06D6A0", "#FFBE0B"]


def _shap_to_color(shap_val: float, vmax: float = 1.0, dark: bool = False) -> str:
    t = 0.5 + 0.5 * np.clip(shap_val / (vmax + 1e-9), -1, 1)
    rgba = SHAP_CMAP(t)
    return mpl.colors.to_hex(rgba)


def _bezier_band(ax, x0, y0_c, h0, x1, y1_c, h1,
                 color="#888888", alpha=0.25, zorder=1):
    """Draw a smooth filled bezier band connecting two rectangular nodes."""
    t = np.linspace(0, 1, 80)
    xm = (x0 + x1) / 2

    def _bezier_y(ya, yb):
        # Cubic bezier: horizontal tangents at both endpoints
        return (1-t)**3*ya + 3*(1-t)**2*t*ya + 3*(1-t)*t**2*yb + t**3*yb

    y_top = _bezier_y(y0_c + h0/2, y1_c + h1/2)
    y_bot = _bezier_y(y0_c - h0/2, y1_c - h1/2)
    x_vals = xm * (3*(1-t)**2*t + 3*(1-t)*t**2) * 2 + (1-t)**3*x0 + t**3*x1

    poly_x = np.concatenate([x_vals, x_vals[::-1]])
    poly_y = np.concatenate([y_top, y_bot[::-1]])
    ax.fill(poly_x, poly_y, color=color, alpha=alpha, zorder=zorder,
            linewidth=0, antialiased=True)


def _layout_layer(items: list[dict], attr_key: str, x_pos: float,
                  fig_h: float = 10.0, pad: float = 0.6,
                  min_h: float = 0.25, max_h: float = 1.8) -> list[dict]:
    """Place nodes vertically for a cascade layer, sized by attr_key."""
    if not items:
        return []
    attrs = [max(item.get(attr_key, 0), 1e-6) for item in items]
    total = sum(attrs)
    avail = fig_h - 2*pad - max(0, len(items)-1) * 0.15
    nodes = []
    y = fig_h - pad
    for item, attr in zip(items, attrs):
        h = float(np.clip(avail * attr / total, min_h, max_h))
        y -= h
        nodes.append({"item": item, "x": x_pos, "y_c": y + h/2, "h": h, "attr": attr})
        y -= 0.15
    return nodes


def _draw_node(ax, x, y_c, h, w, label: str, color: str,
               text_color: str, dark: bool, fontsize: int = 8):
    """Draw a rounded rectangle node with label."""
    fancy = mpatches.FancyBboxPatch(
        (x - w/2, y_c - h/2), w, h,
        boxstyle="round,pad=0.04",
        facecolor=color, edgecolor=text_color,
        linewidth=0.6, alpha=0.9, zorder=5,
    )
    if dark:
        fancy.set_path_effects([
            pe.Stroke(linewidth=3, foreground=color, alpha=0.4),
            pe.Normal(),
        ])
    ax.add_patch(fancy)
    ax.text(x, y_c, _trunc(label, 30), ha="center", va="center",
            fontsize=fontsize, color=text_color, fontweight="medium",
            zorder=6, wrap=False, clip_on=True)


def _draw_cascade_panel(
    ax,
    gene_nodes: list[dict],
    leaf_nodes:  list[dict],
    inter_nodes: list[dict],
    root_nodes:  list[dict],
    output_node: dict,
    cascade_edges: list[tuple],
    dark: bool,
    node_w: float = 1.3,
) -> None:
    bg     = _DARK_BG   if dark else _LIGHT_BG
    txt_c  = _DARK_TEXT if dark else _LIGHT_TEXT
    ax.set_facecolor(bg)
    ax.figure.set_facecolor(bg)

    # Layer x positions
    layer_x = {
        "gene":   gene_nodes[0]["x"]  if gene_nodes  else 1.5,
        "leaf":   leaf_nodes[0]["x"]  if leaf_nodes  else 4.5,
        "inter":  inter_nodes[0]["x"] if inter_nodes else 8.0,
        "root":   root_nodes[0]["x"]  if root_nodes  else 11.5,
        "output": output_node["x"],
    }

    # Build edge lookup: (src_label, dst_label) → weight
    edge_w: dict[tuple, float] = {}
    if cascade_edges:
        for src, dst, wt in cascade_edges:
            edge_w[(str(src), str(dst))] = max(edge_w.get((str(src), str(dst)), 0), float(wt))

    def _find_node(label, layer):
        for n in layer:
            if n["item"].get("id", n["item"].get("gene", "")) == label or \
               n["item"].get("name", "") == label or \
               n["item"].get("gene", "") == label:
                return n
        return None

    # ── Draw connections (bezier bands) ──────────────────────────────────────
    max_attr = max([n["attr"] for n in gene_nodes + leaf_nodes + inter_nodes + root_nodes], default=1)

    def _connect_layers(src_layer, dst_layer, color_base="#888888"):
        for src_n in src_layer:
            for dst_n in dst_layer:
                sk = src_n["item"].get("id", src_n["item"].get("gene", ""))
                dk = dst_n["item"].get("id", dst_n["item"].get("gene", ""))
                wt = edge_w.get((sk, dk), edge_w.get((dk, sk), 0))
                if wt < 1e-9:
                    continue
                alpha = float(np.clip(0.10 + 0.50 * wt / max_attr, 0.05, 0.55))
                h_src = src_n["h"] * (wt / (src_n["attr"] + 1e-9))
                h_dst = dst_n["h"] * (wt / (dst_n["attr"] + 1e-9))
                _bezier_band(
                    ax,
                    src_n["x"] + node_w/2, src_n["y_c"], np.clip(h_src, 0.05, src_n["h"]),
                    dst_n["x"] - node_w/2, dst_n["y_c"], np.clip(h_dst, 0.05, dst_n["h"]),
                    color=color_base, alpha=alpha,
                )

    # Gene → leaf pathway connections
    _connect_layers(gene_nodes, leaf_nodes, "#4da6ff" if dark else "#90CAF9")
    _connect_layers(leaf_nodes, inter_nodes, "#7c5cbf" if dark else "#CE93D8")
    _connect_layers(inter_nodes, root_nodes, "#d68040" if dark else "#FFCC80")

    # Root → output
    for rn in root_nodes:
        alpha = float(np.clip(0.15 + 0.45 * rn["attr"] / max_attr, 0.1, 0.55))
        _bezier_band(
            ax,
            rn["x"] + node_w/2, rn["y_c"], rn["h"] * 0.5,
            output_node["x"] - node_w/2, output_node["y_c"], output_node["h"] * 0.5,
            color="#F18F01" if dark else "#FF8F00",
            alpha=alpha,
        )

    # ── Draw nodes ────────────────────────────────────────────────────────────
    shap_vals = [n["item"].get("shap", 0) for n in gene_nodes]
    vmax = max(abs(v) for v in shap_vals) if shap_vals else 1.0

    for n in gene_nodes:
        label = n["item"].get("gene", "?")
        shap  = n["item"].get("shap", 0)
        col   = _shap_to_color(shap, vmax, dark)
        _draw_node(ax, n["x"], n["y_c"], n["h"], node_w, label, col, txt_c, dark, fontsize=8)

    pw_palette = _PATHWAY_COLORS if dark else ["#1565C0","#283593","#311B92","#1A237E","#006064","#004D40"]
    for i, n in enumerate(leaf_nodes):
        label = _trunc(n["item"].get("name", "?"), 28)
        col   = pw_palette[i % len(pw_palette)]
        _draw_node(ax, n["x"], n["y_c"], n["h"], node_w + 0.4, label, col, txt_c, dark, fontsize=7)

    for i, n in enumerate(inter_nodes):
        label = _trunc(n["item"].get("name", "?"), 28)
        col   = pw_palette[(i + 2) % len(pw_palette)]
        _draw_node(ax, n["x"], n["y_c"], n["h"], node_w + 0.4, label, col, txt_c, dark, fontsize=7)

    for i, n in enumerate(root_nodes):
        label = _trunc(n["item"].get("name", "?"), 28)
        col   = pw_palette[(i + 4) % len(pw_palette)]
        _draw_node(ax, n["x"], n["y_c"], n["h"], node_w + 0.4, label, col, txt_c, dark, fontsize=8)

    out_col = "#F18F01" if dark else "#E65100"
    _draw_node(ax, output_node["x"], output_node["y_c"],
               output_node["h"], node_w + 0.4,
               "HPV+/HPV−\nPrediction", out_col, "white" if dark else "white", dark, fontsize=9)

    # ── Layer labels ──────────────────────────────────────────────────────────
    fig_h = ax.get_ylim()[1]
    label_y = fig_h + 0.3
    for label, x in [("Genes", layer_x["gene"]), ("Leaf\nPathways", layer_x["leaf"]),
                     ("Intermediate\nPathways", layer_x["inter"]),
                     ("Root\nPathways", layer_x["root"]), ("Output", layer_x["output"])]:
        ax.text(x, label_y, label, ha="center", va="bottom", fontsize=10,
                fontweight="bold", color=txt_c, zorder=7)

    ax.set_xlim(layer_x["gene"] - node_w, layer_x["output"] + node_w)
    ax.set_ylim(-0.5, fig_h + 1.2)
    ax.axis("off")


def plot_biological_cascade(
    cascade: dict,
    bio_map: dict | None = None,
    save_path: str | None = None,
) -> None:
    """
    Two-version biological cascade figure (dark + light).

    Reads from cascade dict produced by reconstruct_hpv_cascade().
    """
    top_genes = cascade.get("top_genes", [])[:15]
    top_leaf   = cascade.get("top_leaf_pathways", [])[:10]
    top_inter  = cascade.get("top_intermediate_pathways", [])[:8]
    top_root   = cascade.get("top_root_pathways", [])[:6]
    edges      = cascade.get("cascade_edges", [])

    if not top_genes:
        log.warning("Cascade has no gene data — skipping cascade plot.")
        return

    FIG_H = 11.0
    X_GENE  = 1.5
    X_LEAF  = 4.8
    X_INTER = 8.2
    X_ROOT  = 11.6
    X_OUT   = 14.5

    gene_nodes  = _layout_layer(top_genes, "shap", X_GENE, FIG_H)
    leaf_nodes  = _layout_layer(top_leaf,  "attribution", X_LEAF, FIG_H)
    inter_nodes = _layout_layer(top_inter, "attribution", X_INTER, FIG_H)
    root_nodes  = _layout_layer(top_root,  "attribution", X_ROOT, FIG_H)
    output_node = {"x": X_OUT, "y_c": FIG_H/2, "h": 1.5, "attr": 1.0,
                   "item": {"name": "HPV+/HPV−"}}

    for dark, tag in [(True, "dark"), (False, "light")]:
        fig, ax = plt.subplots(figsize=(18, 11))
        ax.set_xlim(0, 16)
        ax.set_ylim(-0.5, FIG_H + 1.5)

        _draw_cascade_panel(
            ax, gene_nodes, leaf_nodes, inter_nodes, root_nodes, output_node,
            edges, dark=dark,
        )

        title_col = _DARK_TEXT if dark else _LIGHT_TEXT
        ax.set_title(
            "Biological Cascade: HPV-Driven HNSCC Proliferation Pathways Revealed by BINN",
            fontsize=13, pad=14, color=title_col, fontweight="bold",
        )

        # SHAP color bar legend
        sm = plt.cm.ScalarMappable(cmap=SHAP_CMAP, norm=Normalize(-1, 1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation="horizontal",
                            fraction=0.025, pad=0.01, aspect=40,
                            location="bottom")
        cbar.set_label("Gene SHAP Value (← HPV−  |  HPV+ →)", fontsize=9,
                       color=title_col)
        cbar.ax.tick_params(colors=title_col, labelsize=8)

        fig.patch.set_facecolor(_DARK_BG if dark else _LIGHT_BG)
        fig.tight_layout(rect=[0, 0.04, 1, 0.97])

        sp = save_path or os.path.join(config.FIGURE_DIR, "biological_cascade")
        base = sp.replace("biological_cascade", f"biological_cascade_{tag}")
        if save_path:
            _save_fig(fig, base)
        else:
            plt.show()
            plt.close(fig)


# ── 9b. Sankey-like Biological Cascade (gene→protein→PPI→pathways→output) ───

_FLOW_LAYER_COLORS_DARK = {
    "gene": "#7AA2F7",
    "protein": "#2EC4B6",
    "ppi": "#FF7F50",
    "group": "#C77DFF",
    "leaf": "#4CC9F0",
    "inter": "#4895EF",
    "root": "#4361EE",
    "output": "#F8961E",
}

_FLOW_LAYER_COLORS_LIGHT = {
    "gene": "#4F6D9B",
    "protein": "#2A9D8F",
    "ppi": "#E76F51",
    "group": "#9B5DE5",
    "leaf": "#1D6996",
    "inter": "#2A6F97",
    "root": "#184E77",
    "output": "#BC6C25",
}

_FLOW_LAYER_ORDER = ["gene", "protein", "ppi", "group", "leaf", "inter", "root", "output"]


def _safe_num(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _blend_hex(c1: str, c2: str, t: float) -> str:
    """Linear blend between two hex colors."""
    t = float(np.clip(t, 0.0, 1.0))
    a = np.array(mpl.colors.to_rgb(c1), dtype=float)
    b = np.array(mpl.colors.to_rgb(c2), dtype=float)
    out = (1.0 - t) * a + t * b
    return mpl.colors.to_hex(out)


def _contrast_text(hex_color: str, dark: bool) -> str:
    """Pick readable text color for a filled node color."""
    r, g, b = mpl.colors.to_rgb(hex_color)
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    if dark:
        return "#EAF2FF" if lum < 0.52 else "#102038"
    return "#FFFFFF" if lum < 0.45 else "#17253A"


def _build_flow_graph_from_cascade(
    cascade: dict,
    max_genes: int = 14,
    max_ppis: int = 24,
    max_leaf: int = 12,
    max_inter: int = 6,
    max_root: int = 8,
) -> tuple[dict[str, dict], list[dict], list[tuple[str, str, float]]]:
    """
    Build a layered flow graph:
    genes -> proteins -> PPI pairs -> pathway-groups -> leaf -> inter -> root -> output.
    """
    top_genes = cascade.get("top_genes", [])[:max_genes]
    top_ppis = cascade.get("top_ppis", [])[:max_ppis]
    top_leaf = cascade.get("top_leaf_pathways", [])[:max_leaf]
    all_inter = cascade.get("top_intermediate_pathways", [])
    top_root = cascade.get("top_root_pathways", [])[:max_root]
    cascade_edges = cascade.get("cascade_edges", [])
    top_inter = list(all_inter[:max_inter])

    # Ensure intermediate pathways directly referenced by cascade edges are retained.
    inter_rows = {
        str(p.get("id", "")).strip(): p
        for p in all_inter
        if str(p.get("id", "")).strip()
    }
    selected_inter_ids = {str(p.get("id", "")).strip() for p in top_inter}
    for src, dst, _ in cascade_edges:
        src_id, dst_id = str(src).strip(), str(dst).strip()
        for iid in (src_id, dst_id):
            if iid in inter_rows and iid not in selected_inter_ids:
                top_inter.append(inter_rows[iid])
                selected_inter_ids.add(iid)

    nodes: dict[str, dict] = {}
    edges_map: dict[tuple[str, str], float] = {}

    def _add_edge(src: str, dst: str, weight: float) -> None:
        w = max(float(weight), 0.0)
        if w <= 0:
            return
        edges_map[(src, dst)] = edges_map.get((src, dst), 0.0) + w

    # Genes + proteins
    for g in top_genes:
        gene = str(g.get("gene", "")).strip()
        if not gene:
            continue
        shap = _safe_num(g.get("shap"), 0.0)
        protein = str(g.get("protein") or gene).strip() or gene
        uniprot = g.get("uniprot")

        gid = f"gene::{gene}"
        pid = f"protein::{gene}"
        p_label = protein if not uniprot else f"{protein} ({uniprot})"

        nodes[gid] = {
            "id": gid,
            "kind": "gene",
            "label": gene,
            "value": max(abs(shap), 1e-9),
            "shap": shap,
        }
        nodes[pid] = {
            "id": pid,
            "kind": "protein",
            "label": _trunc(p_label, 38),
            "value": max(abs(shap), 1e-9),
            "gene": gene,
        }
        _add_edge(gid, pid, max(abs(shap), 1e-9))

    # Pathway hierarchy nodes
    leaf_ids = set()
    inter_ids = set()
    root_ids = set()
    name_to_leaf: dict[str, str] = {}
    name_to_inter: dict[str, str] = {}
    name_to_root: dict[str, str] = {}

    for p in top_leaf:
        pid = str(p.get("id", "")).strip()
        if not pid:
            continue
        name = str(p.get("name", pid))
        leaf_ids.add(pid)
        name_to_leaf[name] = pid
        nodes[f"leaf::{pid}"] = {
            "id": f"leaf::{pid}",
            "kind": "leaf",
            "label": _trunc(name, 44),
            "value": max(_safe_num(p.get("attribution"), 0.0), 1e-9),
            "pathway_id": pid,
            "pathway_name": name,
        }

    for p in top_inter:
        pid = str(p.get("id", "")).strip()
        if not pid:
            continue
        name = str(p.get("name", pid))
        inter_ids.add(pid)
        name_to_inter[name] = pid
        nodes[f"inter::{pid}"] = {
            "id": f"inter::{pid}",
            "kind": "inter",
            "label": _trunc(name, 44),
            "value": max(_safe_num(p.get("attribution"), 0.0), 1e-9),
            "pathway_id": pid,
            "pathway_name": name,
        }

    for p in top_root:
        pid = str(p.get("id", "")).strip()
        if not pid:
            continue
        name = str(p.get("name", pid))
        root_ids.add(pid)
        name_to_root[name] = pid
        nodes[f"root::{pid}"] = {
            "id": f"root::{pid}",
            "kind": "root",
            "label": _trunc(name, 44),
            "value": max(_safe_num(p.get("attribution"), 0.0), 1e-9),
            "pathway_id": pid,
            "pathway_name": name,
        }

    # PPI pair nodes and pathway-group nodes
    pathway_group_weights: dict[str, float] = {}
    for row in top_ppis:
        g1 = str(row.get("gene1", "")).strip()
        g2 = str(row.get("gene2", "")).strip()
        if not g1 or not g2:
            continue
        p1 = f"protein::{g1}"
        p2 = f"protein::{g2}"
        if p1 not in nodes or p2 not in nodes:
            continue

        imp_raw = _safe_num(row.get("importance"), 0.0)
        if imp_raw <= 0:
            continue
        imp = max(imp_raw, 1e-9)
        pair = tuple(sorted((g1, g2)))
        ppi_id = f"ppi::{pair[0]}::{pair[1]}"
        if ppi_id not in nodes:
            nodes[ppi_id] = {
                "id": ppi_id,
                "kind": "ppi",
                "label": _trunc(f"{pair[0]} ↔ {pair[1]}", 36),
                "value": imp,
            }
        else:
            nodes[ppi_id]["value"] = max(_safe_num(nodes[ppi_id].get("value"), 0.0), imp)

        _add_edge(p1, ppi_id, imp / 2.0)
        _add_edge(p2, ppi_id, imp / 2.0)

        shared = row.get("shared_pathways", []) or []
        if not isinstance(shared, list):
            shared = [str(shared)]
        group_name = None
        # Prefer pathways already present in the selected hierarchy layers.
        for s in shared:
            s_name = str(s).strip()
            if s_name in name_to_leaf or s_name in name_to_inter or s_name in name_to_root:
                group_name = s_name
                break
        if group_name is None and shared:
            group_name = str(shared[0]).strip()
        if not group_name:
            group_name = "Unassigned PPI pathway context"

        gid = f"group::{group_name}"
        if gid not in nodes:
            nodes[gid] = {
                "id": gid,
                "kind": "group",
                "label": _trunc(group_name, 44),
                "value": 0.0,
                "pathway_name": group_name,
            }
        nodes[gid]["value"] = _safe_num(nodes[gid]["value"], 0.0) + imp
        pathway_group_weights[group_name] = pathway_group_weights.get(group_name, 0.0) + imp
        _add_edge(ppi_id, gid, imp)

    # Connect pathway-group nodes into the hierarchy.
    fallback_leaf = next(iter(leaf_ids), None)
    fallback_inter = next(iter(inter_ids), None)
    for group_name, w in pathway_group_weights.items():
        gid = f"group::{group_name}"
        if group_name in name_to_leaf:
            _add_edge(gid, f"leaf::{name_to_leaf[group_name]}", w)
        elif group_name in name_to_inter:
            _add_edge(gid, f"inter::{name_to_inter[group_name]}", w)
        elif group_name in name_to_root:
            _add_edge(gid, f"root::{name_to_root[group_name]}", w)
        elif fallback_leaf is not None:
            # Keep pathway-group context in-flow even when pathway naming differs.
            _add_edge(gid, f"leaf::{fallback_leaf}", 0.7 * w)
        elif fallback_inter is not None:
            _add_edge(gid, f"inter::{fallback_inter}", 0.7 * w)

    # Add hierarchy edges from stored cascade edges (id-based).
    for src, dst, wt in cascade_edges:
        src_id = str(src)
        dst_id = str(dst)
        w = max(_safe_num(wt, 0.0), 0.0)
        if w <= 0:
            continue
        if src_id in leaf_ids and dst_id in inter_ids:
            _add_edge(f"leaf::{src_id}", f"inter::{dst_id}", w)
        elif src_id in leaf_ids and dst_id in root_ids:
            _add_edge(f"leaf::{src_id}", f"root::{dst_id}", w)
        elif src_id in inter_ids and dst_id in root_ids:
            _add_edge(f"inter::{src_id}", f"root::{dst_id}", w)
        elif dst_id in leaf_ids:
            # Stored cascade uses gene -> leaf edges; route via protein layer.
            prot = f"protein::{src_id}"
            gene = f"gene::{src_id}"
            target = f"leaf::{dst_id}"
            if prot in nodes:
                _add_edge(prot, target, w)
            elif gene in nodes:
                _add_edge(gene, target, w)

    # Ensure hierarchy continuity for display: leaf -> inter/root and inter -> root.
    best_inter = None
    best_root = None
    if inter_ids:
        best_inter = max(inter_ids, key=lambda i: _safe_num(nodes.get(f"inter::{i}", {}).get("value"), 0.0))
    if root_ids:
        best_root = max(root_ids, key=lambda i: _safe_num(nodes.get(f"root::{i}", {}).get("value"), 0.0))

    def _has_out(src: str, prefixes: tuple[str, ...]) -> bool:
        return any(s == src and any(d.startswith(p) for p in prefixes) for (s, d) in edges_map)

    for lid in leaf_ids:
        src = f"leaf::{lid}"
        if _has_out(src, ("inter::", "root::")):
            continue
        w = max(_safe_num(nodes.get(src, {}).get("value"), 0.0), 1e-9)
        if best_inter is not None:
            _add_edge(src, f"inter::{best_inter}", w)
        elif best_root is not None:
            _add_edge(src, f"root::{best_root}", w)

    if best_root is not None:
        for iid in inter_ids:
            src = f"inter::{iid}"
            if _has_out(src, ("root::",)):
                continue
            w = max(_safe_num(nodes.get(src, {}).get("value"), 0.0), 1e-9)
            _add_edge(src, f"root::{best_root}", w)

    # Ensure all roots connect to OUTPUT.
    out_id = "output::HPV"
    nodes[out_id] = {
        "id": out_id,
        "kind": "output",
        "label": "HPV+/HPV− Output",
        "value": 1.0,
    }
    for rid in root_ids:
        src = f"root::{rid}"
        w = max(_safe_num(nodes[src].get("value"), 0.0), 1e-9)
        _add_edge(src, out_id, w)

    edges = [(s, d, w) for (s, d), w in edges_map.items() if s in nodes and d in nodes and w > 0]
    if not edges:
        return {}, [], []

    # Keep only nodes on at least one directed path to OUTPUT to avoid clutter.
    flow_graph = nx.DiGraph()
    for s, d, w in edges:
        flow_graph.add_edge(s, d, weight=w)
    if out_id in flow_graph:
        keep = nx.ancestors(flow_graph, out_id) | {out_id}
        edges = [(s, d, w) for s, d, w in edges if s in keep and d in keep]
        nodes = {nid: n for nid, n in nodes.items() if nid in keep}

    if not edges or not nodes:
        return {}, [], []

    # Refresh node values from in/out flow magnitude (after pruning).
    in_flow: dict[str, float] = {}
    out_flow: dict[str, float] = {}
    for s, d, w in edges:
        out_flow[s] = out_flow.get(s, 0.0) + w
        in_flow[d] = in_flow.get(d, 0.0) + w
    for nid, n in nodes.items():
        base = _safe_num(n.get("value"), 0.0)
        n["value"] = max(base, in_flow.get(nid, 0.0), out_flow.get(nid, 0.0), 1e-9)

    layers: list[dict] = []
    for kind in _FLOW_LAYER_ORDER:
        layer_nodes = [n for n in nodes.values() if n.get("kind") == kind]
        layer_nodes.sort(key=lambda x: _safe_num(x.get("value"), 0.0), reverse=True)
        if layer_nodes:
            layers.append({"kind": kind, "nodes": layer_nodes})

    return nodes, layers, edges


def _draw_sankey_cascade_panel(
    ax: plt.Axes,
    nodes: dict[str, dict],
    layers: list[dict],
    edges: list[tuple[str, str, float]],
    dark: bool,
) -> None:
    if not layers or not edges:
        return

    bg = _DARK_BG if dark else _LIGHT_BG
    txt = _DARK_TEXT if dark else _LIGHT_TEXT
    layer_cols = _FLOW_LAYER_COLORS_DARK if dark else _FLOW_LAYER_COLORS_LIGHT
    ax.set_facecolor(bg)
    ax.figure.set_facecolor(bg)

    # X positions distributed by existing layers only.
    n_layers = len(layers)
    x_positions = np.linspace(1.3, 16.7, n_layers)
    fig_h = 12.0
    x_min, x_max = 0.2, 17.8
    y_min, y_max = -0.6, fig_h + 1.3

    # Gradient panel background for stronger visual depth.
    c_l = "#040A16" if dark else "#F7FAFC"
    c_r = "#132543" if dark else "#E9EFF7"
    grad = np.linspace(0.0, 1.0, 1024)
    left = np.array(mpl.colors.to_rgb(c_l), dtype=float)
    right = np.array(mpl.colors.to_rgb(c_r), dtype=float)
    grad_rgb = (1 - grad)[None, :, None] * left + grad[None, :, None] * right
    grad_rgb = np.repeat(grad_rgb, 4, axis=0)
    ax.imshow(grad_rgb, extent=(x_min, x_max, y_min, y_max), aspect="auto", zorder=0)

    # Subtle layer guide lines.
    for x in x_positions:
        ax.plot([x, x], [0.1, fig_h - 0.1], lw=0.8, color=txt, alpha=0.08, zorder=1)

    # Aggregate flow totals for cleaner band thickness allocation.
    in_flow: dict[str, float] = {}
    out_flow: dict[str, float] = {}
    for src, dst, w in edges:
        out_flow[src] = out_flow.get(src, 0.0) + w
        in_flow[dst] = in_flow.get(dst, 0.0) + w

    # Layout nodes layer by layer.
    draw_nodes: list[dict] = []
    node_lookup: dict[str, dict] = {}
    node_w: dict[str, float] = {}
    width_by_kind = {
        "gene": 1.25, "protein": 1.45, "ppi": 1.65, "group": 1.80,
        "leaf": 1.95, "inter": 2.00, "root": 2.05, "output": 1.90,
    }

    for i, layer in enumerate(layers):
        kind = layer["kind"]
        for rank, n in enumerate(layer["nodes"]):
            n["value"] = max(_safe_num(n.get("value"), 0.0), 1e-9)
            # Compress dynamic range to avoid tiny unreadable boxes.
            n["_layout_value"] = max(np.sqrt(n["value"]), 1e-9)
            n["_rank"] = rank
        laid = _layout_layer(
            layer["nodes"], "_layout_value", x_positions[i], fig_h=fig_h, pad=0.75,
            min_h=0.30 if kind != "output" else 1.15,
            max_h=2.15 if kind != "output" else 2.30,
        )
        for ln in laid:
            nid = ln["item"]["id"]
            ln["flow_in"] = in_flow.get(nid, 0.0)
            ln["flow_out"] = out_flow.get(nid, 0.0)
            draw_nodes.append(ln)
            node_lookup[nid] = ln
            node_w[nid] = width_by_kind.get(kind, 1.5)

    max_w = max((w for _, _, w in edges), default=1.0)
    max_w = max(max_w, 1e-9)

    # Draw weighted flow bands (thin first, thick last for visual prominence).
    for src, dst, w in sorted(edges, key=lambda x: x[2]):
        if src not in node_lookup or dst not in node_lookup:
            continue
        if w <= 0:
            continue
        s = node_lookup[src]
        d = node_lookup[dst]
        s_kind = s["item"].get("kind", "gene")
        base_col = layer_cols.get(s_kind, "#8A8A8A")
        col = _blend_hex(base_col, "#EAF2FF" if dark else "#FFFFFF", 0.10 if dark else 0.18)

        s_tot = max(_safe_num(s.get("flow_out"), 0.0), 1e-9)
        d_tot = max(_safe_num(d.get("flow_in"), 0.0), 1e-9)
        h_s = np.clip(s["h"] * (w / s_tot), 0.04, s["h"] * 0.95)
        h_d = np.clip(d["h"] * (w / d_tot), 0.04, d["h"] * 0.95)

        flow_strength = np.sqrt(w / max_w)
        alpha = float(np.clip(0.09 + 0.56 * flow_strength, 0.08, 0.70))
        _bezier_band(
            ax,
            s["x"] + node_w[src] / 2.0, s["y_c"], h_s,
            d["x"] - node_w[dst] / 2.0, d["y_c"], h_d,
            color=col, alpha=alpha, zorder=2,
        )

    # Draw nodes with improved contrast and subtle glow.
    shap_vals = [n.get("shap", 0.0) for n in nodes.values() if n.get("kind") == "gene"]
    vmax = max([abs(v) for v in shap_vals], default=1.0)
    vmax = max(vmax, 1e-12)

    for dn in draw_nodes:
        item = dn["item"]
        nid = item["id"]
        kind = item.get("kind", "gene")
        label = _trunc(str(item.get("label", nid)), 38 if kind in {"group", "inter", "root"} else 28)
        width = node_w[nid]

        if kind == "gene":
            fill = _shap_to_color(_safe_num(item.get("shap"), 0.0), vmax, dark)
        elif kind == "output":
            fill = layer_cols.get("output", "#F8961E")
        else:
            base = layer_cols.get(kind, "#888888")
            rank = int(item.get("_rank", 0))
            fill = _blend_hex(base, "#FFFFFF", 0.06 * min(rank, 5)) if dark else _blend_hex(base, "#FFFFFF", 0.10 * min(rank, 4))

        edge = _blend_hex(fill, "#FFFFFF" if dark else "#1E2B43", 0.45 if dark else 0.38)
        tcol = "white" if kind == "output" else _contrast_text(fill, dark)

        box = mpatches.FancyBboxPatch(
            (dn["x"] - width / 2.0, dn["y_c"] - dn["h"] / 2.0),
            width, dn["h"],
            boxstyle="round,pad=0.045,rounding_size=0.06",
            facecolor=fill,
            edgecolor=edge,
            linewidth=1.05,
            alpha=0.96,
            zorder=6,
        )
        if dark:
            box.set_path_effects([
                pe.Stroke(linewidth=3.0, foreground=_blend_hex(fill, "#9FC5FF", 0.45), alpha=0.28),
                pe.Normal(),
            ])
        ax.add_patch(box)

        if dn["h"] >= 0.24:
            fs = float(np.clip(7.0 + 1.6 * (dn["h"] / 1.3), 6.6, 9.6))
            txt_fx = [pe.withStroke(linewidth=1.4, foreground="#000000", alpha=0.18)] if dark else []
            ax.text(
                dn["x"], dn["y_c"], label,
                ha="center", va="center",
                fontsize=fs,
                color=tcol,
                fontweight="semibold" if kind in {"root", "output", "leaf"} else "medium",
                zorder=7,
                clip_on=True,
                path_effects=txt_fx,
            )

    # Layer captions.
    labels = {
        "gene": "Genes",
        "protein": "Proteins",
        "ppi": "PPI Pairs",
        "group": "Pathway Groups",
        "leaf": "Leaf Pathways",
        "inter": "Intermediate",
        "root": "High-Level",
        "output": "Output",
    }
    for i, layer in enumerate(layers):
        kind = layer["kind"]
        cap = f"{labels.get(kind, kind)}\n(n={len(layer['nodes'])})"
        cap_col = _blend_hex(layer_cols.get(kind, txt), "#FFFFFF" if dark else "#101828", 0.30 if dark else 0.15)
        ax.text(
            x_positions[i], fig_h + 0.48, cap,
            ha="center", va="bottom", fontsize=9.4,
            fontweight="bold", color=cap_col, zorder=8,
            path_effects=[pe.withStroke(linewidth=1.1, foreground=bg, alpha=0.75)],
        )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.axis("off")


def plot_biological_cascade_sankeyish(
    cascade: dict,
    save_path: str | None = None,
) -> None:
    """Sankey-like biological cascade: gene -> protein -> PPI -> pathway hierarchy -> output."""
    nodes, layers, edges = _build_flow_graph_from_cascade(cascade)
    if not nodes or not layers or not edges:
        log.warning("Insufficient cascade/PPI data for Sankey-like cascade plot.")
        return

    for dark, tag in [(True, "dark"), (False, "light")]:
        fig, ax = plt.subplots(figsize=(20.5, 12.2))
        _draw_sankey_cascade_panel(ax, nodes, layers, edges, dark=dark)
        title_color = _DARK_TEXT if dark else _LIGHT_TEXT
        ax.set_title(
            "HPV Biological Cascade Flow",
            fontsize=15, pad=20, color=title_color, fontweight="bold",
        )
        ax.text(
            0.5, 1.012,
            "Gene -> Protein -> PPI interactions grouped by pathway context -> Reactome hierarchy -> Model output",
            transform=ax.transAxes,
            ha="center", va="bottom",
            fontsize=10.1,
            color=_blend_hex(title_color, "#8FA3BF" if dark else "#5A6B84", 0.30),
            fontweight="medium",
        )

        sm = plt.cm.ScalarMappable(cmap=SHAP_CMAP, norm=Normalize(-1, 1))
        sm.set_array([])
        cbar = plt.colorbar(
            sm, ax=ax, orientation="horizontal", fraction=0.025,
            pad=0.015, aspect=48, location="bottom"
        )
        cbar.set_label("Gene SHAP Value (HPV− <- -> HPV+)", fontsize=9.3, color=title_color)
        cbar.ax.tick_params(colors=title_color, labelsize=8)
        cbar.outline.set_edgecolor(_blend_hex(title_color, "#FFFFFF" if dark else "#2E3A4D", 0.55))
        cbar.outline.set_linewidth(0.8)

        fig.patch.set_facecolor(_DARK_BG if dark else _LIGHT_BG)
        fig.tight_layout(rect=[0, 0.04, 1, 0.97])

        sp = save_path or os.path.join(config.FIGURE_DIR, "biological_cascade_sankey")
        base = sp.replace("biological_cascade_sankey", f"biological_cascade_sankey_{tag}")
        if save_path:
            _save_fig(fig, base)
        else:
            plt.show()
            plt.close(fig)


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

        # Downsample if very large
        max_dim = 200
        if mat.shape[0] > max_dim or mat.shape[1] > max_dim:
            from skimage.transform import downscale_local_mean
            fy = max(1, mat.shape[0] // max_dim)
            fx = max(1, mat.shape[1] // max_dim)
            mat = downscale_local_mean(mat.astype(float), (fy, fx))

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

    N = 10
    status = {}

    def _run(step: int, label: str, fn, *args, **kwargs):
        try:
            fn(*args, **kwargs)
            print(f"  [{step}/{N}] {label:<40s} ✓")
            status[label] = "ok"
        except Exception as exc:
            print(f"  [{step}/{N}] {label:<40s} ✗  ({exc})")
            status[label] = str(exc)

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

    # 5. SHAP beeswarm (requires raw SHAP values — skip if not in memory)
    if not gene_shap_df.empty and "mean_abs_shap" in gene_shap_df.columns:
        sv_proxy = np.diag(gene_shap_df["mean_abs_shap"].values)  # proxy 2-D
        _run(5, "SHAP beeswarm",
             plot_shap_beeswarm, sv_proxy,
             gene_shap_df["gene"].tolist() if "gene" in gene_shap_df else [],
             None, 30, _fp(figure_dir, "shap_beeswarm"))
    else:
        print(f"  [5/{N}] SHAP beeswarm                           – (gene_shap_values.csv missing)")

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

    # 9. Biological cascade (sankey-like, dark + light)
    if cascade:
        _run(9, "Biological cascade Sankey-like",
             plot_biological_cascade_sankeyish, cascade,
             _fp(figure_dir, "biological_cascade_sankey"))
    else:
        print(f"  [9/{N}] Biological cascade                       – (hpv_cascade.json missing)")

    # 10. Model comparison (radar + boxplot + training)
    _run(10, "Model comparison radar",
         plot_model_comparison_radar, fold_metrics,
         _fp(figure_dir, "model_comparison_radar"))
    _run(10, "Model comparison boxplot",
         plot_model_comparison_boxplot, fold_metrics,
         stat_tests_df if not stat_tests_df.empty else None,
         _fp(figure_dir, "model_comparison_boxplot"))
    _run(10, "Training curves",
         plot_training_curves, histories,
         _fp(figure_dir, "binn_training_curves"))

    print(f"\nAll figures saved to {figure_dir}/")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    generate_all_figures()
