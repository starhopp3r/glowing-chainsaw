"""
Nested cross-validation pipeline for BINN vs baseline classifiers.

Outer loop (5-fold stratified)  — estimates generalisation performance.
Inner loop (3-fold via GridSearchCV) — tunes baseline hyperparameters.

All preprocessing (MAD filtering, standardisation) is fit exclusively on the
outer training fold to prevent information leakage.
"""
from __future__ import annotations

import json
import logging
import os
import pickle
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy import stats as sp_stats
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.baselines import BaselineWrapper, _MODEL_NAMES
from src.binn_model import BINN
from src.data_acquisition import apply_mad_filter, compute_mad, load_preprocessed_data
from src.network_builder import build_fold_network
from src.training import BINNTrainer

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# ── Device safety ─────────────────────────────────────────────────────────────

def _select_binn_device(connectivity_matrices: list[torch.Tensor]) -> torch.device:
    """
    Choose a safe training device for BINN based on largest mask size.

    Large dense masked layers can exceed MPS/CUDA allocation limits, so this
    falls back to CPU when the largest matrix crosses MAX_DEVICE_MATRIX_GIB.
    """
    preferred = config.DEVICE
    if preferred.type == "cpu" or not connectivity_matrices:
        return preferred

    max_elems = max(C.numel() for C in connectivity_matrices)
    max_gib = max_elems * 4.0 / (1024**3)  # float32 dense layer size
    limit_gib = float(getattr(config, "MAX_DEVICE_MATRIX_GIB", 1.5))
    if max_gib > limit_gib:
        log.warning(
            f"Largest BINN layer would be ~{max_gib:.2f} GiB on {preferred.type}; "
            "falling back to CPU for this fold."
        )
        return torch.device("cpu")
    if preferred.type != "cpu":
        log.info(
            f"Using {preferred.type} for BINN (largest layer ~{max_gib:.2f} GiB)."
        )
    return preferred


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_fold_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    fold: int,
) -> dict:
    """
    Compute a comprehensive set of classification metrics for a single fold.

    Parameters
    ----------
    y_true  : (n,) binary ground-truth labels
    y_prob  : (n,) predicted probabilities for the positive class
    y_pred  : (n,) binary predicted labels
    model_name : identifier string
    fold    : 0-based fold index

    Returns
    -------
    dict with scalar metrics, curve arrays, and raw predictions.
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = (cm.ravel() if cm.size == 4 else (0, 0, 0, 0))
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    prec_arr, rec_arr, _ = precision_recall_curve(y_true, y_prob)

    # Specificity-sensitivity curve at the same thresholds as ROC
    sensitivity_arr = tpr          # TPR at each threshold
    specificity_arr = 1.0 - fpr   # TNR at each threshold

    auroc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
    auprc = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0

    return {
        "model": model_name,
        "fold": fold,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(specificity),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auroc": float(auroc),
        "auprc": float(auprc),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "confusion_matrix": cm.tolist(),
        "fpr_array": fpr.tolist(),
        "tpr_array": tpr.tolist(),
        "precision_array": prec_arr.tolist(),
        "recall_array": rec_arr.tolist(),
        "specificity_array": specificity_arr.tolist(),
        "sensitivity_array": sensitivity_arr.tolist(),
        "y_true": y_true.tolist(),
        "y_prob": y_prob.tolist(),
        "y_pred": y_pred.tolist(),
    }


# ── Aggregation ───────────────────────────────────────────────────────────────

_SUMMARY_METRICS = ["auroc", "auprc", "accuracy", "balanced_accuracy", "f1", "mcc",
                    "precision", "recall", "specificity", "brier_score"]


def aggregate_results(all_fold_metrics: list[dict]) -> pd.DataFrame:
    """
    Produce a summary DataFrame with mean ± std per model per metric.

    Parameters
    ----------
    all_fold_metrics : flat list of dicts from compute_fold_metrics()

    Returns
    -------
    pd.DataFrame indexed by model name, columns = metric names.
    Formatted as "mean ± std" strings for display.
    """
    df = pd.DataFrame(all_fold_metrics)
    scalar_df = df[[col for col in df.columns if col in _SUMMARY_METRICS + ["model"]]]

    rows = []
    for model_name, grp in scalar_df.groupby("model"):
        row = {"model": model_name}
        for metric in _SUMMARY_METRICS:
            if metric in grp.columns:
                mu = grp[metric].mean()
                sd = grp[metric].std(ddof=1)
                row[metric] = f"{mu:.4f} ± {sd:.4f}"
        rows.append(row)

    summary = pd.DataFrame(rows).set_index("model")
    return summary


# ── Statistical tests ─────────────────────────────────────────────────────────

def run_statistical_tests(
    all_fold_metrics: list[dict],
    reference_model: str = "binn",
) -> pd.DataFrame:
    """
    Paired t-test and Wilcoxon signed-rank test comparing each model's
    per-fold AUROC against the reference model (default: BINN).

    Returns
    -------
    pd.DataFrame with columns: model, t_stat, t_pvalue, w_stat, w_pvalue.
    """
    df = pd.DataFrame(all_fold_metrics)
    ref_auroc = (
        df[df["model"] == reference_model]
        .sort_values("fold")["auroc"]
        .values
    )

    rows = []
    for model_name, grp in df.groupby("model"):
        if model_name == reference_model:
            continue
        cmp_auroc = grp.sort_values("fold")["auroc"].values
        n = min(len(ref_auroc), len(cmp_auroc))
        diff = ref_auroc[:n] - cmp_auroc[:n]

        t_stat, t_p = sp_stats.ttest_rel(ref_auroc[:n], cmp_auroc[:n])
        try:
            w_stat, w_p = sp_stats.wilcoxon(diff, alternative="two-sided")
        except ValueError:
            w_stat, w_p = float("nan"), float("nan")

        rows.append({
            "model": model_name,
            "reference": reference_model,
            "t_statistic": float(t_stat),
            "t_pvalue": float(t_p),
            "wilcoxon_statistic": float(w_stat),
            "wilcoxon_pvalue": float(w_p),
            "n_folds": n,
        })

    return pd.DataFrame(rows)


# ── NestedCrossValidator ──────────────────────────────────────────────────────

class NestedCrossValidator:
    """
    Nested stratified cross-validation: outer K-fold + inner GridSearchCV.

    Parameters
    ----------
    expression_df : pd.DataFrame
        Samples × genes expression matrix (after probe collapse, before MAD
        filtering or scaling). Index = sample IDs.
    labels : pd.Series
        Binary (0/1) labels, index aligned with expression_df.
    bio_map : dict
        Output of build_full_biological_map(); used to construct per-fold
        BINN network graphs.
    """

    def __init__(
        self,
        expression_df: pd.DataFrame,
        labels: pd.Series,
        bio_map: dict,
    ) -> None:
        self.expression_df = expression_df
        self.labels = labels
        self.bio_map = bio_map

        self.outer_cv = StratifiedKFold(
            n_splits=config.OUTER_FOLDS,
            shuffle=True,
            random_state=config.RANDOM_SEED,
        )

        # Accumulated across folds
        self._all_fold_metrics: list[dict] = []
        self._best_hyperparameters: dict[str, list] = {m: [] for m in _MODEL_NAMES}
        self._training_histories: list[dict] = []
        self._fold_network_info: list[dict | None] = [None] * config.OUTER_FOLDS

    # ── Main entry point ──────────────────────────────────────────────────────

    def run(self) -> dict:
        """
        Execute all outer folds and return the full results dict.

        Returns
        -------
        dict with keys:
            all_fold_metrics       — flat list of per-(model, fold) metric dicts
            summary_table          — pd.DataFrame (model × metric mean±std)
            statistical_tests      — pd.DataFrame (pairwise vs BINN)
            best_hyperparameters   — dict model → list[dict] per fold
            training_histories     — list[dict] BINN training history per fold
            fold_network_info      — list[dict] fold BINN graph/mask metadata
        """
        X = self.expression_df.values.astype(np.float32)
        y = self.labels.values.astype(int)
        gene_names = self.expression_df.columns.tolist()

        for fold_idx, (train_idx, test_idx) in enumerate(
            self.outer_cv.split(X, y)
        ):
            log.info(f"━━━ Outer fold {fold_idx + 1}/{config.OUTER_FOLDS} ━━━")
            self._run_fold(fold_idx, X, y, gene_names, train_idx, test_idx)

        summary = aggregate_results(self._all_fold_metrics)
        stat_tests = run_statistical_tests(self._all_fold_metrics)

        return {
            "all_fold_metrics": self._all_fold_metrics,
            "summary_table": summary,
            "statistical_tests": stat_tests,
            "best_hyperparameters": self._best_hyperparameters,
            "training_histories": self._training_histories,
            "fold_network_info": self._fold_network_info,
        }

    # ── Single fold ───────────────────────────────────────────────────────────

    def _run_fold(
        self,
        fold_idx: int,
        X: np.ndarray,
        y: np.ndarray,
        gene_names: list[str],
        train_idx: np.ndarray,
        test_idx: np.ndarray,
    ) -> None:
        X_train_raw, X_test_raw = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # ── 1. MAD filtering (on train only) ─────────────────────────────────
        train_df = pd.DataFrame(X_train_raw, columns=gene_names)
        mad = compute_mad(train_df)
        train_filtered = apply_mad_filter(train_df, mad, config.MAD_PERCENTILE)
        kept_genes = train_filtered.columns.tolist()

        test_df = pd.DataFrame(X_test_raw, columns=gene_names)
        test_filtered = test_df[kept_genes]

        X_train_filt = train_filtered.values.astype(np.float32)
        X_test_filt = test_filtered.values.astype(np.float32)

        # ── 2. Standardisation (on train only) ───────────────────────────────
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train_filt).astype(np.float32)
        X_test_sc = scaler.transform(X_test_filt).astype(np.float32)

        # ── 3. BINN ──────────────────────────────────────────────────────────
        self._run_binn_fold(
            fold_idx, kept_genes,
            X_train_sc, y_train,
            X_test_sc, y_test,
        )

        # ── 4. Baselines (use all MAD-filtered genes, scaled) ─────────────
        for model_name in _MODEL_NAMES:
            self._run_baseline_fold(
                fold_idx, model_name,
                X_train_sc, y_train,
                X_test_sc, y_test,
            )

    # ── BINN fold ─────────────────────────────────────────────────────────────

    def _run_binn_fold(
        self,
        fold_idx: int,
        kept_genes: list[str],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> None:
        """Build fold-specific BINN network, train it, evaluate on test set."""
        log.info(f"  [fold {fold_idx}] Building BINN network ({len(kept_genes)} genes)...")

        # Build fold-specific network from MAD-filtered gene list
        net = build_fold_network(kept_genes, self.bio_map)
        conn_mats = net["connectivity_matrices"]
        layer_sizes = net["layer_sizes"]

        if not conn_mats:
            log.warning(f"  [fold {fold_idx}] No connectivity matrices; skipping BINN.")
            return

        # Filter X to genes actually present in the network input layer
        binn_input_genes = net["layer_node_names"][0]
        gene_to_idx = {g: i for i, g in enumerate(kept_genes)}
        binn_gene_idx = [gene_to_idx[g] for g in binn_input_genes if g in gene_to_idx]

        if not binn_gene_idx:
            log.warning(f"  [fold {fold_idx}] No overlap between BINN genes and filtered genes; skipping.")
            return

        X_train_binn = X_train[:, binn_gene_idx]
        X_test_binn = X_test[:, binn_gene_idx]

        # Stratified split for inner BINN validation
        inner_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_SEED + fold_idx)
        binn_inner_splits = list(inner_skf.split(X_train_binn, y_train))
        binn_tr_idx, binn_val_idx = binn_inner_splits[0]

        X_binn_tr = X_train_binn[binn_tr_idx]
        y_binn_tr = y_train[binn_tr_idx]
        X_binn_val = X_train_binn[binn_val_idx]
        y_binn_val = y_train[binn_val_idx]

        # Instantiate & train BINN
        model = BINN(
            connectivity_matrices=conn_mats,
            layer_sizes=layer_sizes,
            dropout_rate=config.DROPOUT_RATE,
        )
        trainer = BINNTrainer(model, device=_select_binn_device(conn_mats))

        log.info(f"  [fold {fold_idx}] Training BINN...")
        history = trainer.fit(X_binn_tr, y_binn_tr, X_binn_val, y_binn_val)
        history["fold"] = fold_idx
        self._training_histories.append(history)

        # Save fold network metadata for downstream SHAP analysis.
        self._fold_network_info[fold_idx] = (
            {
                "fold": fold_idx,
                "connectivity_matrices": [c.detach().cpu() for c in conn_mats],
                "layer_sizes": list(layer_sizes),
                "layer_node_names": net["layer_node_names"],
                "node_metadata": net.get("node_metadata", {}),
            }
        )

        # Save model state dict
        model_path = os.path.join(config.MODEL_DIR, f"binn_fold{fold_idx}.pt")
        torch.save(model.state_dict(), model_path)
        log.info(f"  [fold {fold_idx}] BINN saved → {model_path}")

        # Evaluate
        y_prob = trainer.predict_proba(X_test_binn)
        y_pred = (y_prob >= 0.5).astype(int)
        metrics = compute_fold_metrics(y_test, y_prob, y_pred, "binn", fold_idx)
        self._all_fold_metrics.append(metrics)
        log.info(
            f"  [fold {fold_idx}] BINN — AUROC={metrics['auroc']:.4f}  "
            f"F1={metrics['f1']:.4f}  ACC={metrics['accuracy']:.4f}"
        )

    # ── Baseline fold ─────────────────────────────────────────────────────────

    def _run_baseline_fold(
        self,
        fold_idx: int,
        model_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> None:
        """Tune (inner GridSearchCV) and evaluate a single baseline model."""
        log.info(f"  [fold {fold_idx}] Fitting baseline: {model_name}")
        wrapper = BaselineWrapper(model_name=model_name, random_state=config.RANDOM_SEED)
        wrapper.fit(X_train, y_train, inner_cv=config.INNER_FOLDS)

        self._best_hyperparameters[model_name].append(
            {"fold": fold_idx, "params": wrapper.best_params_}
        )

        y_prob = wrapper.predict_proba(X_test)
        y_pred = wrapper.predict(X_test)
        metrics = compute_fold_metrics(y_test, y_prob, y_pred, model_name, fold_idx)
        self._all_fold_metrics.append(metrics)
        log.info(
            f"  [fold {fold_idx}] {model_name} — AUROC={metrics['auroc']:.4f}  "
            f"F1={metrics['f1']:.4f}  ACC={metrics['accuracy']:.4f}"
        )


# ── Saving utilities ──────────────────────────────────────────────────────────

def _make_json_serialisable(obj: Any) -> Any:
    """Recursively convert numpy scalars and arrays to Python native types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _make_json_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_json_serialisable(v) for v in obj]
    return obj


def save_results(results: dict, output_dir: str = config.METRIC_DIR) -> None:
    """
    Persist all cross-validation results to output_dir.

    Files written
    -------------
    nested_cv_results.json      — full fold-by-fold metrics (all models)
    summary_table.csv           — mean ± std per model per metric
    statistical_tests.csv       — pairwise test results vs BINN
    best_hyperparameters.json   — best params per baseline per fold
    training_histories.json     — BINN training curves per fold
    fold_network_info.pkl       — fold network masks/metadata for SHAP
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Full fold metrics
    metrics_path = os.path.join(output_dir, "nested_cv_results.json")
    with open(metrics_path, "w") as fh:
        json.dump(_make_json_serialisable(results["all_fold_metrics"]), fh, indent=2)
    log.info(f"Saved full metrics → {metrics_path}")

    # 2. Summary table
    summary_path = os.path.join(output_dir, "summary_table.csv")
    results["summary_table"].to_csv(summary_path)
    log.info(f"Saved summary table → {summary_path}")

    # 3. Statistical tests
    stat_path = os.path.join(output_dir, "statistical_tests.csv")
    results["statistical_tests"].to_csv(stat_path, index=False)
    log.info(f"Saved statistical tests → {stat_path}")

    # 4. Best hyperparameters
    hp_path = os.path.join(output_dir, "best_hyperparameters.json")
    with open(hp_path, "w") as fh:
        json.dump(_make_json_serialisable(results["best_hyperparameters"]), fh, indent=2)
    log.info(f"Saved best hyperparameters → {hp_path}")

    # 5. Training histories
    hist_path = os.path.join(output_dir, "training_histories.json")
    with open(hist_path, "w") as fh:
        json.dump(_make_json_serialisable(results["training_histories"]), fh, indent=2)
    log.info(f"Saved training histories → {hist_path}")

    # 6. Fold network info for SHAP reconstruction
    if "fold_network_info" in results:
        with open(config.FOLD_NETWORK_INFO_FILE, "wb") as fh:
            pickle.dump(results["fold_network_info"], fh)
        log.info(f"Saved fold network info → {config.FOLD_NETWORK_INFO_FILE}")


# ── Orchestrator ──────────────────────────────────────────────────────────────

def run_nested_cv() -> None:
    """
    End-to-end nested cross-validation pipeline.

    1. Load preprocessed expression data and labels.
    2. Load biological mapping.
    3. Run nested CV (BINN + 4 baselines).
    4. Aggregate, test, and save all results.
    5. Print final summary.
    """
    # ── Load preprocessed data ────────────────────────────────────────────────
    expr_df, labels = load_preprocessed_data()
    log.info(
        "Device selection: priority=%s | primary=%s | xgboost=%s",
        " > ".join(d.upper() for d in config.DEVICE_PRIORITY),
        config.DEVICE_IDENTIFIER,
        config.XGBOOST_DEVICE,
    )
    log.info(
        f"Dataset: {expr_df.shape[0]} samples × {expr_df.shape[1]} genes  "
        f"| HPV+: {labels.sum()}  HPV−: {(labels == 0).sum()}"
    )

    # ── Load biological map ────────────────────────────────────────────────────
    bio_map_path = config.BIO_MAP_FILE
    if not os.path.exists(bio_map_path):
        raise FileNotFoundError(
            f"Biological map not found at {bio_map_path}.\n"
            "Run biological_mapping.py first:  python src/biological_mapping.py"
        )
    log.info(f"Loading biological map from {bio_map_path}")
    with open(bio_map_path, "rb") as fh:
        bio_map = pickle.load(fh)

    # ── Run nested CV ─────────────────────────────────────────────────────────
    validator = NestedCrossValidator(
        expression_df=expr_df,
        labels=labels,
        bio_map=bio_map,
    )
    results = validator.run()

    # ── Save ──────────────────────────────────────────────────────────────────
    save_results(results)

    # ── Print summary ─────────────────────────────────────────────────────────
    print()
    print("═" * 70)
    print("  Nested CV Summary  (mean ± std across folds)")
    print("═" * 70)
    print(results["summary_table"].to_string())
    print()
    print("Pairwise statistical tests vs BINN (AUROC):")
    print(results["statistical_tests"].to_string(index=False))
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_nested_cv()
