"""
SHAP-Based Biological Interpretation for BINN.

Computes gene-level and pathway-level attribution scores, maps them to PPI
edges, reconstructs the HPV→HNSCC proliferation cascade through the BINN
layer hierarchy, and aggregates results across CV folds.

Attribution methods (in order of preference):
  1. shap.DeepExplainer   — exact Shapley values via PyTorch hooks (CPU)
  2. shap.GradientExplainer — gradient-based Shapley approximation
  3. Gradient × Input     — fast fallback if shap is unavailable / fails

Layer-wise attributions always use Gradient × Activation (hook-based).
"""
from __future__ import annotations

import json
import logging
import os
import pickle
import sys
import warnings
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.binn_model import BINN

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ── Optional SHAP import ───────────────────────────────────────────────────────

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    log.warning(
        "shap not installed — falling back to Gradient×Input attribution. "
        "Install with:  uv add shap"
    )


# ── Gradient × Input (fallback) ───────────────────────────────────────────────

def _gradient_times_input(
    model: BINN,
    X: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """
    Gradient × Input attribution at the input layer.

    Returns
    -------
    (n_samples, n_genes) float32 array
    """
    model.eval()
    X_t = torch.from_numpy(X.astype(np.float32)).to(device)
    X_t.requires_grad_(True)
    logits = model(X_t)
    logits.sum().backward()
    attr = (X_t.grad.detach() * X_t.detach()).cpu().numpy().astype(np.float32)
    return attr


# ── Input-layer SHAP ──────────────────────────────────────────────────────────

def _deep_explainer_blockers(model: nn.Module) -> list[str]:
    """
    Return unsupported module names for DeepExplainer in this architecture.

    SHAP's PyTorch DeepExplainer does not robustly support ``nn.Mish`` in
    additivity checks, so we route to GradientExplainer when present.
    """
    blockers: set[str] = set()
    for mod in model.modules():
        if isinstance(mod, nn.Mish):
            blockers.add("Mish")
    return sorted(blockers)


def compute_shap_values(
    model: BINN,
    X_test: np.ndarray,
    X_train: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """
    Compute SHAP values at the gene (input) layer.

    Uses shap.DeepExplainer on CPU (avoids MPS hook instability) when
    architecture support is available. For known unsupported modules (e.g.
    ``nn.Mish``), it skips DeepExplainer and uses GradientExplainer directly.
    Falls back to Gradient×Input if SHAP methods fail.

    Parameters
    ----------
    model   : trained BINN — temporarily moved to CPU, then restored
    X_test  : (n_test, n_genes) float32
    X_train : (n_train, n_genes) float32 — source of background samples
    device  : original device (model is restored after computation)

    Returns
    -------
    (n_test, n_genes) float32 SHAP values
    """
    if not SHAP_AVAILABLE:
        log.info("shap unavailable — using Gradient×Input.")
        return _gradient_times_input(model, X_test, device)

    model.eval()

    # Build background: random subsample of training data (on CPU)
    n_bg = min(100, len(X_train))
    rng = np.random.default_rng(config.RANDOM_SEED)
    bg_idx = rng.choice(len(X_train), n_bg, replace=False)
    background = torch.from_numpy(X_train[bg_idx].astype(np.float32))

    X_test_t = torch.from_numpy(X_test.astype(np.float32))

    def _parse_shap(vals) -> np.ndarray:
        if isinstance(vals, list):
            vals = vals[0]
        arr = np.array(vals, dtype=np.float32)
        # DeepExplainer / GradientExplainer may return (n, features, outputs)
        # for single-output models; squeeze the last dim.
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr.squeeze(-1)
        return arr

    # Move to CPU for SHAP; always restore to original device on exit.
    model.cpu()
    try:
        blockers = _deep_explainer_blockers(model)
        if blockers:
            log.info(
                "Skipping DeepExplainer due to unsupported module(s): %s. "
                "Using GradientExplainer.",
                ", ".join(blockers),
            )
        else:
            # 1. Try DeepExplainer
            try:
                with warnings.catch_warnings():
                    # DeepExplainer may emit unrecognized nn.Module warnings for some
                    # architectures; we treat those as non-fatal and rely on fallback.
                    warnings.filterwarnings(
                        "ignore",
                        message=r"unrecognized nn\.Module: .*",
                        category=UserWarning,
                    )
                    explainer = shap.DeepExplainer(model, background)
                    shap_vals = _parse_shap(explainer.shap_values(X_test_t))
                log.info(f"DeepExplainer SHAP: shape={shap_vals.shape}")
                return shap_vals
            except Exception as exc:
                log.info(f"DeepExplainer skipped/failed ({exc}). Trying GradientExplainer...")

        # 2. Try GradientExplainer
        try:
            explainer = shap.GradientExplainer(model, background)
            shap_vals = _parse_shap(explainer.shap_values(X_test_t))
            log.info(f"GradientExplainer SHAP: shape={shap_vals.shape}")
            return shap_vals
        except Exception as exc2:
            log.warning(f"GradientExplainer failed ({exc2}). Falling back to Gradient×Input.")

        model.to(device)
        return _gradient_times_input(model, X_test, device)
    finally:
        model.to(device)


# ── Layer-wise Gradient × Activation ─────────────────────────────────────────

def compute_layerwise_shap(
    model: BINN,
    X_test: np.ndarray,
    X_train: np.ndarray,        # kept for API consistency
    layer_node_names: list[list[str]],
    device: torch.device,
) -> dict[str, dict[str, float]]:
    """
    Gradient × Activation attributions at every BINN layer.

    For each hidden layer ℓ:
      activation_ℓ  ← forward hook output (post Mish → BN → Dropout)
      gradient_ℓ    ← backward hook grad of loss w.r.t. layer output
      attribution_ℓ = mean_samples(|gradient_ℓ × activation_ℓ|)

    Parameters
    ----------
    model            : trained BINN
    X_test           : (n_test, n_genes) float32
    X_train          : unused, kept for API symmetry with compute_shap_values
    layer_node_names : layer_node_names from build_fold_network()
    device           : device the model lives on

    Returns
    -------
    dict with keys "layer_0_genes", "layer_1_pathways", ..., "layer_L_roots"
    Each maps node_name → mean_abs_attribution (float).
    """
    model.eval()
    X_t = torch.from_numpy(X_test.astype(np.float32)).to(device)

    fwd_acts: dict[int, torch.Tensor] = {}
    bwd_grads: dict[int, torch.Tensor] = {}
    handles: list = []

    def _fwd_hook(idx: int):
        def hook(module, inp, out):
            fwd_acts[idx] = out.detach().cpu()
        return hook

    def _bwd_hook(idx: int):
        def hook(module, grad_inp, grad_out):
            if grad_out and grad_out[0] is not None:
                bwd_grads[idx] = grad_out[0].detach().cpu()
        return hook

    # Hidden layers are indexed 1..n_hidden in the attribution dict
    for i, layer in enumerate(model.hidden_layers):
        handles.append(layer.register_forward_hook(_fwd_hook(i + 1)))
        handles.append(layer.register_full_backward_hook(_bwd_hook(i + 1)))

    try:
        X_t.requires_grad_(True)
        logits = model(X_t)
        logits.sum().backward()
    except Exception as exc:
        log.warning(f"Backward pass for layer-wise SHAP failed: {exc}")
        for h in handles:
            h.remove()
        return {}

    result: dict[str, dict[str, float]] = {}

    # Layer 0: gene inputs
    if X_t.grad is not None:
        attr_0 = (X_t.grad.detach().cpu() * X_t.detach().cpu()).abs().mean(0).numpy()
    else:
        attr_0 = np.zeros(X_t.shape[1])
    names_0 = layer_node_names[0] if layer_node_names else []
    result["layer_0_genes"] = {
        name: float(attr_0[i]) for i, name in enumerate(names_0) if i < len(attr_0)
    }

    # Hidden layers
    n_hidden = len(model.hidden_layers)
    for i in range(n_hidden):
        idx = i + 1
        # Last hidden layer feeds into root pathways
        label = "roots" if i == n_hidden - 1 else "pathways"
        key = f"layer_{idx}_{label}"

        if idx in fwd_acts and idx in bwd_grads:
            acts = fwd_acts[idx]
            grads = bwd_grads[idx]
            n = min(acts.shape[0], grads.shape[0])
            attr = (grads[:n] * acts[:n]).abs().mean(0).numpy()
        else:
            attr = np.array([])

        names = layer_node_names[idx] if idx < len(layer_node_names) else []
        result[key] = {
            name: float(attr[j]) for j, name in enumerate(names) if j < len(attr)
        }

    for h in handles:
        h.remove()

    return result


# ── Pathway-level aggregation ─────────────────────────────────────────────────

def aggregate_shap_to_pathways(
    shap_values: np.ndarray,
    gene_names: list[str],
    gene_to_reactome: dict[str, set[str]],
    reactome_names: dict[str, str],
) -> pd.DataFrame:
    """
    Aggregate gene-level SHAP magnitudes to pathway-level scores.

    For each Reactome pathway p:
      mean_abs_shap = mean(|SHAP_g| for g ∈ G_p)
      max_abs_shap  = max(|SHAP_g|  for g ∈ G_p)
      sum_abs_shap  = sum(|SHAP_g|  for g ∈ G_p)

    |SHAP_g| is the mean absolute SHAP value across all test samples for gene g.

    Parameters
    ----------
    shap_values      : (n_samples, n_genes) array
    gene_names       : list of gene names matching shap_values columns
    gene_to_reactome : gene → set of pathway IDs  (from bio_map)
    reactome_names   : pathway_id → human name    (from bio_map)

    Returns
    -------
    pd.DataFrame sorted by mean_abs_shap descending.
    """
    # Mean absolute SHAP per gene across test samples
    mean_abs_shap = np.abs(shap_values).mean(axis=0)  # (n_genes,)
    gene_shap: dict[str, float] = dict(zip(gene_names, mean_abs_shap.tolist()))

    # Build pathway → [genes]
    pathway_to_genes: dict[str, list[str]] = {}
    for gene, pathways in gene_to_reactome.items():
        if gene not in gene_shap:
            continue
        for pid in pathways:
            pathway_to_genes.setdefault(pid, []).append(gene)

    rows = []
    for pid, genes in pathway_to_genes.items():
        vals = [gene_shap[g] for g in genes if g in gene_shap]
        if not vals:
            continue
        arr = np.array(vals)
        top_idx = int(np.argmax(arr))
        rows.append({
            "pathway_id": pid,
            "pathway_name": reactome_names.get(pid, pid),
            "mean_abs_shap": float(arr.mean()),
            "max_abs_shap": float(arr.max()),
            "sum_abs_shap": float(arr.sum()),
            "n_genes": len(vals),
            "top_gene": genes[top_idx],
            "top_gene_shap": float(arr.flat[top_idx]),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


# ── PPI-level interpretation ──────────────────────────────────────────────────

def analyze_ppi_importance(
    shap_values: np.ndarray,
    gene_names: list[str],
    ppi_edges: list[tuple[str, str, int]],
    node_metadata: dict,
    gene_to_reactome: dict[str, set[str]] | None = None,
) -> pd.DataFrame:
    """
    Score PPI edges by the SHAP magnitudes of their endpoint genes.

    Parameters
    ----------
    shap_values      : (n_samples, n_genes)
    gene_names       : matching gene names for shap_values columns
    ppi_edges        : list of (gene1, gene2, combined_score)
    node_metadata    : from build_node_metadata()
    gene_to_reactome : optional, for shared-pathway annotation

    Returns
    -------
    pd.DataFrame sorted by ppi_importance_add descending.
    """
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    gene_shap: dict[str, float] = dict(zip(gene_names, mean_abs_shap.tolist()))

    gene_to_reactome = gene_to_reactome or {}

    rows = []
    for g1, g2, score in ppi_edges:
        s1 = gene_shap.get(g1)
        s2 = gene_shap.get(g2)
        if s1 is None or s2 is None:
            continue

        meta1 = node_metadata.get(g1, {})
        meta2 = node_metadata.get(g2, {})

        shared = sorted(
            gene_to_reactome.get(g1, set()) & gene_to_reactome.get(g2, set())
        )

        rows.append({
            "gene_1": g1,
            "protein_1": meta1.get("name", g1),
            "gene_2": g2,
            "protein_2": meta2.get("name", g2),
            "string_score": score,
            "shap_gene_1": float(s1),
            "shap_gene_2": float(s2),
            "ppi_importance_add": float(s1 + s2),
            "ppi_importance_mult": float(s1 * s2),
            "n_shared_pathways": len(shared),
            "shared_pathways": shared,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("ppi_importance_add", ascending=False).reset_index(drop=True)


# ── Level 1: Protein-to-Protein SHAP ─────────────────────────────────────────

def compute_ppi_shap(
    shap_values: np.ndarray,
    gene_names: list[str],
    ppi_edge_to_pathways: dict,
    bio_map: dict,
) -> pd.DataFrame:
    """
    Score each STRING PPI edge by the SHAP values of its two endpoint proteins.

    Parameters
    ----------
    shap_values         : (n_samples, n_genes) signed SHAP array
    gene_names          : gene symbols matching shap_values columns
    ppi_edge_to_pathways: from ppi_overlay["ppi_edge_to_pathways"]
                          keys = (gene_a, gene_b), values = dict with
                          "score", "shared_pathways", "cross_pathways"
    bio_map             : full biological map (for protein name lookup)

    Returns
    -------
    pd.DataFrame sorted by multiplicative descending.
    """
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    mean_signed_shap = shap_values.mean(axis=0)
    gene_abs: dict[str, float] = dict(zip(gene_names, mean_abs_shap.tolist()))
    gene_signed: dict[str, float] = dict(zip(gene_names, mean_signed_shap.tolist()))

    node_meta = bio_map.get("node_metadata", {})
    reactome_names = bio_map.get("reactome_names", {})

    rows = []
    for (ga, gb), info in ppi_edge_to_pathways.items():
        sa = gene_abs.get(ga)
        sb = gene_abs.get(gb)
        if sa is None or sb is None:
            continue

        shared = info.get("shared_pathways", [])
        cross = info.get("cross_pathways", [])
        pathway_ctx = sorted(set(shared + [p for pair in cross for p in pair]))

        rows.append({
            "gene_a": ga,
            "gene_b": gb,
            "protein_a": node_meta.get(ga, {}).get("name", ga),
            "protein_b": node_meta.get(gb, {}).get("name", gb),
            "string_score": info.get("score", 0),
            "shap_a": float(sa),
            "shap_b": float(sb),
            "additive": float(sa + sb),
            "multiplicative": float(sa * sb),
            "directional": float(gene_signed.get(ga, 0.0) + gene_signed.get(gb, 0.0)),
            "pathway_context": [reactome_names.get(p, p) for p in pathway_ctx],
            "is_cross_pathway": len(cross) > 0,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("multiplicative", ascending=False).reset_index(drop=True)


# ── Level 2a: Pathway-level SHAP ─────────────────────────────────────────────

def compute_pathway_shap(
    shap_values: np.ndarray,
    gene_names: list[str],
    gene_to_reactome: dict[str, set[str]],
    reactome_names: dict[str, str],
    pathway_ppi_subgraphs: dict | None = None,
    hierarchy_depth: dict[str, int] | None = None,
) -> pd.DataFrame:
    """
    Compute per-pathway SHAP statistics with extra metadata vs the basic
    aggregate_shap_to_pathways().

    Parameters
    ----------
    shap_values            : (n_samples, n_genes)
    gene_names             : matching gene names
    gene_to_reactome       : gene → set[pathway_id]
    reactome_names         : pathway_id → display name
    pathway_ppi_subgraphs  : from ppi_overlay, optional
    hierarchy_depth        : {pathway_id: depth_int}, optional

    Returns
    -------
    pd.DataFrame sorted by mean_abs_shap descending.
    """
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    mean_signed_shap = shap_values.mean(axis=0)
    median_abs = float(np.median(mean_abs_shap))

    gene_abs: dict[str, float] = dict(zip(gene_names, mean_abs_shap.tolist()))
    gene_signed: dict[str, float] = dict(zip(gene_names, mean_signed_shap.tolist()))

    pathway_ppi_subgraphs = pathway_ppi_subgraphs or {}
    hierarchy_depth = hierarchy_depth or {}

    # Build pathway → genes in our network
    pathway_to_genes: dict[str, list[str]] = {}
    for gene, pathways in gene_to_reactome.items():
        if gene not in gene_abs:
            continue
        for pid in pathways:
            pathway_to_genes.setdefault(pid, []).append(gene)

    rows = []
    for pid, genes in pathway_to_genes.items():
        vals_abs = [gene_abs[g] for g in genes if g in gene_abs]
        if not vals_abs:
            continue
        arr = np.array(vals_abs)
        top_idx = int(np.argmax(arr))
        n_sig = int((arr > median_abs).sum())

        # PPI info for this pathway
        subgraph = pathway_ppi_subgraphs.get(pid, {})
        ppi_edges_p = subgraph.get("ppi_edges", [])
        n_ppi = len(ppi_edges_p)
        top_ppi = ("", "")
        if ppi_edges_p:
            best = max(ppi_edges_p, key=lambda e: gene_abs.get(e[0], 0) + gene_abs.get(e[1], 0))
            top_ppi = (best[0], best[1])

        directional = sum(gene_signed.get(g, 0.0) for g in genes)

        rows.append({
            "pathway_id": pid,
            "pathway_name": reactome_names.get(pid, pid),
            "hierarchy_level": hierarchy_depth.get(pid, -1),
            "mean_abs_shap": float(arr.mean()),
            "sum_abs_shap": float(arr.sum()),
            "max_abs_shap": float(arr.max()),
            "n_genes": len(vals_abs),
            "n_significant_genes": n_sig,
            "n_ppi_edges": n_ppi,
            "top_gene": genes[top_idx],
            "top_ppi_a": top_ppi[0],
            "top_ppi_b": top_ppi[1],
            "directional": float(directional),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


# ── Level 2b: Pathway-to-Pathway SHAP via Gradient × Activation ──────────────

def compute_pathway_to_pathway_shap(
    model: "BINN",
    X_test: np.ndarray,
    layer_node_names: list[list[str]],
    reactome_hierarchy: nx.DiGraph,
    reactome_names: dict[str, str],
    ppi_overlay: dict | None = None,
    device: torch.device | None = None,
) -> pd.DataFrame:
    """
    Gradient × Activation attributions for each pathway→pathway edge.

    For each edge (child_pathway in layer ℓ) → (parent_pathway in layer ℓ+1):
        attribution = |grad_at_parent| * |act_at_child|, averaged over samples.

    Parameters
    ----------
    model               : trained BINN
    X_test              : (n_test, n_genes) float32
    layer_node_names    : from build_fold_network()
    reactome_hierarchy  : from bio_map["reactome_hierarchy"]
    reactome_names      : from bio_map["reactome_names"]
    ppi_overlay         : from fold_network_info["ppi_overlay"] (for cross-pathway labels)
    device              : torch device

    Returns
    -------
    pd.DataFrame with columns:
        child_pathway, child_name, parent_pathway, parent_name,
        layer, attribution, connection_type
    """
    from src.binn_model import BINN  # local import to avoid circular

    if device is None:
        device = next(model.parameters()).device

    model.eval()
    X_t = torch.from_numpy(X_test.astype(np.float32)).to(device)

    fwd_acts: dict[int, torch.Tensor] = {}
    bwd_grads: dict[int, torch.Tensor] = {}
    handles: list = []

    def _fwd(idx: int):
        def hook(module, inp, out):
            fwd_acts[idx] = out.detach().cpu()
        return hook

    def _bwd(idx: int):
        def hook(module, grad_inp, grad_out):
            if grad_out and grad_out[0] is not None:
                bwd_grads[idx] = grad_out[0].detach().cpu()
        return hook

    for i, layer in enumerate(model.hidden_layers):
        handles.append(layer.register_forward_hook(_fwd(i)))
        handles.append(layer.register_full_backward_hook(_bwd(i)))

    try:
        X_t.requires_grad_(True)
        logits = model(X_t)
        logits.sum().backward()
    except Exception as exc:
        log.warning(f"compute_pathway_to_pathway_shap: backward failed ({exc})")
        return pd.DataFrame()
    finally:
        for h in handles:
            h.remove()

    # Build cross-pathway PPI set for connection_type labelling
    cross_ppi_pairs: set[tuple[str, str]] = set()
    if ppi_overlay:
        for entry in ppi_overlay.get("cross_pathway_ppi_edges", []):
            cross_ppi_pairs.add((entry["pathway_a"], entry["pathway_b"]))
            cross_ppi_pairs.add((entry["pathway_b"], entry["pathway_a"]))

    rows = []
    n_hidden = len(model.hidden_layers)
    # layer index i in hidden_layers corresponds to layer_node_names[i+1]
    # attribution for edge child(layer l) → parent(layer l+1):
    #   acts at layer l (child) = fwd_acts[l-1] (hidden layer index l-1)
    #   grads at layer l+1 (parent) = bwd_grads[l]
    for l in range(1, len(layer_node_names) - 1):
        child_names = layer_node_names[l]
        parent_names = layer_node_names[l + 1] if l + 1 < len(layer_node_names) else []

        act_key = l - 1   # hidden layer index
        grad_key = l      # hidden layer index for parent

        if act_key not in fwd_acts or grad_key not in bwd_grads:
            continue

        acts = fwd_acts[act_key]   # (n, n_child)
        grads = bwd_grads[grad_key]  # (n, n_parent)
        n = min(acts.shape[0], grads.shape[0])

        # Mean |act| per child node
        mean_act = acts[:n].abs().mean(0).numpy()   # (n_child,)
        # Mean |grad| per parent node
        mean_grad = grads[:n].abs().mean(0).numpy()  # (n_parent,)

        # For each Reactome hierarchy edge between child and parent layers
        for ci, cnode in enumerate(child_names):
            if ci >= len(mean_act):
                continue
            for pi, pnode in enumerate(parent_names):
                if pi >= len(mean_grad):
                    continue
                # Only record edges that exist in the hierarchy
                if not reactome_hierarchy.has_node(cnode) or not reactome_hierarchy.has_node(pnode):
                    continue
                # Reactome hierarchy: parent→child (general→specific).
                # In our flow graph child→parent. Check either direction.
                if not (reactome_hierarchy.has_edge(pnode, cnode) or
                        reactome_hierarchy.has_edge(cnode, pnode)):
                    continue

                attr = float(mean_act[ci] * mean_grad[pi])
                pair = (cnode, pnode)
                ctype = "ppi_crosstalk" if pair in cross_ppi_pairs else "hierarchy"

                rows.append({
                    "child_pathway": cnode,
                    "child_name": reactome_names.get(cnode, cnode),
                    "parent_pathway": pnode,
                    "parent_name": reactome_names.get(pnode, pnode),
                    "layer": l,
                    "attribution": attr,
                    "connection_type": ctype,
                })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("attribution", ascending=False).reset_index(drop=True)


# ── Cross-fold dual SHAP aggregation ─────────────────────────────────────────

def aggregate_dual_shap(
    fold_results: list[dict],
    output_dir: str = config.SHAP_DIR,
    n_folds_total: int = 5,
    top_k: int = 20,
) -> dict:
    """
    Average PPI-level and pathway-level SHAP importance across folds.
    Flags high-confidence findings (top-K in >= 4/5 folds).
    Saves all aggregated tables to output_dir.

    Parameters
    ----------
    fold_results   : list of per-fold dicts, each with keys:
                     "ppi_shap_df", "pathway_shap_df", "p2p_shap_df"
    output_dir     : where to save CSV files
    n_folds_total  : used for stability threshold
    top_k          : top-K threshold for high-confidence flag

    Returns
    -------
    dict with keys: ppi_df, pathway_df, p2p_df, stability_df
    """
    os.makedirs(output_dir, exist_ok=True)

    def _edge_key(row):
        return tuple(sorted([row["gene_a"], row["gene_b"]]))

    # ── PPI aggregation ───────────────────────────────────────────────────────
    ppi_top_counts: dict[tuple, int] = {}
    ppi_all: list[pd.DataFrame] = []
    for res in fold_results:
        df = res.get("ppi_shap_df", pd.DataFrame())
        if df.empty:
            continue
        df = df.copy()
        df["edge_key"] = df.apply(_edge_key, axis=1)
        # Track top-K membership
        for ek in df.head(top_k)["edge_key"]:
            ppi_top_counts[ek] = ppi_top_counts.get(ek, 0) + 1
        ppi_all.append(df)

    ppi_df = pd.DataFrame()
    if ppi_all:
        combined = pd.concat(ppi_all, ignore_index=True)
        agg = (combined.groupby("edge_key")
               .agg(
                   gene_a=("gene_a", "first"),
                   gene_b=("gene_b", "first"),
                   protein_a=("protein_a", "first"),
                   protein_b=("protein_b", "first"),
                   string_score=("string_score", "first"),
                   shap_a=("shap_a", "mean"),
                   shap_b=("shap_b", "mean"),
                   additive=("additive", "mean"),
                   multiplicative=("multiplicative", "mean"),
                   directional=("directional", "mean"),
                   is_cross_pathway=("is_cross_pathway", "first"),
               )
               .reset_index())
        agg["top_k_folds"] = agg["edge_key"].map(lambda k: ppi_top_counts.get(k, 0))
        agg["high_confidence"] = agg["top_k_folds"] >= max(1, n_folds_total - 1)
        ppi_df = agg.sort_values("multiplicative", ascending=False).reset_index(drop=True)
        ppi_df.to_csv(os.path.join(output_dir, "ppi_importance.csv"), index=False)
        log.info(f"Saved ppi_importance.csv ({len(ppi_df)} edges)")

    # ── Pathway aggregation ───────────────────────────────────────────────────
    pw_top_counts: dict[str, int] = {}
    pw_all: list[pd.DataFrame] = []
    for res in fold_results:
        df = res.get("pathway_shap_df", pd.DataFrame())
        if df.empty:
            continue
        for pid in df.head(top_k)["pathway_id"]:
            pw_top_counts[pid] = pw_top_counts.get(pid, 0) + 1
        pw_all.append(df)

    pathway_df = pd.DataFrame()
    if pw_all:
        combined = pd.concat(pw_all, ignore_index=True)
        agg = (combined.groupby("pathway_id")
               .agg(
                   pathway_name=("pathway_name", "first"),
                   hierarchy_level=("hierarchy_level", "first"),
                   mean_abs_shap=("mean_abs_shap", "mean"),
                   sum_abs_shap=("sum_abs_shap", "mean"),
                   max_abs_shap=("max_abs_shap", "mean"),
                   n_genes=("n_genes", "first"),
                   n_significant_genes=("n_significant_genes", "mean"),
                   n_ppi_edges=("n_ppi_edges", "first"),
                   top_gene=("top_gene", "first"),
                   directional=("directional", "mean"),
               )
               .reset_index())
        agg["top_k_folds"] = agg["pathway_id"].map(lambda p: pw_top_counts.get(p, 0))
        agg["high_confidence"] = agg["top_k_folds"] >= max(1, n_folds_total - 1)
        pathway_df = agg.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
        pathway_df.to_csv(os.path.join(output_dir, "pathway_importance.csv"), index=False)
        log.info(f"Saved pathway_importance.csv ({len(pathway_df)} pathways)")

    # ── Pathway-to-pathway aggregation ───────────────────────────────────────
    p2p_all: list[pd.DataFrame] = []
    for res in fold_results:
        df = res.get("p2p_shap_df", pd.DataFrame())
        if not df.empty:
            p2p_all.append(df)

    p2p_df = pd.DataFrame()
    if p2p_all:
        combined = pd.concat(p2p_all, ignore_index=True)
        agg = (combined.groupby(["child_pathway", "parent_pathway"])
               .agg(
                   child_name=("child_name", "first"),
                   parent_name=("parent_name", "first"),
                   layer=("layer", "first"),
                   attribution=("attribution", "mean"),
                   connection_type=("connection_type", "first"),
               )
               .reset_index())
        p2p_df = agg.sort_values("attribution", ascending=False).reset_index(drop=True)
        p2p_df.to_csv(os.path.join(output_dir, "pathway_to_pathway_importance.csv"), index=False)
        log.info(f"Saved pathway_to_pathway_importance.csv ({len(p2p_df)} edges)")

    # ── Cross-fold stability ──────────────────────────────────────────────────
    stab_rows = []
    for ek, cnt in ppi_top_counts.items():
        stab_rows.append({"type": "ppi", "key": str(ek), "top_k_folds": cnt,
                          "high_confidence": cnt >= max(1, n_folds_total - 1)})
    for pid, cnt in pw_top_counts.items():
        stab_rows.append({"type": "pathway", "key": pid, "top_k_folds": cnt,
                          "high_confidence": cnt >= max(1, n_folds_total - 1)})
    stability_df = pd.DataFrame(stab_rows)
    if not stability_df.empty and "top_k_folds" in stability_df.columns:
        stability_df = stability_df.sort_values("top_k_folds", ascending=False)
    else:
        stability_df = pd.DataFrame(
            columns=["type", "key", "top_k_folds", "high_confidence"]
        )
    stability_df.to_csv(os.path.join(output_dir, "cross_fold_stability.csv"), index=False)
    log.info(f"Saved cross_fold_stability.csv ({len(stability_df)} entries)")

    return {
        "ppi_df": ppi_df,
        "pathway_df": pathway_df,
        "p2p_df": p2p_df,
        "stability_df": stability_df,
    }


# ── HPV cascade reconstruction ────────────────────────────────────────────────

def reconstruct_hpv_cascade(
    layerwise_attributions: dict[str, dict[str, float]],
    pathway_hierarchy: nx.DiGraph,
    ppi_edges: list[tuple[str, str, int]],
    node_metadata: dict,
    gene_to_reactome: dict[str, set[str]] | None = None,
    reactome_names: dict[str, str] | None = None,
    top_k: int = 20,
) -> dict:
    """
    Reconstruct the biological story: genes → PPIs → leaf pathways → root pathways.

    Parameters
    ----------
    layerwise_attributions : from compute_layerwise_shap()
    pathway_hierarchy      : Reactome DiGraph (parent→child edges, general→specific)
    ppi_edges              : list of (gene1, gene2, score)
    node_metadata          : from build_node_metadata()
    gene_to_reactome       : gene → set[pathway_id]
    reactome_names         : pathway_id → display name
    top_k                  : number of top items at each level

    Returns
    -------
    dict with keys: top_genes, top_ppis, top_leaf_pathways,
                    top_intermediate_pathways, top_root_pathways,
                    cascade_edges, cascade_graph
    """
    gene_to_reactome = gene_to_reactome or {}
    reactome_names = reactome_names or {}

    # ── Top genes ────────────────────────────────────────────────────────────
    gene_attrs = layerwise_attributions.get("layer_0_genes", {})
    top_genes_sorted = sorted(gene_attrs.items(), key=lambda x: x[1], reverse=True)[:top_k]
    top_gene_set = {g for g, _ in top_genes_sorted}

    top_genes = []
    for gene, shap_val in top_genes_sorted:
        meta = node_metadata.get(gene, {})
        top_genes.append({
            "gene": gene,
            "shap": float(shap_val),
            "protein": meta.get("name", gene),
            "uniprot": (meta.get("uniprot_ids") or [None])[0],
        })

    # ── Top PPIs among top genes ──────────────────────────────────────────────
    ppi_in_top = [
        (g1, g2, s) for g1, g2, s in ppi_edges
        if g1 in top_gene_set and g2 in top_gene_set
    ]
    ppi_in_top.sort(key=lambda e: gene_attrs.get(e[0], 0) + gene_attrs.get(e[1], 0), reverse=True)

    top_ppis = []
    for g1, g2, score in ppi_in_top[:top_k]:
        shared = sorted(
            gene_to_reactome.get(g1, set()) & gene_to_reactome.get(g2, set())
        )[:5]
        top_ppis.append({
            "gene1": g1,
            "gene2": g2,
            "string_score": score,
            "importance": float(gene_attrs.get(g1, 0) + gene_attrs.get(g2, 0)),
            "shared_pathways": [reactome_names.get(p, p) for p in shared],
        })

    # ── Gather pathway attributions across all non-gene layers ────────────────
    all_pathway_attrs: dict[str, tuple[float, int]] = {}  # pid → (attribution, layer)
    for key, attrs in layerwise_attributions.items():
        if key == "layer_0_genes":
            continue
        parts = key.split("_")
        try:
            lyr = int(parts[1])
        except (IndexError, ValueError):
            lyr = -1
        for pid, attr in attrs.items():
            if pid not in all_pathway_attrs or attr > all_pathway_attrs[pid][0]:
                all_pathway_attrs[pid] = (attr, lyr)

    # ── Top leaf pathways (layer 1) ───────────────────────────────────────────
    layer1_key = "layer_1_pathways"
    layer1_attrs = layerwise_attributions.get(layer1_key, {})
    top_leaf = sorted(layer1_attrs.items(), key=lambda x: x[1], reverse=True)[:top_k]
    top_leaf_pathways = [
        {"id": pid, "name": reactome_names.get(pid, pid), "attribution": float(a)}
        for pid, a in top_leaf
    ]

    # ── Trace leaf pathways upward to find top intermediate + root pathways ───
    # In the Reactome hierarchy edges go parent→child (general→specific).
    # To find a pathway's ancestors (more general), use reverse traversal.
    ancestors_by_score: dict[str, float] = {}
    for pid, attr in top_leaf[:top_k]:
        try:
            ancs = nx.ancestors(pathway_hierarchy, pid)
        except nx.NetworkXError:
            ancs = set()
        for anc in ancs:
            ancestors_by_score[anc] = ancestors_by_score.get(anc, 0) + attr

    # Separate into intermediate and root (root = no parents in hierarchy)
    root_nodes = {n for n in pathway_hierarchy if pathway_hierarchy.in_degree(n) == 0}
    top_roots = sorted(
        [(p, s) for p, s in ancestors_by_score.items() if p in root_nodes],
        key=lambda x: x[1], reverse=True
    )[:top_k]
    top_intermediate = sorted(
        [(p, s) for p, s in ancestors_by_score.items() if p not in root_nodes],
        key=lambda x: x[1], reverse=True
    )[:top_k]

    top_root_pathways = [
        {"id": pid, "name": reactome_names.get(pid, pid), "attribution": float(a)}
        for pid, a in top_roots
    ]
    top_intermediate_pathways = [
        {"id": pid, "name": reactome_names.get(pid, pid),
         "attribution": float(a), "layer": all_pathway_attrs.get(pid, (0, -1))[1]}
        for pid, a in top_intermediate
    ]

    # ── Build cascade subgraph ────────────────────────────────────────────────
    cascade_graph = nx.DiGraph()

    # Gene → leaf-pathway edges
    for gene in top_gene_set:
        for pid in gene_to_reactome.get(gene, set()):
            if any(p == pid for p, _ in top_leaf):
                cascade_graph.add_edge(gene, pid,
                    weight=float(gene_attrs.get(gene, 0)))

    # Leaf → ancestor edges (in Reactome hierarchy, child → parent reversed)
    top_leaf_set = {p for p, _ in top_leaf}
    for pid in top_leaf_set:
        try:
            for parent in pathway_hierarchy.predecessors(pid):  # parent in general→specific DAG
                if parent in ancestors_by_score:
                    cascade_graph.add_edge(
                        pid, parent,
                        weight=float(ancestors_by_score.get(parent, 0))
                    )
        except nx.NetworkXError:
            pass

    cascade_edges = [
        (u, v, float(d.get("weight", 0)))
        for u, v, d in cascade_graph.edges(data=True)
    ]

    return {
        "top_genes": top_genes,
        "top_ppis": top_ppis,
        "top_leaf_pathways": top_leaf_pathways,
        "top_intermediate_pathways": top_intermediate_pathways,
        "top_root_pathways": top_root_pathways,
        "cascade_edges": cascade_edges,
        "cascade_graph": cascade_graph,
    }


# ── Cross-fold aggregation ────────────────────────────────────────────────────

def aggregate_shap_across_folds(
    fold_shap_results: list[dict],
) -> dict:
    """
    Combine SHAP results across all outer CV folds.

    Each element of fold_shap_results must contain:
      "gene_names"   : list[str]
      "shap_values"  : (n_test, n_genes) array  (signed)
      "pathway_df"   : pd.DataFrame from aggregate_shap_to_pathways()
      "ppi_df"       : pd.DataFrame from analyze_ppi_importance()

    Returns
    -------
    dict with keys:
      gene_mean_abs_shap   : {gene: float} — averaged |SHAP| across folds
      gene_shap_std        : {gene: float} — std of |SHAP| across folds
      gene_rank_stability  : pd.DataFrame — genes with top-20 presence count
      pathway_mean_shap    : pd.DataFrame — mean pathway score across folds
      ppi_mean_importance  : pd.DataFrame — mean PPI importance across folds
    """
    # Collect per-gene absolute SHAP across folds (handle different gene sets)
    gene_fold_shap: dict[str, list[float]] = {}
    for res in fold_shap_results:
        names = res.get("gene_names", [])
        vals = res.get("shap_values", np.array([]))
        if len(vals) == 0 or len(names) == 0:
            continue
        mean_abs = np.abs(vals).mean(axis=0)
        for gene, v in zip(names, mean_abs):
            gene_fold_shap.setdefault(gene, []).append(float(v))

    gene_mean: dict[str, float] = {g: float(np.mean(v)) for g, v in gene_fold_shap.items()}
    gene_std: dict[str, float] = {
        g: float(np.std(v, ddof=1) if len(v) > 1 else 0.0)
        for g, v in gene_fold_shap.items()
    }

    # Rank stability: how many folds does each gene appear in top-20?
    top20_counts: dict[str, int] = {}
    for res in fold_shap_results:
        names = res.get("gene_names", [])
        vals = res.get("shap_values", np.array([]))
        if len(vals) == 0:
            continue
        mean_abs = np.abs(vals).mean(axis=0)
        ranked = sorted(zip(names, mean_abs), key=lambda x: x[1], reverse=True)[:20]
        for g, _ in ranked:
            top20_counts[g] = top20_counts.get(g, 0) + 1

    rank_df = pd.DataFrame(
        [(g, c) for g, c in sorted(top20_counts.items(), key=lambda x: x[1], reverse=True)],
        columns=["gene", "top20_fold_count"],
    )
    rank_df["mean_abs_shap"] = rank_df["gene"].map(gene_mean)
    rank_df["shap_std"] = rank_df["gene"].map(gene_std)

    # Pathway-level aggregation
    pathway_rows: list[pd.DataFrame] = [
        r["pathway_df"] for r in fold_shap_results
        if "pathway_df" in r and not r["pathway_df"].empty
    ]
    pathway_mean = _mean_df_across_folds(
        pathway_rows, key_col="pathway_id",
        value_cols=["mean_abs_shap", "max_abs_shap", "sum_abs_shap"],
        extra_cols=["pathway_name", "n_genes"],
    ) if pathway_rows else pd.DataFrame()

    # PPI-level aggregation
    ppi_rows: list[pd.DataFrame] = [
        r["ppi_df"] for r in fold_shap_results
        if "ppi_df" in r and not r["ppi_df"].empty
    ]
    if ppi_rows:
        # Canonical edge key: sorted (gene1, gene2)
        for df in ppi_rows:
            df["edge_key"] = df.apply(
                lambda row: tuple(sorted([row["gene_1"], row["gene_2"]])), axis=1
            )
        ppi_mean = _mean_df_across_folds(
            ppi_rows, key_col="edge_key",
            value_cols=["ppi_importance_add", "ppi_importance_mult"],
            extra_cols=["gene_1", "gene_2"],
        )
    else:
        ppi_mean = pd.DataFrame()

    return {
        "gene_mean_abs_shap": gene_mean,
        "gene_shap_std": gene_std,
        "gene_rank_stability": rank_df,
        "pathway_mean_shap": pathway_mean,
        "ppi_mean_importance": ppi_mean,
    }


def _mean_df_across_folds(
    dfs: list[pd.DataFrame],
    key_col: str,
    value_cols: list[str],
    extra_cols: list[str],
) -> pd.DataFrame:
    """Average numeric columns across fold DataFrames, keyed by key_col."""
    combined = pd.concat(dfs, ignore_index=True)
    agg_dict = {col: "mean" for col in value_cols}
    for col in extra_cols:
        if col in combined.columns:
            agg_dict[col] = "first"
    result = combined.groupby(key_col).agg(agg_dict).reset_index()
    if value_cols:
        result = result.sort_values(value_cols[0], ascending=False)
    return result


# ── Saving ────────────────────────────────────────────────────────────────────

def _to_jsonable(obj: Any) -> Any:
    """Recursively convert numpy types to Python native for JSON."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, (set, tuple)):
        return [_to_jsonable(v) for v in obj]
    # nx.DiGraph and other non-serialisable objects → skip
    return obj if isinstance(obj, (str, int, float, bool)) else str(obj)


def save_shap_results(
    cross_fold: dict,
    fold_results: list[dict],
    output_dir: str = config.SHAP_DIR,
) -> None:
    """
    Persist SHAP results to output_dir.

    Files written
    -------------
    gene_shap_values.csv            — per-gene mean |SHAP| and std across folds
    pathway_importance_legacy.csv   — legacy pathway aggregation table
    ppi_importance_legacy.csv       — legacy PPI aggregation table
    hpv_cascade.json                — reconstructed proliferation cascade
    layerwise_attributions.json     — per-layer node attributions (all folds)
    fold_shap_stability.csv         — cross-fold gene rank stability
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Gene SHAP values
    gene_df = pd.DataFrame({
        "gene": list(cross_fold["gene_mean_abs_shap"].keys()),
        "mean_abs_shap": list(cross_fold["gene_mean_abs_shap"].values()),
        "shap_std": [cross_fold["gene_shap_std"].get(g, 0.0)
                     for g in cross_fold["gene_mean_abs_shap"]],
    }).sort_values("mean_abs_shap", ascending=False)
    gene_df.to_csv(os.path.join(output_dir, "gene_shap_values.csv"), index=False)
    log.info(f"Saved gene SHAP → {output_dir}/gene_shap_values.csv")

    # 2. Legacy pathway importance (dual-level table is written separately
    # by aggregate_dual_shap to pathway_importance.csv).
    pw_df = cross_fold.get("pathway_mean_shap", pd.DataFrame())
    if not pw_df.empty:
        pw_df.to_csv(
            os.path.join(output_dir, "pathway_importance_legacy.csv"), index=False
        )
        log.info(
            f"Saved legacy pathway importance → "
            f"{output_dir}/pathway_importance_legacy.csv"
        )

    # 3. Legacy PPI importance (dual-level table is written separately
    # by aggregate_dual_shap to ppi_importance.csv).
    ppi_df = cross_fold.get("ppi_mean_importance", pd.DataFrame())
    if not ppi_df.empty:
        ppi_df.to_csv(os.path.join(output_dir, "ppi_importance_legacy.csv"), index=False)
        log.info(f"Saved legacy PPI importance → {output_dir}/ppi_importance_legacy.csv")

    # 4. HPV cascade (from fold 0, as representative)
    cascade_key = "cascade"
    cascades = [r[cascade_key] for r in fold_results if cascade_key in r]
    if cascades:
        # Save fold 0 cascade; exclude the non-serialisable nx.DiGraph
        c = {k: v for k, v in cascades[0].items() if k != "cascade_graph"}
        with open(os.path.join(output_dir, "hpv_cascade.json"), "w") as fh:
            json.dump(_to_jsonable(c), fh, indent=2)
        log.info(f"Saved HPV cascade → {output_dir}/hpv_cascade.json")

    # 5. Layer-wise attributions
    layerwise_all = [r.get("layerwise", {}) for r in fold_results]
    with open(os.path.join(output_dir, "layerwise_attributions.json"), "w") as fh:
        json.dump(_to_jsonable(layerwise_all), fh, indent=2)
    log.info(f"Saved layer-wise attributions → {output_dir}/layerwise_attributions.json")

    # 6. Fold stability table
    stab_df = cross_fold.get("gene_rank_stability", pd.DataFrame())
    if not stab_df.empty:
        stab_df.to_csv(os.path.join(output_dir, "fold_shap_stability.csv"), index=False)
        log.info(f"Saved fold stability → {output_dir}/fold_shap_stability.csv")


# ── Orchestrator ──────────────────────────────────────────────────────────────

def run_shap_analysis(
    fold_models: list,
    fold_data: list,
    bio_map: dict,
    network_info: list,
) -> None:
    """
    End-to-end SHAP analysis across all CV folds.

    Parameters
    ----------
    fold_models  : list of BINN instances or paths to saved .pt state-dict files
    fold_data    : list of dicts with keys X_train, y_train, X_test, y_test, gene_names
    bio_map      : from build_full_biological_map()
    network_info : list of dicts from build_fold_network(), one per fold
                   (each may contain "ppi_overlay" key from Prompt 8)

    Saves all results to config.SHAP_DIR.
    """
    reactome_names: dict[str, str] = bio_map.get("reactome_names", {})
    gene_to_reactome: dict[str, set] = bio_map.get("gene_to_reactome", {})
    ppi_edges: list = bio_map.get("ppi_edges", [])
    hierarchy: nx.DiGraph = bio_map.get("reactome_hierarchy", nx.DiGraph())

    # Hierarchy depth for pathway_shap
    try:
        from src.biological_mapping import get_pathway_depth
        hierarchy_depth = get_pathway_depth(hierarchy)
    except Exception:
        hierarchy_depth = {}

    fold_results: list[dict] = []

    for fold_idx, (fold_model, data, net) in enumerate(
        zip(fold_models, fold_data, network_info)
    ):
        log.info(f"━━━ SHAP fold {fold_idx + 1}/{len(fold_models)} ━━━")

        # ── Load model ────────────────────────────────────────────────────────
        conn_mats = net["connectivity_matrices"]
        layer_sizes = net["layer_sizes"]
        layer_node_names = net["layer_node_names"]
        node_metadata = net.get("node_metadata", {})
        ppi_overlay = net.get("ppi_overlay", {})

        if isinstance(fold_model, (str, os.PathLike)):
            model = BINN(conn_mats, layer_sizes, dropout_rate=config.DROPOUT_RATE)
            state = torch.load(fold_model, map_location="cpu")
            model.load_state_dict(state)
        else:
            model = fold_model
        model.eval()

        X_train: np.ndarray = data["X_train"]
        X_test: np.ndarray = data["X_test"]
        gene_names: list[str] = data["gene_names"]   # BINN input genes for this fold
        y_test: np.ndarray = data.get("y_test", np.array([]))

        # ── Input-layer SHAP ──────────────────────────────────────────────────
        shap_vals = compute_shap_values(model, X_test, X_train, config.DEVICE)

        # ── Layer-wise attributions ───────────────────────────────────────────
        model.to(config.DEVICE)
        layerwise = compute_layerwise_shap(
            model, X_test, X_train, layer_node_names, config.DEVICE
        )

        # ── Pathway aggregation (legacy) ──────────────────────────────────────
        pathway_df = aggregate_shap_to_pathways(
            shap_vals, gene_names, gene_to_reactome, reactome_names
        )

        # ── PPI importance (legacy) ───────────────────────────────────────────
        fold_ppi_edges = [
            (g1, g2, s) for g1, g2, s in ppi_edges
            if g1 in gene_names and g2 in gene_names
        ]
        ppi_df = analyze_ppi_importance(
            shap_vals, gene_names, fold_ppi_edges,
            node_metadata, gene_to_reactome,
        )

        # ── Level 1: Protein-to-Protein SHAP (Prompt 8) ──────────────────────
        ppi_edge_to_pathways = ppi_overlay.get("ppi_edge_to_pathways", {})
        ppi_shap_df = compute_ppi_shap(
            shap_vals, gene_names, ppi_edge_to_pathways, bio_map
        )

        # ── Level 2a: Pathway SHAP with extra metadata (Prompt 8) ────────────
        pathway_ppi_subgraphs = ppi_overlay.get("pathway_ppi_subgraphs", {})
        pathway_shap_df = compute_pathway_shap(
            shap_vals, gene_names, gene_to_reactome, reactome_names,
            pathway_ppi_subgraphs=pathway_ppi_subgraphs,
            hierarchy_depth=hierarchy_depth,
        )

        # ── Level 2b: Pathway-to-pathway SHAP (Prompt 8) ─────────────────────
        p2p_shap_df = compute_pathway_to_pathway_shap(
            model, X_test, layer_node_names, hierarchy, reactome_names,
            ppi_overlay=ppi_overlay, device=config.DEVICE,
        )

        # ── Cascade reconstruction ────────────────────────────────────────────
        cascade = reconstruct_hpv_cascade(
            layerwise, hierarchy, fold_ppi_edges,
            node_metadata, gene_to_reactome, reactome_names,
        )

        fold_results.append({
            "fold": fold_idx,
            "gene_names": gene_names,
            "shap_values": shap_vals,
            "layerwise": layerwise,
            "pathway_df": pathway_df,
            "ppi_df": ppi_df,
            "ppi_shap_df": ppi_shap_df,
            "pathway_shap_df": pathway_shap_df,
            "p2p_shap_df": p2p_shap_df,
            "cascade": cascade,
        })

        log.info(
            f"  Fold {fold_idx}: top gene = "
            f"{max(dict(zip(gene_names, np.abs(shap_vals).mean(0))).items(), key=lambda x: x[1], default=('?', 0))[0]}"
            f" | top pathway = "
            f"{pathway_df['pathway_name'].iloc[0] if not pathway_df.empty else 'N/A'}"
        )

    # ── Aggregate across folds ────────────────────────────────────────────────
    cross_fold = aggregate_shap_across_folds(fold_results)

    # ── Dual-level aggregation (Prompt 8) ─────────────────────────────────────
    aggregate_dual_shap(fold_results, output_dir=config.SHAP_DIR, n_folds_total=len(fold_results))

    # ── Save ─────────────────────────────────────────────────────────────────
    save_shap_results(cross_fold, fold_results)

    # ── Summary ───────────────────────────────────────────────────────────────
    stab = cross_fold.get("gene_rank_stability", pd.DataFrame())
    print()
    print("═" * 60)
    print("  SHAP Analysis Summary")
    print("═" * 60)
    if not stab.empty:
        print("Top genes by SHAP (consistent across folds):")
        print(stab.head(10).to_string(index=False))
    pw = cross_fold.get("pathway_mean_shap", pd.DataFrame())
    if not pw.empty and "pathway_name" in pw.columns:
        print("\nTop pathways by mean SHAP:")
        print(pw[["pathway_name", "mean_abs_shap"]].head(10).to_string(index=False))
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

def run_saved_shap_analysis() -> None:
    """
    Load saved fold models + data from nested CV output and run SHAP analysis.

    Expects:
      outputs/models/binn_fold{k}.pt         — state dicts
      data/processed/expression_matrix.parquet — expression matrix
      data/processed/labels.csv                — labels
      data/processed/bio_map.pkl             — biological map
      data/processed/fold_network_info.pkl   — list of fold network dicts
                                               (saved during nested CV)
    """
    fold_network_path = config.FOLD_NETWORK_INFO_FILE
    bio_map_path = config.BIO_MAP_FILE

    if not os.path.exists(bio_map_path):
        raise FileNotFoundError(f"bio_map.pkl not found at {bio_map_path}")
    if not os.path.exists(fold_network_path):
        raise FileNotFoundError(
            f"fold_network_info.pkl not found at {fold_network_path}. "
            "Re-run nested CV so fold network metadata is saved."
        )

    with open(bio_map_path, "rb") as fh:
        bio_map = pickle.load(fh)
    with open(fold_network_path, "rb") as fh:
        fold_network_info = pickle.load(fh)

    # Load expression + labels for fold reconstruction
    from src.data_acquisition import load_preprocessed_data
    expr_df, labels_s = load_preprocessed_data()

    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from src.data_acquisition import compute_mad, apply_mad_filter

    outer_cv = StratifiedKFold(
        n_splits=config.OUTER_FOLDS, shuffle=True, random_state=config.RANDOM_SEED
    )
    X_all = expr_df.values.astype(np.float32)
    y_all = labels_s.values.astype(int)
    all_genes = expr_df.columns.tolist()

    fold_models_paths: list = []
    fold_data_list: list = []
    fold_network_subset: list = []

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X_all, y_all)):
        model_path = os.path.join(config.MODEL_DIR, f"binn_fold{fold_idx}.pt")
        if not os.path.exists(model_path):
            log.warning(f"Missing model for fold {fold_idx}: {model_path}")
            continue
        if fold_idx >= len(fold_network_info) or fold_network_info[fold_idx] is None:
            log.warning(
                f"Missing fold-network metadata for fold {fold_idx}; "
                "skipping SHAP for this fold."
            )
            continue

        # Reproduce MAD filter + scaler (same pipeline as nested_cv.py)
        import pandas as _pd
        train_df = _pd.DataFrame(X_all[train_idx], columns=all_genes)
        mad = compute_mad(train_df)
        train_filt = apply_mad_filter(train_df, mad, config.MAD_PERCENTILE)
        kept = train_filt.columns.tolist()
        test_filt = _pd.DataFrame(X_all[test_idx], columns=all_genes)[kept]

        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(train_filt.values).astype(np.float32)
        X_te_sc = scaler.transform(test_filt.values).astype(np.float32)

        net = fold_network_info[fold_idx]
        binn_genes = net["layer_node_names"][0]
        gene_to_idx = {g: i for i, g in enumerate(kept)}
        binn_idx = [gene_to_idx[g] for g in binn_genes if g in gene_to_idx]
        if not binn_idx:
            log.warning(
                f"Fold {fold_idx}: no overlap between network genes and kept genes; skipping."
            )
            continue

        fold_models_paths.append(model_path)
        fold_network_subset.append(net)
        fold_data_list.append({
            "X_train": X_tr_sc[:, binn_idx],
            "X_test": X_te_sc[:, binn_idx],
            "y_train": y_all[train_idx],
            "y_test": y_all[test_idx],
            "gene_names": binn_genes,
        })

    if not fold_models_paths:
        raise RuntimeError("No fold models and matching network metadata available for SHAP.")

    run_shap_analysis(fold_models_paths, fold_data_list, bio_map, fold_network_subset)


if __name__ == "__main__":
    run_saved_shap_analysis()
