"""
Central configuration for the BINN-HPV project.
All paths, hyperparameters, and device settings live here.
"""
import os
import platform

import torch

# ── Device ────────────────────────────────────────────────────────────
DEVICE_PRIORITY = ("cuda", "mps", "cpu")


def _mps_is_available() -> bool:
    """Return True when the current PyTorch build exposes an available MPS device."""
    mps_backend = getattr(torch.backends, "mps", None)
    return bool(mps_backend and mps_backend.is_available())


def _select_torch_device() -> torch.device:
    """Select compute device with priority CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if _mps_is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _build_device_identifier(device: torch.device) -> str:
    """Return a human-readable identifier for the selected compute device."""
    if device.type == "cuda":
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        mem_gib = props.total_memory / (1024**3)
        return f"cuda:{idx} | {props.name} | {mem_gib:.1f} GiB VRAM"
    if device.type == "mps":
        return "mps | Apple Metal Performance Shaders"
    cpu_name = platform.processor() or platform.machine() or "CPU"
    return f"cpu | {cpu_name}"


DEVICE = _select_torch_device()
DEVICE_IDENTIFIER = _build_device_identifier(DEVICE)
XGBOOST_DEVICE = "cuda" if DEVICE.type == "cuda" else "cpu"

# ── Paths ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_RAW = os.path.join(BASE_DIR, "data", "raw")
DATA_REACTOME = os.path.join(BASE_DIR, "data", "reactome")
DATA_STRING = os.path.join(BASE_DIR, "data", "string")
DATA_PROCESSED = os.path.join(BASE_DIR, "data", "processed")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
METRIC_DIR = os.path.join(OUTPUT_DIR, "metrics")
SHAP_DIR = os.path.join(OUTPUT_DIR, "shap")
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")

# ── Core artifacts ────────────────────────────────────────────────────
EXPRESSION_MATRIX_FILE = os.path.join(DATA_PROCESSED, "expression_matrix.parquet")
LABELS_FILE = os.path.join(DATA_PROCESSED, "labels.csv")
GENE_LIST_FILE = os.path.join(DATA_PROCESSED, "gene_list.txt")
BIO_MAP_FILE = os.path.join(DATA_PROCESSED, "bio_map.pkl")
FOLD_NETWORK_INFO_FILE = os.path.join(DATA_PROCESSED, "fold_network_info.pkl")

for d in [
    DATA_RAW,
    DATA_REACTOME,
    DATA_STRING,
    DATA_PROCESSED,
    MODEL_DIR,
    METRIC_DIR,
    SHAP_DIR,
    FIGURE_DIR,
]:
    os.makedirs(d, exist_ok=True)

# ── GEO Dataset ──────────────────────────────────────────────────────
GEO_ACCESSION = "GSE40774"
METADATA_COLUMN = "characteristics_ch1.3.hpv status"
GENE_SYMBOL_COLUMN = "GENE_SYMBOL"
POSITIVE_LABELS = {"positive", "pos", "hpv+", "hpv positive", "yes", "1"}
NEGATIVE_LABELS = {"negative", "neg", "hpv-", "hpv negative", "no", "0"}

# ── Preprocessing ────────────────────────────────────────────────────
MAD_PERCENTILE = 50  # median — discard bottom 50% by MAD
FLOAT_DTYPE = "float32"

# ── Cross-Validation ─────────────────────────────────────────────────
OUTER_FOLDS = 5
INNER_FOLDS = 3
RANDOM_SEED = 42

# ── Training Hyperparameters ─────────────────────────────────────────
EPOCHS = 200
EARLY_STOPPING_PATIENCE = 20
BATCH_SIZE = 16
DROPOUT_RATE = 0.3

# ── Muon Optimizer (for 2D hidden weight matrices) ──────────────────
# torch.optim.Muon requires PyTorch >= 2.9; falls back to AdamW if unavailable
MUON_LR = 0.02
MUON_MOMENTUM = 0.95
MUON_WEIGHT_DECAY = 0.01
MUON_NESTEROV = True

# ── AdamW Optimizer (for biases, BatchNorm, output layer) ───────────
ADAM_LR = 1e-3
ADAM_WEIGHT_DECAY = 1e-4
ADAM_BETAS = (0.9, 0.95)

# ── Activation ──────────────────────────────────────────────────────
# Using torch.nn.Mish() — Muon is the OPTIMIZER, Mish is the ACTIVATION
ACTIVATION = "mish"

# ── Reactome / STRING Files ─────────────────────────────────────────
REACTOME_PATHWAYS_FILE = os.path.join(DATA_REACTOME, "ReactomePathways.txt")
REACTOME_RELATIONS_FILE = os.path.join(DATA_REACTOME, "ReactomePathwaysRelation.txt")
UNIPROT_REACTOME_FILE = os.path.join(DATA_REACTOME, "UniProt2Reactome.txt")
UNIPROTKB_FILE = os.path.join(DATA_STRING, "uniprotkb.tsv")
STRING_LINKS_FILE = os.path.join(DATA_STRING, "9606.protein.links.v12.0.txt")
STRING_CONFIDENCE_THRESHOLD = 400  # minimum combined_score (400=medium, 700=high, 900=highest)

# ── Network construction safety knobs ───────────────────────────────
# "all" keeps all mapped pathways for each gene.
# "deepest" keeps only the most specific mapped pathways per gene.
GENE_PATHWAY_MAPPING_MODE = "all"  # {"deepest", "all"}

# Layer assignment for pathway nodes:
# - "distance_from_mapped": shortest ancestor distance from mapped pathways
# - "longest_path": original global-depth layering
PATHWAY_LAYERING_MODE = "distance_from_mapped"  # {"distance_from_mapped", "longest_path"}

# PPI expansion adds gene→pathway edges using neighboring genes in STRING.
USE_PPI_ENRICHMENT = True

# Add lightweight bridge nodes so pathway sinks at lower layers still connect
# to OUTPUT without exploding depth padding.
ENABLE_SINK_BRIDGES = True

# If a single masked layer would exceed this size (float32 GiB), fall back to CPU.
MAX_DEVICE_MATRIX_GIB = 1.5

# Skip depth padding when estimated dense parameter count gets too large.
MAX_DENSE_PARAMS_AFTER_PADDING = 50_000_000
