# Glowing Chainsaw

## Project layout

- `binn/`: runnable Python project (all code + configs + outputs)
- `prompts/`: workflow/design prompt docs

## Prerequisites

- Python `3.13+`
- [`uv`](https://docs.astral.sh/uv/) installed
- Internet access for GEO download (`GSE40774`) during preprocessing

## Setup

```bash
cd binn
uv sync
```

## Git And Git LFS (Large Files)

Largest files currently in this workspace:
- `binn/data/string/9606.protein.links.v12.0.txt` (~602 MB)
- `binn/data/string/uniprotkb.tsv` (~365 MB)
- `binn/data/reactome/UniProt2Reactome.txt` (~40 MB)
- `binn/data/raw/GSE40774_family.soft.gz` (~20 MB)

Set up Git LFS before adding large data files:

```bash
# if not installed yet (macOS)
brew install git-lfs

# one-time install per machine
git lfs install

# track large dataset patterns in this repo
git lfs track "binn/data/string/*.txt"
git lfs track "binn/data/string/*.tsv"
git lfs track "binn/data/reactome/*.txt"
git lfs track "binn/data/raw/*.gz"

# .gitattributes is created/updated by git lfs track
git add .gitattributes
```

Commit and push workflow:

```bash
# from repo root
git add .gitignore binn
git commit -m "Add BINN pipeline, data, and cleanup tooling"

# set branch + remote (if needed)
git branch -M main
git remote add origin <your-github-repo-url>

# first push
git push -u origin main
```

Helpful checks:

```bash
# view files currently tracked by LFS
git lfs ls-files

# verify large files in working tree
find . -type f -size +10M | sort
```

## Run everything

From `binn/`:

```bash
uv run python main.py all
```

Equivalent console script:

```bash
uv run binn all
```

This runs, in order:
1. preprocessing
2. biological mapping
3. nested CV (BINN + baselines)
4. SHAP analysis
5. figure generation

## Run stages individually

From `binn/`:

```bash
uv run python main.py preprocess
uv run python main.py biomap
uv run python main.py cv
uv run python main.py shap
uv run python main.py viz
uv run python main.py clean
```

`viz` supports optional paths:

```bash
uv run python main.py viz \
  --results-dir outputs/metrics \
  --shap-dir outputs/shap \
  --figure-dir outputs/figures
```

Cleanup options:

```bash
# Preview deletions only
uv run python main.py clean --dry-run

# Remove outputs + generated processed artifacts
uv run python main.py clean --full-reset
```

## Key input/output files

Preprocessing outputs (`binn/data/processed/`):
- `expression_matrix.parquet`
- `labels.csv`
- `gene_list.txt`

Biological mapping outputs (`binn/data/processed/`):
- `bio_map.pkl`
- `gene_to_reactome.json`
- `reactome_hierarchy_edges.csv`
- `reactome_names.json`
- `ppi_edges.csv`

Nested CV outputs:
- `binn/outputs/models/binn_fold*.pt`
- `binn/outputs/metrics/nested_cv_results.json`
- `binn/outputs/metrics/summary_table.csv`
- `binn/outputs/metrics/statistical_tests.csv`
- `binn/outputs/metrics/training_histories.json`
- `binn/data/processed/fold_network_info.pkl`

SHAP outputs (`binn/outputs/shap/`):
- `gene_shap_values.csv`
- `pathway_importance.csv`
- `ppi_importance.csv`
- `hpv_cascade.json`

Figures (`binn/outputs/figures/`):
- ROC/PR/Spec-Sens curves
- confusion matrices
- SHAP plots
- pathway/PPI/cascade figures

## Notes

- `config.py` is the single source of truth for paths, hyperparameters, and device selection.
- Device selection is automatic and uses priority `CUDA GPU > Apple MPS > CPU`.
- Startup logs include the selected `DEVICE_IDENTIFIER` (for example GPU model + VRAM, or CPU identifier).
- BINN trains on the selected accelerator when safe; oversized fold-specific layers auto-fallback to CPU.
- XGBoost uses CUDA when available, otherwise CPU, and automatically retries on CPU if CUDA is unavailable in the local XGBoost build.
- sklearn baselines (`svm_rbf`, `knn`, `random_forest`) run on CPU backends.
- If SHAP dependencies are unavailable, attribution automatically falls back to Gradient × Input.
- The default network build is memory-safe and performance-oriented:
  - `PATHWAY_LAYERING_MODE="distance_from_mapped"`
  - `USE_PPI_ENRICHMENT=True`
  - sink-bridge routing enabled (`ENABLE_SINK_BRIDGES=True`)
  - automatic device fallback for oversized layers
