"""
Cleanup utilities for generated BINN artifacts.

Default cleanup removes model/results/figure outputs and Python caches.
Full reset also removes generated files in data/processed/.
"""
from __future__ import annotations

import logging
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


_OUTPUT_DIRS = [
    config.MODEL_DIR,
    config.METRIC_DIR,
    config.SHAP_DIR,
    config.FIGURE_DIR,
]

_ALWAYS_REMOVE_FILES = [
    os.path.join(config.OUTPUT_DIR, ".DS_Store"),
    config.FOLD_NETWORK_INFO_FILE,
]

_FULL_RESET_PROCESSED_FILES = [
    config.EXPRESSION_MATRIX_FILE,
    config.LABELS_FILE,
    config.GENE_LIST_FILE,
    config.BIO_MAP_FILE,
    os.path.join(config.DATA_PROCESSED, "coverage_stats.json"),
    os.path.join(config.DATA_PROCESSED, "gene_to_reactome.json"),
    os.path.join(config.DATA_PROCESSED, "reactome_hierarchy_edges.csv"),
    os.path.join(config.DATA_PROCESSED, "reactome_names.json"),
    os.path.join(config.DATA_PROCESSED, "ppi_edges.csv"),
    os.path.join(config.DATA_PROCESSED, "unmapped_genes.txt"),
    os.path.join(config.DATA_PROCESSED, "mygene_cache.json"),
]

_CACHE_DIRS = [
    os.path.join(config.BASE_DIR, "__pycache__"),
    os.path.join(config.BASE_DIR, "src", "__pycache__"),
]


def _remove_path(path: Path, dry_run: bool) -> bool:
    """Remove a file/dir if it exists. Returns True when something was removed."""
    if not path.exists():
        return False
    if dry_run:
        log.info(f"[dry-run] remove {path}")
        return True
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()
    log.info(f"removed {path}")
    return True


def _clear_directory(path: Path, dry_run: bool) -> int:
    """
    Remove all children in `path` while keeping the directory itself.

    Creates the directory if missing (except in dry-run mode).
    """
    removed = 0
    if not path.exists():
        if dry_run:
            log.info(f"[dry-run] create directory {path}")
        else:
            path.mkdir(parents=True, exist_ok=True)
        return removed

    for child in path.iterdir():
        removed += int(_remove_path(child, dry_run=dry_run))

    if not dry_run:
        path.mkdir(parents=True, exist_ok=True)
    return removed


def run_cleanup(full_reset: bool = False, dry_run: bool = False) -> dict[str, int]:
    """
    Remove generated artifacts.

    Parameters
    ----------
    full_reset : bool
        If True, also remove generated files in data/processed/.
    dry_run : bool
        If True, only report what would be removed.

    Returns
    -------
    dict with removal counts by category.
    """
    removed_output_items = 0
    removed_files = 0
    removed_cache_dirs = 0

    for out_dir in _OUTPUT_DIRS:
        removed_output_items += _clear_directory(Path(out_dir), dry_run=dry_run)

    for fp in _ALWAYS_REMOVE_FILES:
        removed_files += int(_remove_path(Path(fp), dry_run=dry_run))

    for cache_dir in _CACHE_DIRS:
        removed_cache_dirs += int(_remove_path(Path(cache_dir), dry_run=dry_run))

    removed_processed_files = 0
    if full_reset:
        for fp in _FULL_RESET_PROCESSED_FILES:
            removed_processed_files += int(_remove_path(Path(fp), dry_run=dry_run))

    summary = {
        "output_items": removed_output_items,
        "files": removed_files,
        "cache_dirs": removed_cache_dirs,
        "processed_files": removed_processed_files,
    }
    mode = "dry-run" if dry_run else "done"
    log.info(
        "Cleanup %s: output_items=%d, files=%d, cache_dirs=%d, processed_files=%d",
        mode,
        summary["output_items"],
        summary["files"],
        summary["cache_dirs"],
        summary["processed_files"],
    )
    return summary


if __name__ == "__main__":
    run_cleanup(full_reset=False, dry_run=False)
