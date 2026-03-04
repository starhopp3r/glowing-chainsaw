"""Command-line entrypoint for the BINN workflow."""
from __future__ import annotations

import argparse
import logging
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import config

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run BINN preprocessing, training, SHAP analysis, and visualizations."
    )
    parser.add_argument(
        "command",
        choices=["preprocess", "biomap", "cv", "shap", "viz", "clean", "all"],
        help="Pipeline stage to run.",
    )
    parser.add_argument(
        "--results-dir",
        default=config.METRIC_DIR,
        help="Metrics directory for visualization input.",
    )
    parser.add_argument(
        "--shap-dir",
        default=config.SHAP_DIR,
        help="SHAP results directory for visualization input.",
    )
    parser.add_argument(
        "--figure-dir",
        default=config.FIGURE_DIR,
        help="Output directory for generated figures.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="For 'clean': print what would be removed without deleting.",
    )
    parser.add_argument(
        "--full-reset",
        action="store_true",
        help=(
            "For 'clean': also remove generated data files in data/processed/ "
            "(in addition to outputs/models/metrics/shap/figures)."
        ),
    )
    return parser


def _run_step(step: str, dry_run: bool = False, full_reset: bool = False) -> None:
    if step == "preprocess":
        from src.data_acquisition import run_preprocessing
        run_preprocessing()
    elif step == "biomap":
        from src.biological_mapping import run_biological_mapping
        run_biological_mapping()
    elif step == "cv":
        from src.nested_cv import run_nested_cv
        run_nested_cv()
    elif step == "shap":
        from src.shap_analysis import run_saved_shap_analysis
        run_saved_shap_analysis()
    elif step == "viz":
        from src.visualization import generate_all_figures
        generate_all_figures()
    elif step == "clean":
        from src.cleanup import run_cleanup
        run_cleanup(full_reset=full_reset, dry_run=dry_run)
    else:
        raise ValueError(f"Unknown step '{step}'")


def main() -> None:
    args = _build_parser().parse_args()
    log.info(
        "Compute device priority=%s | selected=%s",
        " > ".join(d.upper() for d in config.DEVICE_PRIORITY),
        config.DEVICE_IDENTIFIER,
    )

    if args.command == "viz":
        from src.visualization import generate_all_figures
        generate_all_figures(
            results_dir=args.results_dir,
            shap_dir=args.shap_dir,
            figure_dir=args.figure_dir,
        )
        return

    if args.command == "all":
        from src.visualization import generate_all_figures
        for step in ("preprocess", "biomap", "cv", "shap"):
            log.info(f"Running step: {step}")
            _run_step(step, dry_run=args.dry_run, full_reset=args.full_reset)
        generate_all_figures(
            results_dir=args.results_dir,
            shap_dir=args.shap_dir,
            figure_dir=args.figure_dir,
        )
        return

    _run_step(args.command, dry_run=args.dry_run, full_reset=args.full_reset)


if __name__ == "__main__":
    main()
