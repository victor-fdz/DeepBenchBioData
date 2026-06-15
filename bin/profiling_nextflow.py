#!/usr/bin/env python3
"""Nextflow wrapper for profiling metric benchmarking.

This script reuses the existing project modules:
- lib.modify_dataset
- lib.profiling_functions
- lib.expression_vs_sequence

It only changes what is required for Nextflow:
- process-local outputs
- stable output folders declared by Nextflow
- non-interactive best metric reporting
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib import expression_vs_sequence as evs
from lib import modify_dataset as mod
from lib import profiling_functions as pf

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="\033[1;32m%(levelname)s\033[0m | \033[1;36m%(name)s\033[0m | %(message)s",
)

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(".")


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to input TSV containing TPM values.",
    )

    parser.add_argument(
        "--dataset-name",
        "--name",
        dest="dataset_name",
        required=True,
        help="Dataset name used for metadata and temporary process-local naming.",
    )

    parser.add_argument(
        "--promoter",
        default=None,
        type=Path,
        help=(
            "Optional promoter alignment TSV for expression-vs-sequence comparison. "
            "Usually not needed for the basic profiling benchmark."
        ),
    )

    return parser.parse_args()


# -----------------------------
# Helpers
# -----------------------------
def extract_tissues(dataframe: pd.DataFrame) -> list[str]:
    """Extract tissue names from TPM column names."""

    features = [
        column
        for column in dataframe.columns
        if column.split("_")[-2] == "tpm"
    ]

    return sorted({
        column.rsplit("_", 2)[0]
        for column in features
    })


def copy_directory_contents(source_dir: Path, target_dir: Path) -> None:
    """Replace target_dir with a copy of source_dir."""

    if target_dir.exists():
        shutil.rmtree(target_dir)

    if source_dir.exists():
        shutil.copytree(source_dir, target_dir)
    else:
        target_dir.mkdir(parents=True, exist_ok=True)


def stage_outputs_for_nextflow(dataset_name: str) -> None:
    """Expose stable process-local outputs at paths declared in the Nextflow module."""

    dataset_dir = OUTPUT_DIR / dataset_name

    copy_directory_contents(dataset_dir / "Intermediate_Datasets", Path("Intermediate_Datasets"))
    copy_directory_contents(dataset_dir / "Benchmarking", Path("Benchmarking"))
    copy_directory_contents(dataset_dir / "Expression_vs_Sequence", Path("Expression_vs_Sequence"))
    copy_directory_contents(dataset_dir / "Profiling", Path("Profiling"))

def metric_column_to_method_name(metric_column: str) -> str:
    """Convert selected metric dataframe column into CLI method name."""

    internal_metric_names = {
        function.__name__
        for function in pf.INTERNAL_METRICS
    }

    for metric_name in internal_metric_names:
        if metric_column == f"{metric_name}_sim":
            return metric_name

    return metric_column

# -----------------------------
# Main
# -----------------------------
def main() -> str:
    args = parse_args()

    logger.info("Loading expression dataset: %s", args.input)
    dataframe = pd.read_csv(args.input, sep="\t")

    tissues = extract_tissues(dataframe)

    logger.info("Detected %d tissues: %s", len(tissues), ", ".join(tissues))

    # -----------------------------
    # Generate deterministic non-ortholog background
    # -----------------------------
    logger.info("Generating all non-orthologous human x mouse pairs.")

    if not hasattr(mod, "all_nonortholog_pairs"):
        raise AttributeError(
            "lib.modify_dataset.all_nonortholog_pairs is missing. "
            "Apply the normalization all-nonortholog update before running profiling."
        )

    nonortholog_dataframe = mod.all_nonortholog_pairs(
        df=dataframe,
        df_name=args.dataset_name,
        output_dir=OUTPUT_DIR,
    )

    # -----------------------------
    # Apply profiling metrics
    # -----------------------------
    logger.info("Applying profiling metrics to orthologous dataset.")
    ortholog_metrics = pf.apply_metrics(
        df=dataframe,
        tissues=tissues,
    )

    logger.info("Applying profiling metrics to all non-orthologous pairs.")
    nonortholog_metrics = pf.apply_metrics(
        df=nonortholog_dataframe,
        tissues=tissues,
    )

    # -----------------------------
    # Save profiled datasets
    # -----------------------------
    intermediate_dir = OUTPUT_DIR / args.dataset_name / "Intermediate_Datasets"
    intermediate_dir.mkdir(parents=True, exist_ok=True)

    ortholog_metrics_path = intermediate_dir / f"{args.dataset_name}_Orthologs_metrics.tsv"
    nonortholog_metrics_path = intermediate_dir / f"{args.dataset_name}_NonOrthologs_metrics.tsv"

    ortholog_metrics.to_csv(ortholog_metrics_path, sep="\t", index=False)
    nonortholog_metrics.to_csv(nonortholog_metrics_path, sep="\t", index=False)

    logger.info("Profiled datasets saved to %s", intermediate_dir)

    # -----------------------------
    # Benchmark metrics
    # -----------------------------
    logger.info("Benchmarking profiling metrics.")

    ranking_dataframe = pf.benchmark_profiling(
        df_ortho=ortholog_metrics,
        df_nonortho=nonortholog_metrics,
        df_name=args.dataset_name,
        output_dir=OUTPUT_DIR,
    )

    best_metric_column = pf.best_method(
    df_ortho=ortholog_metrics,
    df_nonortho=nonortholog_metrics,
)

    best_metric = metric_column_to_method_name(best_metric_column)

    logger.info("Best profiling metric column: %s", best_metric_column)
    logger.info("Best profiling method: %s", best_metric)

    # -----------------------------
    # Stable profiling summary outputs
    # -----------------------------
    profiling_dir = OUTPUT_DIR / args.dataset_name / "Profiling"
    profiling_dir.mkdir(parents=True, exist_ok=True)

    ranking_path = profiling_dir / "profiling_method_ranking.tsv"
    ranking_dataframe.to_csv(ranking_path, sep="\t", index=False)

    best_metric_path = profiling_dir / "best_metric.txt"
    best_metric_path.write_text(best_metric + "\n")

    manifest = {
        "input": str(args.input),
        "dataset_name": args.dataset_name,
        "tissues": tissues,
        "nonortholog_strategy": "all human x mouse pairs minus true ortholog pairs",
        "benchmark": "two-sample Kolmogorov-Smirnov statistic between ortholog and non-ortholog metric distributions",
        "best_metric": best_metric,
        "profiling_module": "lib.profiling_functions",
        "nextflow_output_policy": "stable process-local output folders are declared by the Nextflow process",
    }

    manifest_path = profiling_dir / "profiling_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    # -----------------------------
    # Optional expression-vs-sequence comparison
    # -----------------------------
    if args.promoter is not None:
        logger.info("Running optional expression-vs-sequence comparison.")

        evs.expression_vs_sequence(
            df_path=args.input,
            df_name=args.dataset_name,
            metric_func_name=best_metric,
            tissues=tissues,
            promoter_path=args.promoter,
            output_dir=OUTPUT_DIR,
        )

    # -----------------------------
    # Stage stable outputs for Nextflow
    # -----------------------------
    stage_outputs_for_nextflow(args.dataset_name)

    logger.info("Profiling outputs generated in process working directory.")

    return best_metric


if __name__ == "__main__":
    main()
