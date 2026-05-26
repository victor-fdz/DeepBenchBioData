#!/usr/bin/env python3
"""
Benchmark profiling metrics for cross-species gene expression data.

This pipeline:
- generates non-orthologous gene pairs from the raw dataset
- applies profiling metrics to orthologous and non-orthologous datasets
- benchmarks metric performance using KS statistics
- optionally compares expression similarity vs promoter sequence identity

Usage:
    python bin/profiling.py --input data/kinases.tsv --name kinases

Optional:
    python bin/profiling.py \
        --input data/kinases.tsv \
        --name kinases \
        --promoter data/promoter_alignment.tsv
"""

import argparse
import logging
import sys
from pathlib import Path

# allow direct execution from repository root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from lib import expression_vs_sequence as evs
from lib import modify_dataset as mod
from lib import profiling_functions as pf

# -----------------------------
# Logging configuration
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="\033[1;32m%(levelname)s\033[0m | \033[1;36m%(name)s\033[0m | %(message)s",
)

logger = logging.getLogger(__name__)

# -----------------------------
# Constants
# -----------------------------
OUTPUT_DIR = Path("results")


# -----------------------------
# CLI
# -----------------------------
def parse_args(cli_args: list[str] | None = None) -> argparse.Namespace:
    """
    Parse command-line arguments.

    Args:
        None

    Returns:
        argparse.Namespace: Parsed user arguments
    """

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to input TSV containing raw TPM values.",
    )

    parser.add_argument(
        "--name",
        required=True,
        help="Dataset name used for output naming.",
    )

    parser.add_argument(
        "--promoter",
        default=None,
        type=Path,
        help="Optional promoter alignment TSV for sequence comparison." \
        "Format: ID_Gene_human, ID_Gene_mouse, promoter_identity",
    )

    return parser.parse_args(cli_args)


# -----------------------------
# Dataset helpers
# -----------------------------
def extract_tissues(df: pd.DataFrame) -> list[str]:
    """
    Extract tissue names from TPM column names.

    Expected column format:
        <tissue>_tpm_<species>

    Args:
        - df (pd.DataFrame): Input expression dataframe

    Returns:
        list[str]: Sorted tissue names
    """

    features = [c for c in df.columns if c.split("_")[-2] == "tpm"]

    return sorted({
        c.rsplit("_", 2)[0]
        for c in features
    })


# -----------------------------
# Main pipeline
# -----------------------------
def main(cli_args: list[str] | None = None) -> str:
    """
    Run profiling benchmarking pipeline.

    Steps:
    - load raw expression dataset
    - generate non-orthologous pairs
    - apply profiling metrics
    - benchmark profiling methods
    - optionally compare expression vs sequence similarity

    Args:
        cli_args (list[str] | None): Command-line arguments

    Returns:
        str: Best profiling metric
    """

    args = parse_args() if cli_args is None else parse_args(cli_args)

    # load raw dataset
    logger.info("Loading dataset.")
    df = pd.read_csv(args.input, sep="\t")

    # extract tissue names
    tissues = extract_tissues(df)

    # generate non-orthologous dataset
    logger.info("Generating non-orthologous dataset.")
    nonortho_df = mod.pairs_exchange(
        df=df,
        df_name=args.name,
        output_dir=OUTPUT_DIR,
    )

    # apply profiling metrics to orthologous dataset
    logger.info("Applying profiling metrics to orthologous dataset.")
    df_ortho_metrics = pf.apply_metrics(
        df=df,
        tissues=tissues,
    )

    # apply profiling metrics to non-orthologous dataset
    logger.info("Applying profiling metrics to non-orthologous dataset.")
    df_nonortho_metrics = pf.apply_metrics(
        df=nonortho_df,
        tissues=tissues,
    )

    # save profiling outputs
    profiling_dir = OUTPUT_DIR / args.name
    profiling_dir.mkdir(parents=True, exist_ok=True)

    common_path = profiling_dir / "Intermediate_Datasets"
    ortho_path = common_path / f"{args.name}_Orthologs_metrics.tsv"
    nonortho_path = common_path / f"{args.name}_NonOrthologs_metrics.tsv"

    df_ortho_metrics.to_csv(ortho_path, sep="\t", index=False)
    df_nonortho_metrics.to_csv(nonortho_path, sep="\t", index=False)

    logger.info("Profiling datasets saved.")

    # benchmark profiling metrics
    logger.info("Benchmarking profiling metrics.")
    pf.benchmark_profiling(
        df_ortho=df_ortho_metrics,
        df_nonortho=df_nonortho_metrics,
        df_name=args.name,
        output_dir=OUTPUT_DIR,
    )

    # select best profiling method
    best_metric = pf.best_method(
        df_ortho=df_ortho_metrics,
        df_nonortho=df_nonortho_metrics,
    )

    logger.info("Best profiling metric: %s", best_metric)

    # optional sequence comparison
    if args.promoter is not None:

        logger.info("Running expression vs sequence comparison.")

        evs.expression_vs_sequence(
            df_path=args.input,
            df_name=args.name,
            metric_func_name=best_metric,
            tissues=tissues,
            promoter_path=args.promoter,
            output_dir=OUTPUT_DIR,
        )

    return best_metric


if __name__ == "__main__":
    main()