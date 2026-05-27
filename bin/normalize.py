#!/usr/bin/env python3
"""Benchmark normalization methods for cross-species gene expression data.

Generates orthologous and non-orthologous normalized datasets, computes
Pearson/Spearman correlations per tissue and method, and produces scatter
and heatmap summary plots.

Usage:
    python bin/normalize.py --input data/kinases.tsv --name kinases --tissue General
"""

import pandas as pd
import argparse
import logging
import sys
from pathlib import Path

# Add project root to Python path for direct execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib import modify_dataset  as mod
from lib import normalization   as normfx
from lib import plotting        as pt
from lib import compute_stats   as compst

# -----------------------------
# Setup
# -----------------------------
logging.basicConfig(
    level=logging.INFO, 
    format="\033[1;32m%(levelname)s\033[0m | \033[1;36m%(name)s\033[0m | %(message)s"
)

logger = logging.getLogger(__name__)

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
        argparse.Namespace: Parsed CLI arguments
    """

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to input TSV with TPM values",
    )

    parser.add_argument(
        "--name",
        required=True,
        help="Dataset name (used for output naming)",
    )

    parser.add_argument(
        "--tissue",
        default="General",
        help="Tissue used for method ranking",
    )

    return parser.parse_args(cli_args)


# -----------------------------
# Main pipeline
# -----------------------------
def main(cli_args: list[str] | None = None) -> str:
    """
    Execute full normalization benchmarking pipeline.

    Steps:
    - Load data
    - Generate non-ortholog pairs
    - Apply normalization
    - Compute statistics
    - Plot results
    - Select best method

    Returns:
        str: Best normalization method
    """

    args = parse_args() if cli_args is None else parse_args(cli_args)

    # -----------------------------
    # Load input data
    # -----------------------------
    df = pd.read_csv(args.input, sep="\t")

    # extract TPM feature columns
    features = [c for c in df.columns if c.split("_")[-2] == "tpm"]

    # extract tissues from feature names
    tissues = list({c.rsplit("_", 2)[0] for c in features})

    # -----------------------------
    # Generate all non-ortholog pairs
    # -----------------------------
    nonortho_df = mod.all_nonortholog_pairs(df, args.name, OUTPUT_DIR)

    # -----------------------------
    # Apply normalization
    # -----------------------------
    normfx.apply_normalizations(
        df=df,
        df_name=args.name,
        features=features,
        orthology="Orthologs",
        output_dir=OUTPUT_DIR,
    )

    normfx.apply_normalizations(
        df=nonortho_df,
        df_name=args.name,
        features=features,
        orthology="NonOrthologs",
        output_dir=OUTPUT_DIR,
    )

    # -----------------------------
    # Load normalized datasets
    # -----------------------------
    dfs_ortho, names_ortho = normfx.load_normalized_data(
        args.name, "Orthologs", OUTPUT_DIR
    )

    dfs_nonortho, names_nonortho = normfx.load_normalized_data(
        args.name, "NonOrthologs", OUTPUT_DIR
    )

    # -----------------------------
    # Compute statistics
    # -----------------------------
    stats_ortho = compst.compute_stats(
        dfs_ortho, names_ortho, features, tissues, "Orthologs"
    )

    stats_nonortho = compst.compute_stats(
        dfs_nonortho, names_nonortho, features, tissues, "NonOrthologs"
    )

    stats_all = compst.merge_and_increment(
        args.name,
        stats_ortho,
        stats_nonortho,
        OUTPUT_DIR,
    )

    # -----------------------------
    # Plot results
    # -----------------------------
    ortho_dict = dict(zip(names_ortho, dfs_ortho))
    nonortho_dict = dict(zip(names_nonortho, dfs_nonortho))

    pt.plot_correlations(
        ortho_dict,
        stats_all,
        tissues,
        args.name,
        "Orthologs",
        OUTPUT_DIR,
    )

    pt.plot_correlations(
        nonortho_dict,
        stats_all,
        tissues,
        args.name,
        "NonOrthologs",
        OUTPUT_DIR,
    )

    # rank methods
    sorted_methods = pt.sort_methods_by_correlation(
        stats_all,
        orthology="Increment",
        tissue=args.tissue,
    )

    pt.plot_heatmap(
        stats_all,
        args.name,
        sorted_methods,
        "Pearson_R",
        "Increment",
        OUTPUT_DIR,
    )

    # -----------------------------
    # Select best method
    # -----------------------------
    best = compst.best_method(args.name, args.tissue, OUTPUT_DIR)

    logger.info("Best normalization method: %s", best)

    # -----------------------------
    # sanity check: method validity
    # -----------------------------
    method_names = [
        "_".join(m.split("_")[:-1]) if len(m.split("_")) > 1 else m
        for m in sorted_methods
    ]

    if best not in method_names:
        logger.warning(
            "Best method '%s' not in available methods: %s",
            best,
            ", ".join(method_names),
        )

    return best


if __name__ == "__main__":
    main()
