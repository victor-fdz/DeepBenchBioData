#!/usr/bin/env python3
"""Nextflow wrapper for the normalization benchmark.

This script intentionally reuses the project normalization modules:
- lib.modify_dataset
- lib.normalization
- lib.compute_stats
- lib.plotting

It only changes what is required for Nextflow:
- no interactive method selection
- outputs are staged into stable process-local folders declared by Nextflow
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import pandas as pd

# Add project root to Python path for direct execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib import compute_stats as compst
from lib import modify_dataset as mod
from lib import normalization as normfx
from lib import plotting as pt

# -----------------------------
# Setup
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
        help="Path to input TSV with expression values.",
    )

    parser.add_argument(
        "--dataset-name",
        "--name",
        dest="dataset_name",
        required=True,
        help="Dataset name used for metadata and temporary process-local naming.",
    )

    parser.add_argument(
        "--tissue",
        default="General",
        help="Tissue used for automatic method ranking.",
    )

    parser.add_argument(
        "--selection-metric",
        default="pearson_increment",
        choices=[
            "pearson_increment",
            "spearman_increment",
            "pearson_ortho",
            "spearman_ortho",
        ],
        help="Metric used for automatic normalization method selection.",
    )

    parser.add_argument(
        "--expression-unit",
        default="tpm",
        choices=["tpm", "counts"],
    )

    return parser.parse_args()


# -----------------------------
# Helpers
# -----------------------------
def select_best_method(
    stats_dataframe: pd.DataFrame,
    tissue: str,
    selection_metric: str,
    expression_unit: str = "tpm",
) -> tuple[str, pd.DataFrame]:
    """Select the best normalization method without interactive input."""

    tissue_stats = stats_dataframe[stats_dataframe["tissue"] == tissue].copy()

    if tissue_stats.empty:
        available_tissues = sorted(stats_dataframe["tissue"].dropna().unique())
        raise ValueError(
            f"No normalization statistics found for tissue {tissue!r}. "
            f"Available tissues: {available_tissues}"
        )

    score_column_by_metric = {
        "pearson_increment": "Pearson_R_increment",
        "spearman_increment": "Spearman_rho_increment",
        "pearson_ortho": "Pearson_R_ortho",
        "spearman_ortho": "Spearman_rho_ortho",
    }

    score_column = score_column_by_metric[selection_metric]

    ranking = (
        tissue_stats
        .sort_values(score_column, ascending=False)
        .reset_index(drop=True)
    )

    ranking.insert(0, "rank", range(1, len(ranking) + 1))
    ranking["selection_tissue"] = tissue
    ranking["selection_metric"] = selection_metric
    ranking["selection_score_column"] = score_column

    selected_method = str(ranking.loc[0, "method"])

    if expression_unit == "counts" and selected_method == "original":
        if len(ranking) < 2:
            raise ValueError(
                "Best method is 'original' for raw counts, but no second method is available."
            )

        logger.warning(
            "Best method was 'original' with raw counts. Selecting second-best method instead."
        )

        selected_method = str(ranking.loc[1, "method"])

    return selected_method, ranking

def copy_directory_contents(source_dir: Path, target_dir: Path) -> None:
    """Replace target_dir with a copy of source_dir."""

    if target_dir.exists():
        shutil.rmtree(target_dir)

    shutil.copytree(source_dir, target_dir)


def create_nextflow_output_aliases(normalization_dir: Path) -> None:
    """Create stable lowercase plot names expected by the Nextflow module.

    lib.plotting.py controls the real figure generation. This function only
    creates predictable aliases for Nextflow output declarations.
    """

    alias_pairs = {
        "Orthologs_scatter.png": "orthologs_scatter.png",
        "NonOrthologs_scatter.png": "nonorthologs_scatter.png",
        "Increment_PearsonR_heatmap.png": "increment_pearson_heatmap.png",
    }

    for source_name, alias_name in alias_pairs.items():
        source_path = normalization_dir / source_name
        alias_path = normalization_dir / alias_name

        if source_path.exists():
            shutil.copy2(source_path, alias_path)


def stage_outputs_for_nextflow(dataset_name: str) -> None:
    """Expose stable process-local outputs at paths declared in the Nextflow module."""

    dataset_dir = OUTPUT_DIR / dataset_name

    copy_directory_contents(dataset_dir / "Orthologs", Path("Orthologs"))
    copy_directory_contents(dataset_dir / "NonOrthologs", Path("NonOrthologs"))
    copy_directory_contents(dataset_dir / "Intermediate_Datasets", Path("Intermediate_Datasets"))
    copy_directory_contents(dataset_dir / "Normalization", Path("Normalization"))


def get_feature_columns(df, expression_unit):
    return [c for c in df.columns if f"_{expression_unit}_" in c]


def get_tissues(feature_columns, expression_unit):
    return sorted({
        c.rsplit(f"_{expression_unit}_", 1)[0]
        for c in feature_columns
    })


def expr_col(tissue, species, expression_unit):
    return f"{tissue}_{expression_unit}_{species}"


def get_gene_name_column(df):
    if "name_gene" in df.columns:
        return "name_gene"
    if "gene_name" in df.columns:
        return "gene_name"
    raise ValueError("Missing gene name column: expected 'name_gene' or 'gene_name'.")


# -----------------------------
# Main pipeline
# -----------------------------
def main() -> str:
    args = parse_args()

    logger.info("Loading raw expression dataset: %s", args.input)
    df = pd.read_csv(args.input, sep="\t")

    # extract expression feature columns
    features = get_feature_columns(df, args.expression_unit)

    # extract tissues from feature names
    tissues = get_tissues(features, args.expression_unit)

    logger.info(
        "Detected %d expression columns across %d tissues.",
        len(features),
        len(tissues),
    )

    # -----------------------------
    # Generate all non-ortholog pairs
    # -----------------------------
    logger.info("Generating all non-orthologous human x mouse pairs.")
    nonortho_df = mod.all_nonortholog_pairs(
        df,
        args.dataset_name,
        OUTPUT_DIR,
        expression_unit=args.expression_unit,
    )

    # -----------------------------
    # Apply normalization
    # -----------------------------
    logger.info("Applying normalization methods to orthologous dataset.")
    normfx.apply_normalizations(
        df=df,
        df_name=args.dataset_name,
        features=features,
        orthology="Orthologs",
        output_dir=OUTPUT_DIR,
    )

    logger.info("Applying normalization methods to all non-orthologous pairs.")
    normfx.apply_normalizations(
        df=nonortho_df,
        df_name=args.dataset_name,
        features=features,
        orthology="NonOrthologs",
        output_dir=OUTPUT_DIR,
    )

    # -----------------------------
    # Load normalized datasets
    # -----------------------------
    dfs_ortho, names_ortho = normfx.load_normalized_data(
        args.dataset_name,
        "Orthologs",
        OUTPUT_DIR,
    )

    dfs_nonortho, names_nonortho = normfx.load_normalized_data(
        args.dataset_name,
        "NonOrthologs",
        OUTPUT_DIR,
    )

    # -----------------------------
    # Compute statistics
    # -----------------------------
    logger.info("Computing normalization correlation statistics.")

    stats_ortho = compst.compute_stats(
        dfs_ortho,
        names_ortho,
        features,
        tissues,
        "Orthologs",
        expression_unit=args.expression_unit,
    )

    stats_nonortho = compst.compute_stats(
        dfs_nonortho,
        names_nonortho,
        features,
        tissues,
        "NonOrthologs",
        expression_unit=args.expression_unit,
    )

    stats_all = compst.merge_and_increment(
        args.dataset_name,
        stats_ortho,
        stats_nonortho,
        OUTPUT_DIR,
    )

    # -----------------------------
    # Plot results using lib.plotting
    # -----------------------------
    logger.info("Generating normalization plots using lib.plotting.")

    ortho_dict = dict(zip(names_ortho, dfs_ortho))
    nonortho_dict = dict(zip(names_nonortho, dfs_nonortho))

    pt.plot_correlations(
        ortho_dict,
        stats_all,
        tissues,
        args.dataset_name,
        "Orthologs",
        OUTPUT_DIR,
        expression_unit=args.expression_unit,
    )

    pt.plot_correlations(
        nonortho_dict,
        stats_all,
        tissues,
        args.dataset_name,
        "NonOrthologs",
        OUTPUT_DIR,
        expression_unit=args.expression_unit,
    )

    sorted_methods = pt.sort_methods_by_correlation(
        stats_all,
        orthology="Increment",
        tissue=args.tissue,
    )

    pt.plot_heatmap(
        stats_all,
        args.dataset_name,
        sorted_methods,
        "Pearson_R",
        "Increment",
        OUTPUT_DIR,
    )

    # -----------------------------
    # Automatic method selection for Nextflow
    # -----------------------------
    best_method, ranking = select_best_method(
        stats_dataframe=stats_all,
        tissue=args.tissue,
        selection_metric=args.selection_metric,
        expression_unit=args.expression_unit,
    )

    normalization_dir = OUTPUT_DIR / args.dataset_name / "Normalization"

    ranking_path = normalization_dir / "normalization_method_ranking.tsv"
    ranking.to_csv(ranking_path, sep="\t", index=False)

    best_method_path = normalization_dir / "best_method.txt"
    best_method_path.write_text(best_method + "\n")

    manifest = {
        "input": str(args.input),
        "dataset_name": args.dataset_name,
        "tissues": tissues,
        "feature_columns": features,
        "normalization_methods": names_ortho,
        "nonortholog_strategy": "all human x mouse pairs minus true ortholog pairs",
        "selection_tissue": args.tissue,
        "selection_metric": args.selection_metric,
        "best_method": best_method,
        "plotting_module": "lib.plotting",
        "nextflow_output_policy": "stable process-local output folders are declared by the Nextflow process",
    }

    manifest_path = normalization_dir / "normalization_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    logger.info("Best normalization method: %s", best_method)

    # -----------------------------
    # Stable output staging for Nextflow
    # -----------------------------
    stage_outputs_for_nextflow(args.dataset_name)

    logger.info("Normalization outputs generated in process working directory.")

    return best_method


if __name__ == "__main__":
    main()
