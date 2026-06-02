#!/usr/bin/env python3
"""Compute correlation statistics for normalized datasets."""

expression_unit = "counts"

import logging
from pathlib import Path

import pandas as pd
from scipy.stats import pearsonr, spearmanr

logger = logging.getLogger(__name__)


# -----------------------------
# Core statistics
# -----------------------------
def compute_stats(
    dfs: list[pd.DataFrame],
    dfs_names: list[str],
    features: list[str],
    tissues: list[str],
    orthology: str,
    expression_unit: str = "tpm"
) -> pd.DataFrame:
    """
    Compute Pearson and Spearman correlations for all methods and tissues.

    Args:
        - dfs (list[pd.DataFrame]): Normalized expression dataframes
        - dfs_names (list[str]): Method names corresponding to dfs
        - features (list[str]): Expression feature columns (unused but kept for pipeline consistency)
        - tissues (list[str]): Tissue names
        - orthology (str): Orthology type (Orthologs / NonOrthologs)
        - expression_unit (str): Expression unit (e.g., "tpm", "counts")

    Returns:
        pd.DataFrame: Table with correlation statistics per tissue and method
    """

    results = []

    # iterate over normalization methods
    for df, method_name in zip(dfs, dfs_names):
        method_label = method_name

        # compute per tissue + global aggregation
        for tissue in tissues + ["General"]:

            # -----------------------------
            # extract expression vectors
            # -----------------------------
            if tissue == "General":
                human_vals = df[[f"{t}_{expression_unit}_human" for t in tissues]].values.flatten()
                mouse_vals = df[[f"{t}_{expression_unit}_mouse" for t in tissues]].values.flatten()
            else:
                human_vals = df[f"{tissue}_{expression_unit}_human"]
                mouse_vals = df[f"{tissue}_{expression_unit}_mouse"]

            # -----------------------------
            # correlation computation
            # -----------------------------
            r_p, _ = pearsonr(human_vals, mouse_vals)
            r_s, _ = spearmanr(human_vals, mouse_vals)

            results.append({
                "tissue": tissue,
                "method": method_label,
                "Pearson_R": r_p,
                "Spearman_rho": r_s,
            })

    logger.info(
        "Computed stats: %d methods × %d tissues (%s)",
        len(dfs),
        len(tissues) + 1,
        orthology,
    )

    return pd.DataFrame(results)


# -----------------------------
# Merge + increment signal
# -----------------------------
def merge_and_increment(
    df_name: str,
    stats_ortho: pd.DataFrame,
    stats_nonortho: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    """
    Merge ortholog and non-ortholog correlation stats and compute signal increment.

    Increment definition:
        increment = ortho - max(0, non-ortho)

    Args:
        - df_name (str): Dataset name
        - stats_ortho (pd.DataFrame): Ortholog correlation stats
        - stats_nonortho (pd.DataFrame): Non-ortholog correlation stats
        - output_dir (Path): Output directory

    Returns:
        pd.DataFrame: Merged dataframe with increment columns
    """

    # merge both datasets on tissue + method
    stats_all = pd.merge(
        stats_ortho,
        stats_nonortho,
        on=["tissue", "method"],
        suffixes=("_ortho", "_nonortho"),
    )

    # compute signal above noise
    stats_all["Pearson_R_increment"] = (
        stats_all["Pearson_R_ortho"] - stats_all["Pearson_R_nonortho"].clip(lower=0)
    )

    stats_all["Spearman_rho_increment"] = (
        stats_all["Spearman_rho_ortho"] - stats_all["Spearman_rho_nonortho"].clip(lower=0)
    )

    # save results
    out_path = output_dir / df_name / "Normalization" / f"{df_name}_stats.tsv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    stats_all.to_csv(out_path, sep="\t", index=False)

    logger.info("Saved merged statistics to %s", out_path)

    return stats_all




def merge_and_increment_replicates(
    df_name: str,
    stats_ortho: pd.DataFrame,
    stats_nonortho_replicates: list[pd.DataFrame],
    output_dir: Path,
) -> pd.DataFrame:
    """
    Merge ortholog statistics with the mean non-ortholog statistics across replicate shuffles.

    Increment definition:
        increment = ortho - max(0, mean(non-ortho))

    Args:
        - df_name (str): Dataset name
        - stats_ortho (pd.DataFrame): Ortholog correlation stats
        - stats_nonortho_replicates (list[pd.DataFrame]): Non-ortholog stats from replicate shuffles
        - output_dir (Path): Output directory

    Returns:
        pd.DataFrame: Merged dataframe with replicate-averaged increment columns
    """

    stats_nonortho_all = pd.concat(
        stats_nonortho_replicates,
        ignore_index=True,
    )

    stats_nonortho_summary = (
        stats_nonortho_all
        .groupby(["tissue", "method"], as_index=False)
        .agg(
            Pearson_R_nonortho_mean=("Pearson_R", "mean"),
            Pearson_R_nonortho_std=("Pearson_R", "std"),
            Spearman_rho_nonortho_mean=("Spearman_rho", "mean"),
            Spearman_rho_nonortho_std=("Spearman_rho", "std"),
            nonortholog_replicates=("replicate_seed", "nunique"),
        )
    )

    stats_all = pd.merge(
        stats_ortho,
        stats_nonortho_summary,
        on=["tissue", "method"],
    )

    stats_all = stats_all.rename(
        columns={
            "Pearson_R": "Pearson_R_ortho",
            "Spearman_rho": "Spearman_rho_ortho",
        }
    )

    # Compatibility aliases used by plotting.py for NonOrthologs plots.
    stats_all["Pearson_R_nonortho"] = stats_all["Pearson_R_nonortho_mean"]
    stats_all["Spearman_rho_nonortho"] = stats_all["Spearman_rho_nonortho_mean"]

    stats_all["Pearson_R_increment"] = (
        stats_all["Pearson_R_ortho"]
        - stats_all["Pearson_R_nonortho_mean"].clip(lower=0)
    )

    stats_all["Spearman_rho_increment"] = (
        stats_all["Spearman_rho_ortho"]
        - stats_all["Spearman_rho_nonortho_mean"].clip(lower=0)
    )

    out_dir = output_dir / df_name / "Normalization"
    out_dir.mkdir(parents=True, exist_ok=True)

    stats_all.to_csv(
        out_dir / f"{df_name}_stats.tsv",
        sep="\t",
        index=False,
    )

    stats_nonortho_all.to_csv(
        out_dir / f"{df_name}_nonortholog_replicate_stats.tsv",
        sep="\t",
        index=False,
    )

    stats_nonortho_summary.to_csv(
        out_dir / f"{df_name}_nonortholog_mean_stats.tsv",
        sep="\t",
        index=False,
    )

    logger.info("Saved replicate-averaged statistics to %s", out_dir)

    return stats_all

# -----------------------------
# Method selection
# -----------------------------
def best_method(
    df_name: str,
    tissue_criteria: str,
    output_dir: Path,
    expression_unit: str = "tpm",
    ranking_column: str = "Pearson_R_increment",
) -> str:
    """Select best normalization method based on increment statistics."""

    stats_path = output_dir / df_name / "Normalization" / f"{df_name}_stats.tsv"
    df = pd.read_csv(stats_path, sep="\t")

    df = df[df["tissue"] == tissue_criteria].copy()

    if df.empty:
        raise ValueError(f"No stats found for tissue: {tissue_criteria}")

    if ranking_column not in df.columns:
        raise ValueError(f"Missing ranking column: {ranking_column}")

    df = df.sort_values(ranking_column, ascending=False).reset_index(drop=True)

    print(f"\nTop 5 methods by {ranking_column}:")
    print(df.head(5)[["method", ranking_column]].to_string(index=False))

    selected_method = str(df.loc[0, "method"])

    if expression_unit == "counts" and selected_method == "original":
        if len(df) < 2:
            raise ValueError(
                "Best method is 'original' for raw counts, but no second method is available."
            )

        logger.warning(
            "Best method was 'original' with raw counts. Selecting second-best method instead."
        )

        selected_method = str(df.loc[1, "method"])

    logger.info("Selected normalization method: %s", selected_method)

    return selected_method