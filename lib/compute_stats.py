#!/usr/bin/env python3
"""Compute correlation statistics for normalized datasets."""

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
) -> pd.DataFrame:
    """
    Compute Pearson and Spearman correlations for all methods and tissues.

    Args:
        - dfs (list[pd.DataFrame]): Normalized expression dataframes
        - dfs_names (list[str]): Method names corresponding to dfs
        - features (list[str]): TPM feature columns (unused but kept for pipeline consistency)
        - tissues (list[str]): Tissue names
        - orthology (str): Orthology type (Orthologs / NonOrthologs)

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
                human_vals = df[[f"{t}_tpm_human" for t in tissues]].values.flatten()
                mouse_vals = df[[f"{t}_tpm_mouse" for t in tissues]].values.flatten()
            else:
                human_vals = df[f"{tissue}_tpm_human"]
                mouse_vals = df[f"{tissue}_tpm_mouse"]

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
) -> str:
    """
    Select best normalization method based on increment statistics.

    Displays top candidates and asks user for selection.

    Args:
        - df_name (str): Dataset name
        - tissue_criteria (str): Tissue used for ranking
        - output_dir (Path): Output directory

    Returns:
        str: Selected method name
    """

    # load computed stats
    stats_path = output_dir / df_name / "Normalization" / f"{df_name}_stats.tsv"
    df = pd.read_csv(stats_path, sep="\t")

    # filter tissue of interest
    df = df[df["tissue"] == tissue_criteria]

    # rank methods
    top_pearson = df.nlargest(5, "Pearson_R_increment")[
        ["method", "Pearson_R_increment"]
    ]

    top_spearman = df.nlargest(5, "Spearman_rho_increment")[
        ["method", "Spearman_rho_increment"]
    ]

    # display ranking
    print("\nTop 5 methods by Pearson R increment:")
    print(top_pearson.to_string(index=False))

    print("\nTop 5 methods by Spearman Rho increment:")
    print(top_spearman.to_string(index=False))

    # user selection
    chosen = input("Enter method name to use:\n").strip()

    logger.info("User selected method: %s", chosen)

    return chosen