#!/usr/bin/env python3
"""Expression similarity vs promoter sequence identity analysis."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from lib import modify_dataset      as mod
from lib import profiling_functions as pf

logger = logging.getLogger(__name__)


def expression_vs_sequence(
    df_path: Path,
    df_name: str,
    metric_func_name: str,
    tissues: list[str],
    promoter_path: Path,
    output_dir: Path,
) -> Path:
    """Compare expression similarity against promoter sequence identity.

    Generates all human × mouse gene pair combinations (cross join), applies
    the requested expression similarity metric, merges with promoter alignment
    data, and produces a scatter plot coloured by orthology status.

    Args:
        df_path: Path to the input TSV with raw TPM values.
        df_name: Dataset name for output file naming.
        metric_func_name: Name of the metric to compute (e.g. ``"cosine_sim"``).
        tissues: Tissue names in the dataset.
        promoter_path: Path to promoter alignment TSV with columns
            ``Gene_human``, ``Gene_mouse``, ``promoter_identity``.
        output_dir: Root output directory.

    Returns:
        Path to the saved merged dataframe TSV.
    """
    # Resolve metric function and output column name
    metric_func = next(fx for fx in pf.ALL_METRICS if fx.__name__ == metric_func_name)
    is_external = metric_func in pf.EXTERNAL_METRICS
    metric_col  = metric_func_name if is_external else f"{metric_func_name}_sim"

    # Generate all possible gene pair combinations (cross join)
    df = mod.all_pairs(df_path, df_name, output_dir)

    # Apply selected metric
    if is_external:
        df = metric_func(df, tissues)
    else:
        df = pf.calculate_internal_metric(df, metric_func, tissues)
        df = pf.calculate_internal_metric(df, pf.gini, tissues)  # Add gini for tissue-specificity context

    # Orthology flag: pairs where gene names match are true orthologs
    df["orthology_flag"] = (
        (df["gene_name_human"] == df["gene_name_mouse"])
        .map({True: "Orthologs", False: "NonOrthologs"})
    )

    # Merge with promoter sequence identity data
    df_promoter = pd.read_csv(promoter_path, sep="\t").rename(
        columns={"Gene_human": "gene_id_human", "Gene_mouse": "gene_id_mouse"}
    )
    df_merged = pd.merge(df, df_promoter, on=["gene_id_human", "gene_id_mouse"], how="inner")
    logger.info("Merged dataset: %d gene pairs retained after join.", len(df_merged))

    # Save merged dataframe
    out_dir = output_dir / df_name / "Expression_vs_Sequence"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"allPairs_{metric_func_name}.tsv"
    df_merged.to_csv(out_path, sep="\t", index=False)

    # Split by orthology type
    ortho_pairs    = df_merged[df_merged["orthology_flag"] == "Orthologs"]
    nonortho_pairs = df_merged[df_merged["orthology_flag"] == "NonOrthologs"]

    # Correlation stats
    r_p_o, _ = pearsonr (ortho_pairs   ["promoter_identity"], ortho_pairs   [metric_col])
    r_s_o, _ = spearmanr(ortho_pairs   ["promoter_identity"], ortho_pairs   [metric_col])
    r_p_n, _ = pearsonr (nonortho_pairs["promoter_identity"], nonortho_pairs[metric_col])
    r_s_n, _ = spearmanr(nonortho_pairs["promoter_identity"], nonortho_pairs[metric_col])
    info_line = f"Orthologs: Rp={round(r_p_o, 2)} Rs={round(r_s_o, 3)} | NonOrthologs: Rp={round(r_p_n, 2)} Rs={round(r_s_n, 3)}"
    logger.info(info_line)

    # Scatter plot
    plt.figure(figsize=(6.5, 3.75))
    plt.scatter(
        x=nonortho_pairs["promoter_identity"], y=nonortho_pairs[metric_col],
        c="orange", alpha=0.1, s=1, label="NonOrthologs",
    )
    plt.scatter(
        x=ortho_pairs["promoter_identity"], y=ortho_pairs[metric_col],
        c="blue", alpha=1.0, s=2, label="Orthologs",
    )
    plt.xlabel("Promoter Identity")
    plt.ylabel(metric_col)
    plt.title(f"Expression Similarity vs Promoter Sequence Identity\n{info_line}")
    plt.legend()
    plt.tight_layout()

    plot_path = out_dir /f"promoterIdentity_vs_{metric_func_name}.png"
    plt.savefig(plot_path, format="png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Scatter plot saved to %s", plot_path)

    return out_path
