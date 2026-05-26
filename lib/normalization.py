#!/usr/bin/env python3
"""Normalization functions for cross-species gene expression TPM data."""

import logging
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from pydeseq2.preprocessing import deseq2_norm as pydeseq2_normalize
from skbio.stats.composition import clr as clr_skbio

logger = logging.getLogger(__name__)

# -----------------------------
# Constants
# -----------------------------
NORMALIZATION_METHODS: list[Callable] = []  # filled after definitions


# -----------------------------
# Basic normalization methods
# -----------------------------
def log1p_norm(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    """
    Apply log(1 + x) transformation.

    Args:
        - df (pd.DataFrame): Input expression dataframe
        - numeric_cols (list[str]): Columns to transform

    Returns:
        pd.DataFrame: Log1p-normalized dataframe
    """

    df_out = df.copy()
    df_out[numeric_cols] = np.log1p(df_out[numeric_cols])
    return df_out


def clr_norm(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    """
    Apply centered log-ratio (CLR) normalization.

    Args:
        - df (pd.DataFrame): Input expression dataframe
        - numeric_cols (list[str]): Columns to transform

    Returns:
        pd.DataFrame: CLR-normalized dataframe
    """

    df_out = df.copy()

    # CLR is applied column-wise with pseudocount to avoid zeros
    for col in numeric_cols:
        df_out[col] = clr_skbio(df_out[col] + 1)

    return df_out


def ranking_norm(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    """
    Replace values with descending rank per column.

    Args:
        - df (pd.DataFrame): Input dataframe
        - numeric_cols (list[str]): Columns to rank

    Returns:
        pd.DataFrame: Rank-normalized dataframe
    """

    df_out = df.copy()

    # rank per column independently
    df_out[numeric_cols] = df_out[numeric_cols].rank(
        method="min",
        axis=0,
        ascending=False,
    )

    return df_out


def quantile_norm(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    """
    Apply mathematically standard quantile normalization.

    * Special thanks to Matteo Zambon.

    Args:
        - df (pd.DataFrame): Input dataframe
        - numeric_cols (list[str]): Columns to normalize

    Returns:
        pd.DataFrame: Quantile-normalized dataframe
    """

    df_out = df.copy()

    # Step 1: log transform
    values = np.log1p(df_out[numeric_cols].values.astype(float))

    # Step 2: sort each column
    sorted_vals = np.sort(values, axis=0)

    # Step 3: mean across rows (rank means)
    rank_means = sorted_vals.mean(axis=1)

    # Step 4: get ranks per column
    ranks = values.argsort(axis=0).argsort(axis=0)

    # Step 5: map ranks → mean values (fully vectorized)
    df_out[numeric_cols] = rank_means[ranks]

    return df_out


def quantile_sample_norm(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    """
    Apply quantile normalization per tissue.

    * Inspired in Zhao et al. (2020). DOI:10.1093/bib/bbt051

    Args:
        - df (pd.DataFrame): Input dataframe
        - numeric_cols (list[str]): Columns to normalize

    Returns:
        pd.DataFrame: Tissue-wise quantile-normalized dataframe
    """

    df_out = df.copy()

    # extract tissues from column names
    tissues = list({c.rsplit("_tpm_", 1)[0] for c in numeric_cols if "_tpm_" in c})

    for tissue in tissues:
        cols = [f"{tissue}_tpm_human", f"{tissue}_tpm_mouse"]

        # reuse quantile_norm on subset
        df_out[cols] = quantile_norm(df_out, cols)[cols]

    return df_out


def deseq2_norm(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    """
    Apply DESeq2 median-of-ratios normalization.

    Args:
        - df (pd.DataFrame): Input dataframe (genes × samples)
        - numeric_cols (list[str]): Sample columns

    Returns:
        pd.DataFrame: DESeq2-normalized dataframe (log10 scaled)
    """

    df_out = df.copy()

    # PyDESeq2 expects samples × genes (so we transpose)
    counts = np.round(df_out[numeric_cols].values.T).astype(int)

    norm_counts, _ = pydeseq2_normalize(counts)

    # back to genes × samples
    df_out[numeric_cols] = np.log10(norm_counts.T + 1)

    return df_out


# -----------------------------
# TMM normalization (edgeR-like)
# -----------------------------
def _calculate_tmm_factors(
    counts: np.ndarray,
    logratio_trim: float = 0.3,
    sum_trim: float = 0.05,
) -> np.ndarray:
    """
    Compute TMM normalization factors.

    Args:
        - counts (np.ndarray): Raw count matrix
        - logratio_trim (float): trimming threshold (M-values)
        - sum_trim (float): trimming threshold (A-values)

    Returns:
        np.ndarray: Normalization factors per sample
    """

    n_samples = counts.shape[1]
    norm_factors = np.ones(n_samples)

    lib_size = counts.sum(axis=0)

    # fallback if invalid library sizes
    if np.any(lib_size == 0):
        return norm_factors

    # reference sample = closest to mean library size
    ref_idx = np.argmin(np.abs(lib_size - np.mean(lib_size)))

    for i in range(n_samples):
        if i == ref_idx:
            continue

        obs = counts[:, i]
        ref = counts[:, ref_idx]

        # avoid zeros
        valid = (obs > 0) & (ref > 0)
        if not np.any(valid):
            continue

        obs_v = obs[valid]
        ref_v = ref[valid]

        n_i = lib_size[i]
        n_r = lib_size[ref_idx]

        # M and A values
        m = np.log2((obs_v / n_i) / (ref_v / n_r))
        a = 0.5 * np.log2((obs_v / n_i) * (ref_v / n_r))

        # trimming
        m_lo, m_hi = np.percentile(m, [logratio_trim * 100, (1 - logratio_trim) * 100])
        a_lo, a_hi = np.percentile(a, [sum_trim * 100, (1 - sum_trim) * 100])

        mask = (m >= m_lo) & (m <= m_hi) & (a >= a_lo) & (a <= a_hi)

        if np.any(mask):
            weights = (n_i - obs_v) / (n_i * obs_v) + (n_r - ref_v) / (n_r * ref_v)
            norm_factors[i] = 2 ** (np.sum(weights[mask] * m[mask]) / np.sum(weights[mask]))

    # scale factors
    norm_factors /= np.exp(np.mean(np.log(norm_factors)))
    return norm_factors


def edger_tmm_norm(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    """
    Apply TMM normalization (edgeR-style approximation).

    Args:
        - df (pd.DataFrame): Input dataframe
        - numeric_cols (list[str]): Columns to normalize

    Returns:
        pd.DataFrame: TMM-normalized dataframe (log10 CPM)
    """

    df_out = df.copy()

    counts = np.round(df_out[numeric_cols].values).astype(int)

    factors = _calculate_tmm_factors(counts)

    lib_size = counts.sum(axis=0)
    effective_lib = lib_size * factors

    cpm = (counts / effective_lib) * 1e6

    df_out[numeric_cols] = np.log10(cpm + 1)

    return df_out

def invented_norm(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    """
    Placeholder for an invented normalization method.

    This is a stub function to demonstrate how to add new methods to the pipeline.

    Args:
        - df (pd.DataFrame): Input dataframe
        - numeric_cols (list[str]): Columns to normalize    
    Returns:
        pd.DataFrame: Normalized dataframe
    """
    df_out = df.copy()

    # Example: divide each value of the gene by the mean of the row (gene-wise mean normalization)
    row_means = df_out[numeric_cols].mean(axis=1)
    df_out[numeric_cols] = df_out[numeric_cols].div(row_means, axis=0)
    return df_out


def no_norm(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    """
    No normalization (identity function).

    Args:
        - df (pd.DataFrame): Input dataframe
        - numeric_cols (list[str]): Columns to "normalize" (ignored)

    Returns:
        pd.DataFrame: Unchanged dataframe
    """
    return df.copy()

# -----------------------------
# Method registry
# -----------------------------
NORMALIZATION_METHODS = [
    log1p_norm,
    clr_norm,
    ranking_norm,
    quantile_norm,
    quantile_sample_norm,
    deseq2_norm,
    edger_tmm_norm,
    invented_norm,
    no_norm
]


# -----------------------------
# Pipeline execution
# -----------------------------
def apply_normalizations(
    df: pd.DataFrame,
    df_name: str,
    features: list[str],
    orthology: str,
    output_dir: Path,
    methods: list[Callable] = NORMALIZATION_METHODS,
) -> None:
    """
    Apply all normalization methods and save outputs.

    Args:
        - df (pd.DataFrame): Input dataframe
        - df_name (str): Dataset name
        - features (list[str]): Feature columns
        - orthology (str): Orthology group
        - output_dir (Path): Output root
        - methods (list[Callable]): Normalization functions
    """

    out_dir = output_dir / df_name / orthology
    out_dir.mkdir(parents=True, exist_ok=True)

    # save raw input
    df.to_csv(out_dir / f"{orthology}_{df_name}_original.tsv", sep="\t", index=False)

    # apply each normalization
    for method in methods:
        df_norm = method(df, numeric_cols=features)

        df_norm.to_csv(
            out_dir / f"{orthology}_{df_name}_{method.__name__}.tsv",
            sep="\t",
            index=False,
        )

    logger.info("Saved all normalization outputs for %s / %s", df_name, orthology)


def load_normalized_data(
    df_name: str,
    orthology: str,
    output_dir: Path,
) -> tuple[list[pd.DataFrame], list[str]]:
    """
    Load all normalized datasets from disk.

    Args:
        - df_name (str): Dataset name
        - orthology (str): Orthology group
        - output_dir (Path): Output root

    Returns:
        tuple[list[pd.DataFrame], list[str]]: DataFrames and method names
    """

    out_dir = output_dir / df_name / orthology

    method_names = ["original"] + [m.__name__ for m in NORMALIZATION_METHODS]

    dfs, names = [], []

    for name in method_names:
        path = out_dir / f"{orthology}_{df_name}_{name}.tsv"
        dfs.append(pd.read_csv(path, sep="\t"))
        names.append(name)

    logger.info("Loaded %d normalization sets for %s / %s", len(dfs), df_name, orthology)

    return dfs, names