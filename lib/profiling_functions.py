#!/usr/bin/env python3
"""Expression profiling metric functions and benchmarking for cross-species gene expression data."""

expression_unit = "counts"

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ks_2samp

logger = logging.getLogger(__name__)


# ============================================================ #
# PER-VECTOR TISSUE-SPECIFICITY METRICS (internal)             #
# Applied independently to each species; similarity is derived #
# from the absolute difference between species scores.         #
# ============================================================ #

def net(vector: np.ndarray) -> float:
    """Number of expressed tissues (expression > 1)."""
    return float((vector > 1).sum())


def tau(vector: np.ndarray) -> float:
    """Tau tissue-specificity index (0 = ubiquitous, 1 = specific).

    Source: https://github.com/apcamargo/tspex
    """
    if not np.any(vector):
        return 0.0
    n = len(vector)
    if n == 1:
        return 0.0
    vector_r = vector / np.max(vector)
    return float(np.sum(1 - vector_r) / (n - 1))


def _entropy(vector: np.ndarray) -> float:
    """Shannon entropy of an expression vector (internal helper)."""
    n = len(vector)
    if not np.any(vector):
        return np.log2(n)
    p = vector / np.sum(vector)
    p = p[p != 0]
    return float(-np.dot(p, np.log2(p)))


def shannon(vector: np.ndarray) -> float:
    """Shannon tissue-specificity index (normalized, 0–1).

    Source: https://github.com/apcamargo/tspex
    """
    if not np.any(vector):
        return 0.0
    n = len(vector)
    if n == 1:
        return 0.0
    return float((np.log2(n) - _entropy(vector)) / np.log2(n))


def gini(vector: np.ndarray) -> float:
    """Gini tissue-specificity coefficient (normalized, 0–1).

    Source: https://github.com/apcamargo/tspex
    """
    if not np.any(vector):
        return 0.0
    vector = np.sort(vector)
    n = len(vector)
    if n == 1:
        return 0.0
    index = np.arange(1, n + 1)
    gc = np.sum((2 * index - n - 1) * vector) / (n * np.sum(vector))
    return float(gc * (n / (n - 1)))


# ============================================================ #
# PAIR-LEVEL EXPRESSION SIMILARITY METRICS (external)          #
# Computed directly on (human, mouse) expression vector pairs. #
# ============================================================ #

def met(df: pd.DataFrame, tissues: list[str], expression_unit: str = "tpm") -> pd.DataFrame:
    """Matched Expressed Tissues: fraction of tissues co-expressed in both species.

    A gene is considered expressed if expression >= 1. Score range: [0, 1].

    Args:
        df: Input dataframe with expression columns.
        tissues: Tissue names in the dataset.
        expression_unit: Expression unit (e.g., "tpm", "counts").

    Returns:
        Dataframe with added ``met`` column.
    """
    h_cols = [f"{t}_{expression_unit}_human" for t in tissues]
    m_cols = [f"{t}_{expression_unit}_mouse" for t in tissues]

    expr_h = (df[h_cols].values >= 1).astype(int) * 2 - 1  # 1 if expressed, -1 otherwise
    expr_m = (df[m_cols].values >= 1).astype(int) * 2 - 1

    df["met"] = ((expr_h * expr_m) == 1).sum(axis=1) / len(tissues)
    return df


def ssd(df: pd.DataFrame, tissues: list[str], expression_unit: str = "tpm") -> pd.DataFrame:
    """Negative Sum of Squared Differences (higher = more similar profiles).

    Args:
        df: Input dataframe with expression columns.
        tissues: Tissue names in the dataset.
        expression_unit: Expression unit (e.g., "tpm", "counts").

    Returns:
        Dataframe with added ``ssd`` column.
    """
    h_cols = [f"{t}_{expression_unit}_human" for t in tissues]
    m_cols = [f"{t}_{expression_unit}_mouse" for t in tissues]

    df["ssd"] = -((df[m_cols].values - df[h_cols].values) ** 2).sum(axis=1)
    return df


def cosine_sim(df: pd.DataFrame, tissues: list[str], expression_unit: str = "tpm") -> pd.DataFrame:
    """Row-wise cosine similarity between human and mouse expression vectors.

    Args:
        df: Input dataframe with expression columns.
        tissues: Tissue names in the dataset.
        expression_unit: Expression unit (e.g., "tpm", "counts").

    Returns:
        Dataframe with added ``cosine_sim`` column.
    """
    h_cols = [f"{t}_{expression_unit}_human" for t in tissues]
    m_cols = [f"{t}_{expression_unit}_mouse" for t in tissues]

    A = df[h_cols].values
    B = df[m_cols].values

    dot  = np.sum(A * B, axis=1)
    norm = np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1)
    df["cosine_sim"] = dot / (norm + 1e-9)
    return df

# Support function
def _zscore(vector: np.ndarray) -> float:
    """Z-score tissue-specificity index (normalized, 0–1).

    Source: https://github.com/apcamargo/tspex
    """
    n = len(vector)
    if n == 1:
        return 0.0
    std = np.std(vector, ddof=1)
    if std == 0:
        return 0.0
    zs = (vector - np.mean(vector)) / std

    return zs

def z_score_cosine_sim(df: pd.DataFrame, tissues: list[str], expression_unit: str = "tpm") -> pd.DataFrame:
    """Row-wise cosine similarity between human and mouse z-score expression vectors."""

    h_cols = [f"{t}_{expression_unit}_human" for t in tissues]
    m_cols = [f"{t}_{expression_unit}_mouse" for t in tissues]

    missing_cols = sorted((set(h_cols) | set(m_cols)) - set(df.columns))
    if missing_cols:
        raise ValueError(f"Missing columns for z_score_cosine_sim: {missing_cols}")

    human_values = df[h_cols].to_numpy(dtype=float)
    mouse_values = df[m_cols].to_numpy(dtype=float)

    human_std = human_values.std(axis=1, ddof=1)
    mouse_std = mouse_values.std(axis=1, ddof=1)

    if (human_std == 0).any() or (mouse_std == 0).any():
        raise ValueError(
            "Cannot compute z_score_cosine_sim for rows with zero expression variance."
        )

    human_z = (human_values - human_values.mean(axis=1, keepdims=True)) / human_std[:, None]
    mouse_z = (mouse_values - mouse_values.mean(axis=1, keepdims=True)) / mouse_std[:, None]

    dot = np.sum(human_z * mouse_z, axis=1)
    norm = np.linalg.norm(human_z, axis=1) * np.linalg.norm(mouse_z, axis=1)

    if (norm == 0).any():
        raise ValueError("Cannot compute z_score_cosine_sim for zero-norm rows.")

    df["z_score_cosine_sim"] = dot / norm

    return df


# Canonical ordered method lists
INTERNAL_METRICS = [net, shannon, tau, gini]
EXTERNAL_METRICS = [met, ssd, cosine_sim, z_score_cosine_sim]
ALL_METRICS      = INTERNAL_METRICS + EXTERNAL_METRICS


# ============================================================ #
# METRIC APPLICATION                                           #
# ============================================================ #

def calculate_internal_metric(
    df: pd.DataFrame,
    metric_func,
    tissues: list[str],
    expression_unit: str = "tpm"
) -> pd.DataFrame:
    """Apply a per-vector internal metric to human and mouse expression profiles.

    Computes the metric for each species independently, then derives a
    similarity score: ``sim = 1 - |human_score - mouse_score|``.

    Args:
        df: Input dataframe with expression columns.
        metric_func: Per-vector metric function (e.g. ``gini``, ``tau``).
        tissues: Tissue names in the dataset.
        expression_unit: Expression unit (e.g., "tpm", "counts").
    
    Returns:
        Dataframe with added ``<metric>_human``, ``<metric>_mouse``,
        and ``<metric>_sim`` columns.
    """
    name   = metric_func.__name__
    h_cols = [f"{t}_{expression_unit}_human" for t in tissues]
    m_cols = [f"{t}_{expression_unit}_mouse" for t in tissues]

    df[f"{name}_human"] = df[h_cols].apply(lambda row: metric_func(row.values), axis=1)
    df[f"{name}_mouse"] = df[m_cols].apply(lambda row: metric_func(row.values), axis=1)
    df[f"{name}_sim"]   = 1 - (df[f"{name}_human"] - df[f"{name}_mouse"]).abs()

    return df


def apply_metrics(
    df: pd.DataFrame,
    tissues: list[str],
    internal: list = INTERNAL_METRICS,
    external: list = EXTERNAL_METRICS,
    expression_unit: str = "tpm",
) -> pd.DataFrame:
    """Apply all profiling metrics to a dataframe.

    Internal metrics are applied via ``calculate_internal_metric``.
    External metrics are applied directly, receiving the full dataframe and tissues.

    Args:
        df: Input dataframe with raw TPM values.
        tissues: Tissue names in the dataset.
        internal: Per-vector metric functions (default: INTERNAL_METRICS).
        external: Pair-level metric functions (default: EXTERNAL_METRICS).

    Returns:
        Dataframe with all metric columns added.
    """
    df_out = df.copy()
    for func in internal:
        df_out = calculate_internal_metric(df_out, func, tissues, expression_unit)
    for func in external:
        df_out = func(df_out, tissues, expression_unit)

    logger.info("Applied %d profiling metrics.", len(internal) + len(external))
    return df_out

# -----------------------------
# Single metric application
# -----------------------------
def apply_metric(
    df: pd.DataFrame,
    metric_name: str,
    tissues: list[str],
    expression_unit: str = "tpm",
) -> pd.DataFrame:
    """
    Apply a single profiling metric to dataframe.

    Args:
        - df (pd.DataFrame): Input dataframe
        - metric_name (str): Metric function name
        - tissues (list[str]): Tissue names

    Returns:
        pd.DataFrame: Profiled dataframe

    Raises:
        ValueError: If metric is unknown
    """

    # external metrics
    external = {
        fx.__name__: fx
        for fx in EXTERNAL_METRICS
    }

    # internal metrics
    internal = {
        fx.__name__: fx
        for fx in INTERNAL_METRICS
    }

    df_out = df.copy()

    # -----------------------------
    # External metric
    # -----------------------------
    if metric_name in external:

        logger.info(
            "Selected profiling metric %s is an external metric. "
            "Applying it to the data...",
            metric_name,
        )

        metric_function = external[metric_name]

        return metric_function(df_out, tissues, expression_unit)

    # -----------------------------
    # Internal metric
    # -----------------------------
    if metric_name in internal:

        logger.info(
            "Selected profiling metric %s is an internal metric. "
            "Applying it to the data...",
            metric_name,
        )

        metric_function = internal[metric_name]

        return calculate_internal_metric(
            df_out,
            metric_function,
            tissues,
            expression_unit
        )

    raise ValueError(
        f"Unknown profiling metric: {metric_name}"
    )


# ============================================================ #
# BENCHMARKING                                                 #
# ============================================================ #

def benchmark_profiling(
    df_ortho: pd.DataFrame,
    df_nonortho: pd.DataFrame,
    df_name: str,
    output_dir: Path,
) -> pd.DataFrame:
    """KS test and ECDF plots comparing orthologs vs non-orthologs per metric.

    For each metric, the KS statistic (D) quantifies how well it separates
    orthologous from non-orthologous pairs — higher D = better metric.

    Args:
        df_ortho: Metrics dataframe for orthologous pairs.
        df_nonortho: Metrics dataframe for non-orthologous pairs.
        df_name: Dataset name for output file naming.
        output_dir: Root output directory.

    Returns:
        DataFrame with KS D and p-value per metric, sorted by D descending.
    """
    logger.info("Benchmarking profiling metrics for %s.", df_name)

    metric_cols = (
        [f"{m.__name__}_sim" for m in INTERNAL_METRICS]
        + [m.__name__ for m in EXTERNAL_METRICS]
    )
    metric_cols = [c for c in metric_cols if c in df_ortho.columns and c in df_nonortho.columns]

    n = len(metric_cols)
    _, axes = plt.subplots(n,1,figsize=(6, 2.8 * n),squeeze=False,)

    ks_results = []

    for ax, col in zip(axes[:, 0], metric_cols):
        s_ortho    = df_ortho[col].dropna()
        s_nonortho = df_nonortho[col].dropna()

        d_stat, p_val = ks_2samp(s_ortho, s_nonortho)
        ks_results.append({"metric": col, "KS_D": d_stat, "p_value": p_val})

        sns.ecdfplot(
            data=s_ortho,
            label=f"Orthologs (n={len(s_ortho)})",
            color="blue",
            linewidth=2.5,
            ax=ax,
        )

        sns.ecdfplot(
            data=s_nonortho,
            label=f"NonOrthologs (n={len(s_nonortho)})",
            color="orange",
            linewidth=2.5,
            ax=ax,
        )

        # Mark maximum KS distance
        all_vals = np.sort(np.concatenate([s_ortho, s_nonortho]))

        cdf_o = (
            np.searchsorted(
                np.sort(s_ortho),
                all_vals,
                side="right",
            ) / len(s_ortho)
        )

        cdf_n = (
            np.searchsorted(
                np.sort(s_nonortho),
                all_vals,
                side="right",
            ) / len(s_nonortho)
        )

        idx = np.argmax(np.abs(cdf_o - cdf_n))

        ax.axvline(
            s_ortho.mean(),
            color="blue",
            linestyle="--",
            alpha=0.6,
            linewidth=2,
        )

        ax.axvline(
            s_nonortho.mean(),
            color="orange",
            linestyle="--",
            alpha=0.6,
            linewidth=2,
        )

        ax.vlines(
            x=all_vals[idx],
            ymin=min(cdf_o[idx], cdf_n[idx]),
            ymax=max(cdf_o[idx], cdf_n[idx]),
            color="black",
            linewidth=2.5,
            label=f"KS D={d_stat:.3f}",
        )

        # Bigger text elements for publication readability
        ax.set_title(
            f"{col}  |  KS D={d_stat:.3f}, p={p_val:.2e}",
            fontsize=14,
            fontweight="bold",
        )

        ax.set_ylabel(
            "Cumulative Proportion",
            fontsize=12,
        )

        ax.set_xlabel(
            col,
            fontsize=12,
        )

        ax.tick_params(
            axis="both",
            labelsize=11,
        )

        ax.legend(
            fontsize=10,
            frameon=False,
        )

        ax.grid(
            True,
            linestyle="--",
            alpha=0.3,
        )

    plt.tight_layout()

    out_dir = output_dir / df_name / "Benchmarking"
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / f"{df_name}_profiling_benchmark_ECDF.png", dpi=300)
    plt.close()

    ks_df = pd.DataFrame(ks_results).sort_values("KS_D", ascending=False)
    ks_df.to_csv(out_dir / f"{df_name}_profiling_KS_stats.tsv", sep="\t", index=False)
    logger.info("Benchmark results saved to %s", out_dir)

    return ks_df


def best_method(df_ortho: pd.DataFrame, df_nonortho: pd.DataFrame) -> str:
    """Return the metric with the highest KS D statistic.

    Args:
        df_ortho: Metrics dataframe for orthologous pairs.
        df_nonortho: Metrics dataframe for non-orthologous pairs.

    Returns:
        Name of the best-performing metric column.
    """
    metric_cols = (
        [f"{m.__name__}_sim" for m in INTERNAL_METRICS]
        + [m.__name__ for m in EXTERNAL_METRICS]
    )

    ks_scores = {
        col: ks_2samp(df_ortho[col].dropna(), df_nonortho[col].dropna())[0]
        for col in metric_cols
        if col in df_ortho.columns and col in df_nonortho.columns
    }

    best = max(ks_scores, key=ks_scores.get)
    logger.info("Best profiling metric: %s (KS D=%.3f)", best, ks_scores[best])
    return best
