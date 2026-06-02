#!/usr/bin/env python3
"""Functions to generate non-orthologous and all-pairs gene expression datasets.

Supports both expression column formats:

    <tissue>_tpm_<species>
    <tissue>_counts_<species>

Use expression_unit="tpm" or expression_unit="counts".
Default is "tpm" to preserve backwards compatibility.
"""

expression_unit = "counts"

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# -----------------------------
# Constants
# -----------------------------
MAX_SHUFFLE_RETRIES = 10
ORTHOLOGY_LEAK_THRESHOLD = 0.001  # tolerated fraction of remaining true pairs
DEFAULT_MAX_SAME_SPECIES_PAIRS = 1_000_000

VALID_EXPRESSION_UNITS = {"tpm", "counts"}

# reproducibility
np.random.seed(0)


# -----------------------------
# Shared helpers
# -----------------------------
def validate_expression_unit(expression_unit: str) -> None:
    """Validate expression unit name."""

    if expression_unit not in VALID_EXPRESSION_UNITS:
        raise ValueError(
            f"expression_unit must be one of {sorted(VALID_EXPRESSION_UNITS)}. "
            f"Received: {expression_unit!r}"
        )


def get_gene_name_column(df: pd.DataFrame) -> str:
    """Return the gene-name column used by the input dataframe.

    Supported names:
    - name_gene: current counts dataset
    - gene_name: previous TPM pipeline
    - Gene: auxiliary merged-table column
    """

    for column in ["name_gene", "gene_name", "Gene"]:
        if column in df.columns:
            return column

    raise ValueError(
        "Missing gene name column. Expected one of: name_gene, gene_name, Gene."
    )


def get_species_expression_columns(
    df: pd.DataFrame,
    species: str,
    expression_unit: str = "tpm",
) -> list[str]:
    """Return expression columns for one species and one unit."""

    validate_expression_unit(expression_unit)

    suffix = f"_{expression_unit}_{species}"
    columns = [column for column in df.columns if column.endswith(suffix)]

    if not columns:
        raise ValueError(
            f"No expression columns found for species={species!r}, "
            f"expression_unit={expression_unit!r}. Expected columns ending in {suffix!r}."
        )

    return columns


def build_species_block(
    df: pd.DataFrame,
    species: str,
    expression_unit: str = "tpm",
) -> pd.DataFrame:
    """Extract gene ID, gene name and expression columns for one species."""

    validate_expression_unit(expression_unit)

    gene_id_column = f"gene_id_{species}"
    gene_name_column = get_gene_name_column(df)
    expression_columns = get_species_expression_columns(
        df=df,
        species=species,
        expression_unit=expression_unit,
    )

    required_columns = [gene_id_column, gene_name_column] + expression_columns
    missing_columns = [column for column in required_columns if column not in df.columns]

    if missing_columns:
        raise ValueError(
            f"Missing required columns for {species}: {missing_columns}"
        )

    return df[required_columns].rename(
        columns={gene_name_column: f"gene_name_{species}"}
    )


def save_intermediate_dataset(
    dataframe: pd.DataFrame,
    output_dir: Path,
    df_name: str,
    filename: str,
) -> Path:
    """Save an intermediate dataset and return its path."""

    out_path = output_dir / df_name / "Intermediate_Datasets" / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dataframe.to_csv(out_path, sep="\t", index=False)

    return out_path


# -----------------------------
# Core: random non-orthologous pairs
# -----------------------------
def pairs_exchange(
    df: pd.DataFrame,
    df_name: str,
    output_dir: Path,
    expression_unit: str = "tpm",
    _retries: int = 0,
) -> pd.DataFrame:
    """
    Generate non-orthologous gene pairs by independently shuffling human and mouse samples.

    This function breaks gene correspondence between species by:
    - shuffling human rows
    - shuffling mouse rows
    - recombining both datasets

    Args:
        df: Input expression dataframe with human/mouse columns.
        df_name: Dataset name used for output naming.
        output_dir: Root output directory.
        expression_unit: Either "tpm" or "counts".
        _retries: Internal retry counter. Do not set manually.

    Returns:
        Shuffled non-orthologous dataframe.
    """

    validate_expression_unit(expression_unit)

    if _retries >= MAX_SHUFFLE_RETRIES:
        raise RuntimeError(
            f"Could not generate non-orthologous pairs after {MAX_SHUFFLE_RETRIES} attempts."
        )

    human = build_species_block(
        df=df,
        species="human",
        expression_unit=expression_unit,
    )

    mouse = build_species_block(
        df=df,
        species="mouse",
        expression_unit=expression_unit,
    )

    # Shuffle independently to break orthology structure.
    shuffled_human = human.sample(frac=1).reset_index(drop=True)
    shuffled_mouse = mouse.sample(frac=1).reset_index(drop=True)

    # Split expression from identifiers.
    expr_human = shuffled_human.drop(columns=["gene_name_human", "gene_id_human"])
    expr_mouse = shuffled_mouse.drop(columns=["gene_name_mouse", "gene_id_mouse"])

    # Recombine shuffled components.
    shuffled_df = (
        shuffled_human[["gene_id_human"]]
        .join(shuffled_mouse[["gene_id_mouse"]])
        .join(shuffled_human[["gene_name_human"]])
        .join(shuffled_mouse[["gene_name_mouse"]])
        .join(expr_human)
        .join(expr_mouse)
    )

    # Validate accidental remaining true pairs.
    leak = (shuffled_df["gene_id_human"] == shuffled_df["gene_id_mouse"]).mean()

    if leak > ORTHOLOGY_LEAK_THRESHOLD:
        logger.warning(
            "Orthology leak detected (%.2f%%). Retrying shuffle.",
            leak * 100,
        )
        return pairs_exchange(
            df=df,
            df_name=df_name,
            output_dir=output_dir,
            expression_unit=expression_unit,
            _retries=_retries + 1,
        )

    out_path = save_intermediate_dataset(
        dataframe=shuffled_df,
        output_dir=output_dir,
        df_name=df_name,
        filename=f"{df_name}_nonOrthologs.tsv",
    )

    logger.info("Non-orthologous dataset saved to %s", out_path)

    return shuffled_df


# -----------------------------
# Core: all human x mouse pairs
# -----------------------------
def all_pairs(
    df_path: Path,
    df_name: str,
    output_dir: Path,
    expression_unit: str = "tpm",
) -> pd.DataFrame:
    """
    Generate all possible human x mouse gene combinations.

    Args:
        df_path: Input TSV file path.
        df_name: Dataset name used for output naming.
        output_dir: Root output directory.
        expression_unit: Either "tpm" or "counts".

    Returns:
        Full human x mouse Cartesian product.
    """

    validate_expression_unit(expression_unit)

    df = pd.read_csv(df_path, sep="\t")

    human = build_species_block(
        df=df,
        species="human",
        expression_unit=expression_unit,
    )

    mouse = build_species_block(
        df=df,
        species="mouse",
        expression_unit=expression_unit,
    )

    all_pairs_df = human.merge(mouse, how="cross")

    meta_human = ["gene_id_human", "gene_name_human"]
    meta_mouse = ["gene_id_mouse", "gene_name_mouse"]

    expression_human = [
        column for column in human.columns if column not in meta_human
    ]

    expression_mouse = [
        column for column in mouse.columns if column not in meta_mouse
    ]

    column_order = (
        meta_human[:1]
        + meta_mouse[:1]
        + meta_human[1:]
        + meta_mouse[1:]
        + expression_human
        + expression_mouse
    )

    all_pairs_df = all_pairs_df[column_order]

    out_path = save_intermediate_dataset(
        dataframe=all_pairs_df,
        output_dir=output_dir,
        df_name=df_name,
        filename=f"{df_name}_allPairs.tsv",
    )

    logger.info("All-pairs dataset saved to %s", out_path)

    return all_pairs_df


# -----------------------------
# Core: all deterministic non-orthologous pairs
# -----------------------------
def all_nonortholog_pairs(
    df: pd.DataFrame,
    df_name: str,
    output_dir: Path,
    expression_unit: str = "tpm",
) -> pd.DataFrame:
    """
    Generate all possible human x mouse gene combinations and remove true orthologous pairs.

    True orthologs are removed using:
    - original gene_id_human / gene_id_mouse pairs from the input dataframe
    - matching gene_name_human / gene_name_mouse values, when available

    Args:
        df: Input expression dataframe with paired human/mouse ortholog rows.
        df_name: Dataset name used for output naming.
        output_dir: Root output directory.
        expression_unit: Either "tpm" or "counts".

    Returns:
        All non-orthologous human x mouse pairs.
    """

    validate_expression_unit(expression_unit)

    human = build_species_block(
        df=df,
        species="human",
        expression_unit=expression_unit,
    )

    mouse = build_species_block(
        df=df,
        species="mouse",
        expression_unit=expression_unit,
    )

    all_pairs_df = human.merge(mouse, how="cross")

    original_ortholog_pairs = set(
        zip(
            df["gene_id_human"].astype(str),
            df["gene_id_mouse"].astype(str),
        )
    )

    candidate_pairs = list(
        zip(
            all_pairs_df["gene_id_human"].astype(str),
            all_pairs_df["gene_id_mouse"].astype(str),
        )
    )

    nonortholog_mask = ~pd.Series(
        candidate_pairs,
        index=all_pairs_df.index,
    ).isin(original_ortholog_pairs)

    if {"gene_name_human", "gene_name_mouse"}.issubset(all_pairs_df.columns):
        nonortholog_mask = (
            nonortholog_mask
            & (all_pairs_df["gene_name_human"] != all_pairs_df["gene_name_mouse"])
        )

    nonortholog_df = all_pairs_df[nonortholog_mask].reset_index(drop=True)

    meta_human = ["gene_id_human", "gene_name_human"]
    meta_mouse = ["gene_id_mouse", "gene_name_mouse"]

    expression_human = [
        column for column in human.columns if column not in meta_human
    ]

    expression_mouse = [
        column for column in mouse.columns if column not in meta_mouse
    ]

    column_order = (
        meta_human[:1]
        + meta_mouse[:1]
        + meta_human[1:]
        + meta_mouse[1:]
        + expression_human
        + expression_mouse
    )

    nonortholog_df = nonortholog_df[column_order]

    out_path = save_intermediate_dataset(
        dataframe=nonortholog_df,
        output_dir=output_dir,
        df_name=df_name,
        filename=f"{df_name}_allNonOrthologs.tsv",
    )

    logger.info(
        "All non-orthologous pairs saved to %s (%d pairs).",
        out_path,
        len(nonortholog_df),
    )

    return nonortholog_df


# -----------------------------
# Core: same-species pair generation
# -----------------------------
def same_species_pairs(
    df: pd.DataFrame,
    species: str,
    df_name: str,
    output_dir: Path,
    expression_unit: str = "tpm",
    include_self_pairs: bool = False,
    max_pairs: int = DEFAULT_MAX_SAME_SPECIES_PAIRS,
) -> pd.DataFrame:
    """
    Generate all unique within-species gene pairs.

    The output keeps species-neutral branch identifiers:
        gene_id_1, gene_id_2, gene_name_1, gene_name_2

    For compatibility with existing profiling functions, expression columns are
    mapped to the two branch slots used by the cross-species code:

        <tissue>_<expression_unit>_human = branch 1
        <tissue>_<expression_unit>_mouse = branch 2

    The real species is stored in source_species.
    """

    validate_expression_unit(expression_unit)

    if species not in {"human", "mouse"}:
        raise ValueError("species must be either 'human' or 'mouse'.")

    entries = build_species_block(
        df=df,
        species=species,
        expression_unit=expression_unit,
    )

    gene_id_column = f"gene_id_{species}"

    entries = entries.drop_duplicates(subset=[gene_id_column]).reset_index(drop=True)

    candidate_pairs = len(entries) ** 2
    if candidate_pairs > max_pairs:
        raise ValueError(
            f"Refusing to build {candidate_pairs:,} candidate pairs for {species}. "
            "Increase max_pairs if this is intentional."
        )

    left = entries.add_suffix("_1")
    right = entries.add_suffix("_2")
    pairs = left.merge(right, how="cross")

    first_gene = f"{gene_id_column}_1"
    second_gene = f"{gene_id_column}_2"

    if include_self_pairs:
        keep = pairs[first_gene].le(pairs[second_gene])
    else:
        keep = pairs[first_gene].lt(pairs[second_gene])

    pairs = pairs[keep].reset_index(drop=True)

    canonical = _canonicalize_same_species_pairs(
        pairs=pairs,
        species=species,
        expression_unit=expression_unit,
    )

    out_path = save_intermediate_dataset(
        dataframe=canonical,
        output_dir=output_dir,
        df_name=df_name,
        filename=f"{df_name}_{species}_{species}_pairs.tsv",
    )

    logger.info(
        "Same-species %s-%s pairs saved to %s",
        species,
        species,
        out_path,
    )

    return canonical


def _canonicalize_same_species_pairs(
    pairs: pd.DataFrame,
    species: str,
    expression_unit: str = "tpm",
) -> pd.DataFrame:
    """
    Convert species-specific pair columns into a species-neutral two-branch schema.

    Example for mouse-mouse with expression_unit="counts":

        gene_id_mouse_1             -> gene_id_1
        gene_id_mouse_2             -> gene_id_2
        liver_counts_mouse_1        -> liver_counts_human
        liver_counts_mouse_2        -> liver_counts_mouse

    In this context, "_human" and "_mouse" mean branch 1 and branch 2,
    not biological species.
    """

    validate_expression_unit(expression_unit)

    rename_map: dict[str, str] = {}

    for column in pairs.columns:
        if column == f"gene_id_{species}_1":
            rename_map[column] = "gene_id_1"

        elif column == f"gene_id_{species}_2":
            rename_map[column] = "gene_id_2"

        elif column == f"gene_name_{species}_1":
            rename_map[column] = "gene_name_1"

        elif column == f"gene_name_{species}_2":
            rename_map[column] = "gene_name_2"

        elif column.endswith(f"_{expression_unit}_{species}_1"):
            rename_map[column] = column.replace(
                f"_{expression_unit}_{species}_1",
                f"_{expression_unit}_human",
            )

        elif column.endswith(f"_{expression_unit}_{species}_2"):
            rename_map[column] = column.replace(
                f"_{expression_unit}_{species}_2",
                f"_{expression_unit}_mouse",
            )

    out = pairs.rename(columns=rename_map)
    out["source_species"] = species

    meta = [
        "gene_id_1",
        "gene_id_2",
        "gene_name_1",
        "gene_name_2",
        "source_species",
    ]

    expression_columns = [
        column for column in out.columns if column not in meta
    ]

    return out[meta + expression_columns]
