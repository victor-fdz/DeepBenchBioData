#!/usr/bin/env python3
"""Functions to generate non-orthologous and all-pairs gene expression datasets."""

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

# reproducibility
np.random.seed(0)


# -----------------------------
# Core: non-orthologous pairs
# -----------------------------
def pairs_exchange(
    df: pd.DataFrame,
    df_name: str,
    output_dir: Path,
    _retries: int = 0,
) -> pd.DataFrame:
    """
    Generate non-orthologous gene pairs by independently shuffling human and mouse samples.

    This function breaks gene correspondence between species by:
    - shuffling human rows
    - shuffling mouse rows
    - recombining both datasets

    A validation step ensures that residual matching gene IDs are below a threshold.

    Args:
        - df (pd.DataFrame): Input expression dataframe with human/mouse columns
        - df_name (str): Dataset name (used for output naming)
        - output_dir (Path): Root output directory
        - _retries (int): Internal retry counter (do not set manually)

    Returns:
        pd.DataFrame: Shuffled non-orthologous dataset

    Raises:
        RuntimeError: If valid shuffle cannot be achieved within retry limit
    """

    # stop recursion if shuffle fails repeatedly
    if _retries >= MAX_SHUFFLE_RETRIES:
        raise RuntimeError(
            f"Could not generate non-orthologous pairs after {MAX_SHUFFLE_RETRIES} attempts."
        )

    # separate human and mouse expression blocks
    human_cols = [c for c in df.columns if c.endswith("_human")] + ["gene_name"]
    mouse_cols = [c for c in df.columns if c.endswith("_mouse")] + ["gene_name"]

    human = df[human_cols].rename(columns={"gene_name": "gene_name_human"})
    mouse = df[mouse_cols].rename(columns={"gene_name": "gene_name_mouse"})

    # shuffle independently (break orthology structure)
    shuffled_human = human.sample(frac=1).reset_index(drop=True)
    shuffled_mouse = mouse.sample(frac=1).reset_index(drop=True)

    # split expression from identifiers
    expr_human = shuffled_human.drop(columns=["gene_name_human", "gene_id_human"])
    expr_mouse = shuffled_mouse.drop(columns=["gene_name_mouse", "gene_id_mouse"])

    # recombine shuffled components
    shuffled_df = (
        shuffled_human[["gene_id_human"]]
        .join(shuffled_mouse[["gene_id_mouse"]])
        .join(shuffled_human[["gene_name_human"]])
        .join(shuffled_mouse[["gene_name_mouse"]])
        .join(expr_human)
        .join(expr_mouse)
    )

    # validate remaining accidental (low by chance) true pairs
    leak = (shuffled_df["gene_id_human"] == shuffled_df["gene_id_mouse"]).mean()

    if leak > ORTHOLOGY_LEAK_THRESHOLD:
        logger.warning(
            "Orthology leak detected (%.2f%%). Retrying shuffle.",
            leak * 100,
        )
        return pairs_exchange(df, df_name, output_dir, _retries=_retries + 1)

    # save output
    out_path = output_dir / df_name / "Intermediate_Datasets" / f"{df_name}_nonOrthologs.tsv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    shuffled_df.to_csv(out_path, sep="\t", index=False)

    logger.info("Non-orthologous dataset saved to %s", out_path)

    return shuffled_df


# -----------------------------
# Core: all pairs generation
# -----------------------------
def all_pairs(df_path: Path, df_name: str, output_dir: Path) -> pd.DataFrame:
    """
    Generate all possible human × mouse gene combinations (cross join).

    This produces a full Cartesian product between human and mouse genes.

    Args:
        - df_path (Path): Input TSV file path
        - df_name (str): Dataset name (used for output naming)
        - output_dir (Path): Root output directory

    Returns:
        pd.DataFrame: Cross-product dataset of all gene pairs
    """

    # load input data
    df = pd.read_csv(df_path, sep="\t")

    # split human and mouse columns
    human_cols = [c for c in df.columns if "human" in c] + ["gene_name"]
    mouse_cols = [c for c in df.columns if "mouse" in c] + ["gene_name"]

    human = df[human_cols].rename(columns={"gene_name": "gene_name_human"})
    mouse = df[mouse_cols].rename(columns={"gene_name": "gene_name_mouse"})

    # Cartesian product (all-vs-all pairing)
    all_pairs_df = human.merge(mouse, how="cross")

    # enforce column order (IDs → names → human TPM → mouse TPM)
    meta_human = ["gene_id_human", "gene_name_human"]
    meta_mouse = ["gene_id_mouse", "gene_name_mouse"]

    tpm_human = human.drop(columns=meta_human).columns.tolist()
    tpm_mouse = mouse.drop(columns=meta_mouse).columns.tolist()

    all_pairs_df = (
        meta_human[:1]
        + meta_mouse[:1]
        + meta_human[1:]
        + meta_mouse[1:]
        + tpm_human
        + tpm_mouse
    )

    all_pairs_df = human.merge(mouse, how="cross")[all_pairs_df]

    # save output
    out_path = output_dir / df_name / "Intermediate_Datasets" / f"{df_name}_allPairs.tsv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_pairs_df.to_csv(out_path, sep="\t", index=False)

    logger.info("All-pairs dataset saved to %s", out_path)

    return all_pairs_df


# -----------------------------
# Core: same-species pair generation
# -----------------------------
DEFAULT_MAX_SAME_SPECIES_PAIRS = 1_000_000

def same_species_pairs(
    df: pd.DataFrame,
    species: str,
    df_name: str,
    output_dir: Path,
    include_self_pairs: bool = False,
    max_pairs: int = DEFAULT_MAX_SAME_SPECIES_PAIRS,
) -> pd.DataFrame:
    """
    Generate all unique within-species gene pairs.

    The output keeps species-neutral branch identifiers:
        gene_id_1, gene_id_2, gene_name_1, gene_name_2

    For compatibility with existing profiling functions, expression columns are
    still mapped to the two branch slots used by the cross-species code:
        <tissue>_tpm_human = branch 1
        <tissue>_tpm_mouse = branch 2

    The real species is stored in source_species.
    """

    if species not in {"human", "mouse"}:
        raise ValueError("species must be either 'human' or 'mouse'.")

    species_columns = [c for c in df.columns if c.endswith(f"_{species}")] + ["gene_name"]
    entries = df[species_columns].copy()
    entries = entries.rename(columns={"gene_name": f"gene_name_{species}"})

    gene_id_column = f"gene_id_{species}"
    if gene_id_column not in entries.columns:
        raise KeyError(f"Missing required column: {gene_id_column}")

    entries = entries.drop_duplicates(subset=[gene_id_column]).reset_index(drop=True)

    n_entries = len(entries)
    if include_self_pairs:
        candidate_pairs = n_entries * (n_entries + 1) // 2
    else:
        candidate_pairs = n_entries * (n_entries - 1) // 2

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
    keep = pairs[first_gene].le(pairs[second_gene]) if include_self_pairs else pairs[first_gene].lt(pairs[second_gene])
    pairs = pairs[keep].reset_index(drop=True)

    canonical = _canonicalize_same_species_pairs(pairs, species)

    out_path = (
        output_dir /
        df_name /
        "Intermediate_Datasets" /
        f"{df_name}_{species}_{species}_pairs.tsv"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canonical.to_csv(out_path, sep="\t", index=False)

    logger.info("Same-species %s-%s pairs saved to %s", species, species, out_path)

    return canonical


def _canonicalize_same_species_pairs(
    pairs: pd.DataFrame,
    species: str,
) -> pd.DataFrame:
    """
    Convert species-specific pair columns into a species-neutral two-branch schema.

    Example for mouse-mouse:
        gene_id_mouse_1      -> gene_id_1
        gene_id_mouse_2      -> gene_id_2
        liver_tpm_mouse_1    -> liver_tpm_human  # branch 1 for profiling
        liver_tpm_mouse_2    -> liver_tpm_mouse  # branch 2 for profiling
    """

    rename_map: dict[str, str] = {}

    for col in pairs.columns:
        if col == f"gene_id_{species}_1":
            rename_map[col] = "gene_id_1"
        elif col == f"gene_id_{species}_2":
            rename_map[col] = "gene_id_2"
        elif col == f"gene_name_{species}_1":
            rename_map[col] = "gene_name_1"
        elif col == f"gene_name_{species}_2":
            rename_map[col] = "gene_name_2"
        elif col.endswith(f"_tpm_{species}_1"):
            rename_map[col] = col.replace(f"_tpm_{species}_1", "_tpm_human")
        elif col.endswith(f"_tpm_{species}_2"):
            rename_map[col] = col.replace(f"_tpm_{species}_2", "_tpm_mouse")

    out = pairs.rename(columns=rename_map)
    out["source_species"] = species

    meta = ["gene_id_1", "gene_id_2", "gene_name_1", "gene_name_2", "source_species"]
    tpm_cols = [c for c in out.columns if c not in meta]

    return out[meta + tpm_cols]
