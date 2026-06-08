#!/usr/bin/env python3
"""TPM calculation from raw count columns and gene length metadata."""

from pathlib import Path

import pandas as pd


def strip_gene_version(gene_id: object) -> str:
    """Remove Ensembl version suffix, for example ENSG000001.5 -> ENSG000001."""

    return str(gene_id).split(".", 1)[0]

def load_gene_lengths(metadata_path: Path) -> pd.Series:
    """Load gene length metadata.

    Expected columns:
        gene_id
        gene_length_bp
    """

    metadata = pd.read_csv(metadata_path, sep="\t")

    required_columns = {"gene_id", "gene_length_bp"}
    missing_columns = required_columns - set(metadata.columns)

    if missing_columns:
        raise ValueError(
            f"Metadata file {metadata_path} is missing columns: {sorted(missing_columns)}"
        )

    metadata = metadata[["gene_id", "gene_length_bp"]].copy()

    metadata["gene_id"] = metadata["gene_id"].map(strip_gene_version)
    metadata["gene_length_bp"] = pd.to_numeric(
        metadata["gene_length_bp"],
        errors="coerce",
    )

    metadata = metadata.dropna(subset=["gene_id", "gene_length_bp"])
    metadata = metadata[metadata["gene_length_bp"] > 0]
    metadata = metadata.drop_duplicates("gene_id", keep="first")

    return metadata.set_index("gene_id")["gene_length_bp"]


def add_tpm_columns_for_species(
    dataframe: pd.DataFrame,
    gene_lengths_bp: pd.Series,
    species: str,
) -> pd.DataFrame:
    """Add TPM columns for one species.

    Input columns expected:
        gene_id_<species>
        <tissue>_counts_<species>

    Output columns added:
        <tissue>_tpm_<species>
    """

    output = dataframe.copy()

    gene_id_column = f"gene_id_{species}"
    count_suffix = f"_counts_{species}"
    tpm_suffix = f"_tpm_{species}"

    if gene_id_column not in output.columns:
        raise ValueError(f"Missing gene ID column: {gene_id_column}")

    count_columns = [
        column for column in output.columns
        if column.endswith(count_suffix)
    ]

    if not count_columns:
        raise ValueError(
            f"No count columns found for species={species!r}. "
            f"Expected columns ending in {count_suffix!r}."
        )

    clean_gene_ids = output[gene_id_column].map(strip_gene_version)
    lengths_bp = clean_gene_ids.map(gene_lengths_bp)

    if lengths_bp.isna().any():
        missing_examples = (
            output.loc[lengths_bp.isna(), gene_id_column]
            .drop_duplicates()
            .head(10)
            .tolist()
        )

        raise ValueError(
            f"Missing gene_length_bp for {int(lengths_bp.isna().sum())} {species} rows. "
            f"Examples: {missing_examples}"
        )

    lengths_kb = lengths_bp.astype(float) / 1000.0

    for count_column in count_columns:
        tpm_column = count_column.replace(count_suffix, tpm_suffix)

        counts = pd.to_numeric(output[count_column], errors="coerce").fillna(0)
        reads_per_kilobase = counts / lengths_kb
        scaling_factor = reads_per_kilobase.sum() / 1_000_000

        if scaling_factor <= 0:
            raise ValueError(
                f"Cannot calculate TPM for {count_column}: total RPK is zero."
            )

        output[tpm_column] = reads_per_kilobase / scaling_factor

    return output


def add_tpm_columns_from_counts(
    dataframe: pd.DataFrame,
    human_metadata_path: Path,
    mouse_metadata_path: Path,
) -> pd.DataFrame:
    """Add human and mouse TPM columns from raw count columns."""

    human_lengths_bp = load_gene_lengths(human_metadata_path)
    mouse_lengths_bp = load_gene_lengths(mouse_metadata_path)

    output = add_tpm_columns_for_species(
        dataframe=dataframe,
        gene_lengths_bp=human_lengths_bp,
        species="human",
    )

    output = add_tpm_columns_for_species(
        dataframe=output,
        gene_lengths_bp=mouse_lengths_bp,
        species="mouse",
    )

    return output