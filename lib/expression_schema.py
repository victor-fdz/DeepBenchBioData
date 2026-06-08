#!/usr/bin/env python3
"""Shared expression-column handling for the pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from lib import tpm as tpm_functions


CANONICAL_EXPRESSION_UNIT = "expression"
VALID_INPUT_EXPRESSION_UNITS = {"tpm", "counts"}
SPECIES = ("human", "mouse")


def strip_gene_version(gene_id: object) -> str:
    """Remove Ensembl version suffix, for example ENSG000001.5 -> ENSG000001."""

    return str(gene_id).split(".", 1)[0]


def validate_input_expression_unit(input_expression_unit: str) -> None:
    if input_expression_unit not in VALID_INPUT_EXPRESSION_UNITS:
        raise ValueError(
            "input_expression_unit must be one of "
            f"{sorted(VALID_INPUT_EXPRESSION_UNITS)}. "
            f"Received: {input_expression_unit!r}"
        )


def expression_column(tissue: str, species: str) -> str:
    return f"{tissue}_{CANONICAL_EXPRESSION_UNIT}_{species}"


def get_feature_columns(dataframe: pd.DataFrame) -> list[str]:
    suffixes = tuple(
        f"_{CANONICAL_EXPRESSION_UNIT}_{species}"
        for species in SPECIES
    )

    columns = [
        column
        for column in dataframe.columns
        if column.endswith(suffixes)
    ]

    if not columns:
        raise ValueError(
            "No canonical expression columns found. Expected columns like "
            "'<tissue>_expression_human' and '<tissue>_expression_mouse'."
        )

    return columns


def get_tissues(feature_columns: list[str]) -> list[str]:
    return sorted(
        {
            column.rsplit(f"_{CANONICAL_EXPRESSION_UNIT}_", 1)[0]
            for column in feature_columns
        }
    )


def rename_expression_unit_to_canonical(
    dataframe: pd.DataFrame,
    source_expression_unit: str,
    keep_original_columns: bool = False,
) -> pd.DataFrame:
    """Rename <tissue>_<source unit>_<species> to <tissue>_expression_<species>."""

    output = dataframe.copy()

    rename_map: dict[str, str] = {}

    for column in output.columns:
        for species in SPECIES:
            source_suffix = f"_{source_expression_unit}_{species}"
            target_suffix = f"_{CANONICAL_EXPRESSION_UNIT}_{species}"

            if column.endswith(source_suffix):
                rename_map[column] = column[: -len(source_suffix)] + target_suffix

    if not rename_map:
        raise ValueError(
            f"No columns found for source expression unit {source_expression_unit!r}."
        )

    duplicated_targets = [
        target
        for target in rename_map.values()
        if target in output.columns and target not in rename_map
    ]

    if duplicated_targets:
        raise ValueError(
            "Canonical expression columns already exist and would be overwritten: "
            f"{duplicated_targets}"
        )

    if keep_original_columns:
        output = output.assign(
            **{
                target_column: output[source_column]
                for source_column, target_column in rename_map.items()
            }
        )
    else:
        output = output.rename(columns=rename_map)

    return output


def prepare_expression_dataframe(
    dataframe: pd.DataFrame,
    input_expression_unit: str,
    human_gene_metadata: Path | None = None,
    mouse_gene_metadata: Path | None = None,
    convert_counts_to_tpm: bool = True,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Prepare dataframe once, then expose only canonical expression columns downstream."""

    validate_input_expression_unit(input_expression_unit)

    if input_expression_unit == "counts" and convert_counts_to_tpm:
        if human_gene_metadata is None or mouse_gene_metadata is None:
            raise ValueError(
                "human_gene_metadata and mouse_gene_metadata are required when "
                "input_expression_unit='counts' and convert_counts_to_tpm=True."
            )

        dataframe = tpm_functions.add_tpm_columns_from_counts(
            dataframe=dataframe,
            human_metadata_path=human_gene_metadata,
            mouse_metadata_path=mouse_gene_metadata,
        )

        source_expression_unit = "tpm"
    else:
        source_expression_unit = input_expression_unit

    dataframe = rename_expression_unit_to_canonical(
        dataframe=dataframe,
        source_expression_unit=source_expression_unit,
        keep_original_columns=False,
    )

    features = get_feature_columns(dataframe)
    tissues = get_tissues(features)

    return dataframe, features, tissues