#!/usr/bin/env python3
"""PyTorch datasets construction for DNA pair modeling.

This module is intentionally fail-fast:
- pair identifier columns must match one supported schema
- sequence or encoded columns must be complete and valid
- labels must be model-ready, with no silent coercion to NaN
- serialized tensor strings from TSV/CSV are rejected instead of parsed

Recommended data flow:
    TSV split files should keep raw sequence columns.
    DNAPairDatasetWithMeta encodes sequences into tensors at runtime.

Do not save torch.Tensor repr strings to TSV and expect them to be reusable.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from lib.encoding_splitting import one_hot_encode


VALID_DNA_CHARACTERS = set("ACGTN")
VALID_MODEL_LABELS = {0, 1}
VALID_RAW_LABELS = {"P", "N", "U", "0", "1", 0, 1, 0.0, 1.0}


def get_pair_id_columns(df: pd.DataFrame) -> tuple[str, str]:
    """Return the two gene identifier columns used by a pair dataframe."""

    if {"gene_id_1", "gene_id_2"}.issubset(df.columns):
        return "gene_id_1", "gene_id_2"

    if {"gene_id_human", "gene_id_mouse"}.issubset(df.columns):
        return "gene_id_human", "gene_id_mouse"

    raise ValueError(
        "Missing pair identifier columns. Expected either "
        "gene_id_1/gene_id_2 or gene_id_human/gene_id_mouse."
    )


def get_pair_sequence_columns(df: pd.DataFrame) -> tuple[str, str]:
    """Return the two raw sequence columns used by a pair dataframe."""

    if {"seq_1", "seq_2"}.issubset(df.columns):
        return "seq_1", "seq_2"

    if {"human_seq", "mouse_seq"}.issubset(df.columns):
        return "human_seq", "mouse_seq"

    raise ValueError(
        "Missing raw sequence columns. Expected either "
        "seq_1/seq_2 or human_seq/mouse_seq."
    )


def get_pair_encoded_columns(df: pd.DataFrame) -> tuple[str, str]:
    """Return the two encoded sequence columns used by a pair dataframe."""

    if {"encoded_1", "encoded_2"}.issubset(df.columns):
        return "encoded_1", "encoded_2"

    if {"human_encoded", "mouse_encoded"}.issubset(df.columns):
        return "human_encoded", "mouse_encoded"

    raise ValueError(
        "Missing encoded sequence columns. Expected either "
        "encoded_1/encoded_2 or human_encoded/mouse_encoded."
    )


def _has_pair_sequence_columns(df: pd.DataFrame) -> bool:
    return (
        {"seq_1", "seq_2"}.issubset(df.columns)
        or {"human_seq", "mouse_seq"}.issubset(df.columns)
    )


def _has_pair_encoded_columns(df: pd.DataFrame) -> bool:
    return (
        {"encoded_1", "encoded_2"}.issubset(df.columns)
        or {"human_encoded", "mouse_encoded"}.issubset(df.columns)
    )


def _format_bad_values(values: pd.Series, max_values: int = 10) -> str:
    unique_values = values.drop_duplicates().head(max_values).tolist()
    return ", ".join(repr(value) for value in unique_values)


def _validate_pair_row_id(df: pd.DataFrame) -> None:
    if "pair_row_id" not in df.columns:
        raise ValueError(
            "Missing required column pair_row_id. Add it during encoding/splitting "
            "before constructing DNAPairDatasetWithMeta."
        )

    if df["pair_row_id"].isna().any():
        missing_count = int(df["pair_row_id"].isna().sum())
        raise ValueError(f"pair_row_id contains {missing_count} missing values.")


def _validate_raw_sequences(df: pd.DataFrame, sequence_columns: tuple[str, str]) -> None:
    for column in sequence_columns:
        missing_mask = df[column].isna()
        if missing_mask.any():
            raise ValueError(
                f"Sequence column {column!r} contains {int(missing_mask.sum())} missing values. "
                "This means sequence mapping failed upstream and must be fixed there."
            )

        non_string_mask = ~df[column].map(lambda value: isinstance(value, str))
        if non_string_mask.any():
            raise TypeError(
                f"Sequence column {column!r} contains {int(non_string_mask.sum())} non-string values. "
                f"Examples: {_format_bad_values(df.loc[non_string_mask, column])}"
            )

        empty_mask = df[column].str.len() == 0
        if empty_mask.any():
            raise ValueError(
                f"Sequence column {column!r} contains {int(empty_mask.sum())} empty sequences."
            )

        invalid_mask = ~df[column].str.upper().map(lambda sequence: set(sequence).issubset(VALID_DNA_CHARACTERS))
        if invalid_mask.any():
            raise ValueError(
                f"Sequence column {column!r} contains invalid DNA characters in "
                f"{int(invalid_mask.sum())} rows. Expected only A, C, G, T, or N. "
                f"Examples: {_format_bad_values(df.loc[invalid_mask, column])}"
            )


def _tensor_from_object(value: Any, column: str, row_index: int, expected_length: int) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().clone().to(dtype=torch.float32)
    elif isinstance(value, np.ndarray):
        tensor = torch.tensor(value, dtype=torch.float32)
    elif isinstance(value, Sequence) and not isinstance(value, str):
        tensor = torch.tensor(value, dtype=torch.float32)
    elif isinstance(value, str):
        raise TypeError(
            f"Encoded column {column!r} contains a string at dataframe row {row_index}. "
            "Serialized tensor strings from TSV/CSV are not accepted. Keep raw sequence "
            "columns and let DNAPairDatasetWithMeta encode them, or save tensors in a "
            "binary format such as .pt, .pkl, or Parquet."
        )
    else:
        raise TypeError(
            f"Encoded column {column!r} contains unsupported value type "
            f"{type(value).__name__} at dataframe row {row_index}."
        )

    if tuple(tensor.shape) != (4, expected_length):
        raise ValueError(
            f"Encoded column {column!r} has tensor shape {tuple(tensor.shape)} at dataframe row {row_index}. "
            f"Expected (4, {expected_length})."
        )

    return tensor


def _validate_and_convert_encoded_columns(
    df: pd.DataFrame,
    encoded_columns: tuple[str, str],
    expected_length: int,
) -> None:
    for column in encoded_columns:
        missing_mask = df[column].isna()
        if missing_mask.any():
            raise ValueError(
                f"Encoded column {column!r} contains {int(missing_mask.sum())} missing values."
            )

        converted_values: list[torch.Tensor] = []
        for row_index, value in df[column].items():
            converted_values.append(
                _tensor_from_object(
                    value=value,
                    column=column,
                    row_index=int(row_index),
                    expected_length=expected_length,
                )
            )

        df[column] = converted_values


def _validate_and_prepare_labels(df: pd.DataFrame) -> None:
    if "label" not in df.columns:
        raise ValueError("Missing required column label.")

    missing_mask = df["label"].isna()
    if missing_mask.any():
        raise ValueError(
            f"label contains {int(missing_mask.sum())} missing values. "
            "Labels must be prepared upstream before constructing the dataset."
        )

    invalid_mask = ~df["label"].isin(VALID_RAW_LABELS)
    if invalid_mask.any():
        raise ValueError(
            "Unexpected label values detected. Expected only P, N, U, 0, or 1. "
            f"Examples: {_format_bad_values(df.loc[invalid_mask, 'label'])}"
        )

    contains_unlabeled = df["label"].isin(["U", "u"])
    if contains_unlabeled.any():
        raise ValueError(
            f"label contains {int(contains_unlabeled.sum())} U values. "
            "DNAPairDatasetWithMeta is model-ready and cannot train/evaluate on U. "
            "Filter U from train/validation and randomize or remove U in test upstream."
        )

    label_map = {"P": 1, "N": 0, "0": 0, "1": 1, 0: 0, 1: 1, 0.0: 0, 1.0: 1}
    df["label"] = df["label"].map(label_map)

    unmapped_mask = df["label"].isna()
    if unmapped_mask.any():
        raise ValueError(
            f"label contains {int(unmapped_mask.sum())} values that could not be mapped to 0/1."
        )

    df["label"] = df["label"].astype(np.float32)


class DNAPairDatasetWithMeta(Dataset):
    """Strict dataset using dataframe rows with metadata.

    Supports both schemas:
      - cross_species: gene_id_human/gene_id_mouse, human_seq/mouse_seq
      - same_species: gene_id_1/gene_id_2, seq_1/seq_2

    The preferred input is a dataframe with raw sequence columns. Encoded tensor
    columns are accepted only when they contain real in-memory tensors or arrays,
    not string representations saved to TSV/CSV.
    """

    def __init__(self, df: pd.DataFrame, sequence_length: int = 500):
        if df.empty:
            raise ValueError("Cannot create DNAPairDatasetWithMeta from an empty dataframe.")

        self.sequence_length = sequence_length
        self.df = df.reset_index(drop=True).copy()

        self.id_cols = get_pair_id_columns(self.df)
        _validate_pair_row_id(self.df)
        _validate_and_prepare_labels(self.df)

        if _has_pair_sequence_columns(self.df):
            self.sequence_cols = get_pair_sequence_columns(self.df)
            _validate_raw_sequences(self.df, self.sequence_cols)
            self.encoded_cols = self._encode_sequences_to_standard_columns()
        elif _has_pair_encoded_columns(self.df):
            self.sequence_cols = None
            self.encoded_cols = get_pair_encoded_columns(self.df)
            _validate_and_convert_encoded_columns(
                df=self.df,
                encoded_columns=self.encoded_cols,
                expected_length=self.sequence_length,
            )
        else:
            raise ValueError(
                "Missing sequence information. Provide raw sequence columns "
                "seq_1/seq_2 or human_seq/mouse_seq. Encoded columns are accepted "
                "only as real in-memory tensors or arrays, not TSV/CSV strings."
            )

    def _encode_sequences_to_standard_columns(self) -> tuple[str, str]:
        seq1, seq2 = self.sequence_cols

        if {"seq_1", "seq_2"}.issubset(self.df.columns):
            enc1, enc2 = "encoded_1", "encoded_2"
        else:
            enc1, enc2 = "human_encoded", "mouse_encoded"

        self.df[enc1] = self.df[seq1].apply(lambda sequence: one_hot_encode(sequence, self.sequence_length))
        self.df[enc2] = self.df[seq2].apply(lambda sequence: one_hot_encode(sequence, self.sequence_length))

        _validate_and_convert_encoded_columns(
            df=self.df,
            encoded_columns=(enc1, enc2),
            expected_length=self.sequence_length,
        )

        return enc1, enc2

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        enc1, enc2 = self.encoded_cols
        id1, id2 = self.id_cols

        return (
            row[enc1],
            row[enc2],
            torch.tensor(row["label"], dtype=torch.float32),
            int(row["pair_row_id"]),
            row[id1],
            row[id2],
        )
