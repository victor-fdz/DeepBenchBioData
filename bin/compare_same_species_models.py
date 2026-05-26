#!/usr/bin/env python3
"""Compare same-species expression similarity and promoter-embedding similarity.

Workflow:
1. Build all unordered human-human and mouse-mouse gene pairs.
2. Repeat for raw TPM and quantile-normalized TPM.
3. Concatenate human-human and mouse-mouse pairs.
4. Profile and label the concatenated dataframe globally.
5. Split by shared orthology keys so human and mouse test pairs are comparable.
6. Train one Siamese model on human-human pairs and one on mouse-mouse pairs.
7. Predict embedding similarities on both test sets.
8. Merge human and mouse test predictions into one final table.
9. Plot pairwise correlations among expression and embedding similarities.
"""

from __future__ import annotations

import argparse
import logging
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as torch_functional
from scipy.stats import pearsonr
from torch.utils.data import DataLoader, Dataset

from lib import labeling as labeling_functions
from lib import normalization as normalization_functions
from lib import profiling_functions
from lib.encoding_splitting import load_fasta_sequences, one_hot_encode
from lib.model.train_model import run_training

logging.basicConfig(
    level=logging.INFO,
    format="\033[1;32m%(levelname)s\033[0m | \033[1;36m%(name)s\033[0m | %(message)s",
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("results")
DEFAULT_MAX_SAME_SPECIES_PAIRS = 1_000_000

EXTERNAL_METRIC_NAMES = {
    function.__name__
    for function in profiling_functions.EXTERNAL_METRICS
}

LABEL_MAP = {
    "P": 1,
    "N": 0,
    "1": 1,
    "0": 0,
    1: 1,
    0: 0,
    1.0: 1,
    0.0: 0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--human-fasta", required=True, type=Path)
    parser.add_argument("--mouse-fasta", required=True, type=Path)
    parser.add_argument("--profiling", default="cosine_sim")
    parser.add_argument("--labeling", default="rank_labeling_random")
    parser.add_argument("--n-pos", type=int, default=5000)
    parser.add_argument("--n-neg", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--max-same-species-pairs", type=int, default=DEFAULT_MAX_SAME_SPECIES_PAIRS)
    parser.add_argument(
        "--split-mode",
        default="entity_leakage",
        choices=["entity_leakage", "leaked", "orthology_key"],
        help=(
            "entity_leakage/leaked: random pair-key split; exact pairs do not leak, "
            "but genes can appear across splits. orthology_key: no orthology key shared across splits."
        ),
    )
    parser.add_argument(
        "--evaluation-scopes",
        nargs="+",
        default=["all"],
        choices=["all", "labeled"],
        help="Which test-set scopes to evaluate for embedding-vs-expression similarity.",
    )

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)

    parser.add_argument(
        "--normalizations",
        nargs="+",
        default=["tpm", "quantile_norm"],
        choices=["tpm", "quantile_norm"],
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)

    return parser.parse_args()


def format_values(values: list[Any], max_values: int = 10) -> str:
    return ", ".join(repr(value) for value in values[:max_values])


def get_feature_columns(dataframe: pd.DataFrame) -> list[str]:
    return [
        column
        for column in dataframe.columns
        if "_tpm_" in column and column.rsplit("_", 1)[-1] in {"human", "mouse"}
    ]


def get_tissues(feature_columns: list[str]) -> list[str]:
    return sorted({column.rsplit("_", 2)[0] for column in feature_columns})


def metric_column(profiling_method: str) -> str:
    if profiling_method in EXTERNAL_METRIC_NAMES:
        return profiling_method

    return f"{profiling_method}_sim"


def validate_input_dataframe(dataframe: pd.DataFrame) -> None:
    required_columns = {"gene_id_human", "gene_id_mouse", "gene_name"}
    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        raise ValueError(f"Missing required input columns: {sorted(missing_columns)}")

    for column in sorted(required_columns):
        if dataframe[column].isna().any():
            raise ValueError(f"Input column {column!r} contains missing values.")

        if dataframe[column].duplicated().any():
            duplicated_values = dataframe.loc[
                dataframe[column].duplicated(keep=False),
                column,
            ].drop_duplicates().tolist()
            raise ValueError(
                f"Input column {column!r} must be unique. "
                f"Duplicated examples: {format_values(duplicated_values)}"
            )

    feature_columns = get_feature_columns(dataframe)
    if not feature_columns:
        raise ValueError("No TPM feature columns found. Expected columns named <tissue>_tpm_<species>.")

    tissues = get_tissues(feature_columns)
    missing_expression_columns = []
    for tissue in tissues:
        for species in ["human", "mouse"]:
            column = f"{tissue}_tpm_{species}"
            if column not in dataframe.columns:
                missing_expression_columns.append(column)

    if missing_expression_columns:
        raise ValueError(
            "Human and mouse TPM columns must be paired by tissue. "
            f"Missing columns: {missing_expression_columns}"
        )


def add_orthology_keys(dataframe: pd.DataFrame) -> pd.DataFrame:
    validate_input_dataframe(dataframe)

    dataframe = dataframe.copy().reset_index(drop=True)
    dataframe["orthology_key"] = dataframe["gene_name"].astype(str)

    return dataframe


def build_gene_key_map(dataframe: pd.DataFrame) -> dict[str, str]:
    gene_key_map: dict[str, str] = {}

    for _, row in dataframe.iterrows():
        orthology_key = str(row["orthology_key"])
        human_gene_id = str(row["gene_id_human"])
        mouse_gene_id = str(row["gene_id_mouse"])

        gene_key_map[human_gene_id] = orthology_key
        gene_key_map[mouse_gene_id] = orthology_key

    return gene_key_map


def normalize_dataframe(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
    normalization_name: str,
) -> pd.DataFrame:
    if normalization_name == "tpm":
        return dataframe.copy()

    if normalization_name == "quantile_norm":
        return normalization_functions.quantile_norm(
            dataframe.copy(),
            numeric_cols=feature_columns,
        )

    raise ValueError(f"Unsupported normalization: {normalization_name}")


def build_same_species_pairs(
    dataframe: pd.DataFrame,
    species: str,
    tissues: list[str],
    max_pairs: int,
) -> pd.DataFrame:
    """Build unordered within-species pairs with profiling-compatible branch columns."""

    if species not in {"human", "mouse"}:
        raise ValueError("species must be either 'human' or 'mouse'.")

    gene_id_column = f"gene_id_{species}"
    expression_columns = [f"{tissue}_tpm_{species}" for tissue in tissues]

    required_columns = {gene_id_column, "gene_name", "orthology_key", *expression_columns}
    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        raise ValueError(f"Missing columns for {species} same-species pairs: {sorted(missing_columns)}")

    if dataframe[gene_id_column].duplicated().any():
        raise ValueError(f"{gene_id_column} contains duplicated values.")

    n_genes = len(dataframe)
    candidate_pairs = n_genes * (n_genes - 1) // 2

    if candidate_pairs == 0:
        raise ValueError(f"Need at least two {species} genes to build same-species pairs.")

    if candidate_pairs > max_pairs:
        raise ValueError(
            f"Refusing to build {candidate_pairs:,} {species}-{species} pairs. "
            "Increase --max-same-species-pairs if this is intentional."
        )

    base_columns = [gene_id_column, "gene_name", "orthology_key", *expression_columns]
    base_dataframe = dataframe[base_columns].reset_index(drop=True).copy()

    rows = []
    for first_index, second_index in combinations(base_dataframe.index, 2):
        first_row = base_dataframe.loc[first_index]
        second_row = base_dataframe.loc[second_index]

        first_key = str(first_row["orthology_key"])
        second_key = str(second_row["orthology_key"])
        pair_key = "__".join(sorted([first_key, second_key]))

        output_row = {
            "source_species": species,
            "gene_id_1": first_row[gene_id_column],
            "gene_id_2": second_row[gene_id_column],
            "gene_name_1": first_row["gene_name"],
            "gene_name_2": second_row["gene_name"],
            "orthology_key_1": first_key,
            "orthology_key_2": second_key,
            "pair_key": pair_key,
            "pair_name": f"{first_row['gene_name']}__{second_row['gene_name']}",
        }

        for tissue in tissues:
            output_row[f"{tissue}_tpm_human"] = first_row[f"{tissue}_tpm_{species}"]
            output_row[f"{tissue}_tpm_mouse"] = second_row[f"{tissue}_tpm_{species}"]

        rows.append(output_row)

    return pd.DataFrame(rows)


def profile_and_label_pairs(
    pairs_dataframe: pd.DataFrame,
    profiling_method: str,
    labeling_method: str,
    tissues: list[str],
    n_pos: int,
    n_neg: int,
    seed: int,
) -> pd.DataFrame:
    profiled_dataframe = profiling_functions.apply_metric(
        df=pairs_dataframe,
        metric_name=profiling_method,
        tissues=tissues,
    )

    methods = {
        function.__name__: function
        for function in labeling_functions.LABELING_FUNCTIONS
    }

    if labeling_method not in methods:
        raise ValueError(f"Unknown labeling method: {labeling_method}")

    labeled_dataframe, _ = methods[labeling_method](
        df=profiled_dataframe,
        profiling_method=profiling_method,
        n_pos=n_pos,
        n_neg=n_neg,
        random_state=seed,
    )

    return labeled_dataframe


def split_orthology_keys(
    orthology_keys: pd.Series,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> tuple[set[str], set[str], set[str]]:
    if not 0 < val_frac < 1:
        raise ValueError("val_frac must be between 0 and 1.")

    if not 0 < test_frac < 1:
        raise ValueError("test_frac must be between 0 and 1.")

    if val_frac + test_frac >= 1:
        raise ValueError("val_frac + test_frac must be below 1.")

    keys = pd.Series(orthology_keys.dropna().astype(str).unique())
    n_keys = len(keys)

    n_val = max(1, round(n_keys * val_frac))
    n_test = max(1, round(n_keys * test_frac))
    n_train = n_keys - n_val - n_test

    if n_train < 2 or n_val < 2 or n_test < 2:
        raise ValueError(
            "Not enough orthology keys to create non-empty same-species train, validation, and test pairs. "
            f"Observed keys: {n_keys}. Split sizes: train={n_train}, val={n_val}, test={n_test}."
        )

    train_keys = keys.sample(n=n_train, random_state=seed)
    remaining_keys = keys[~keys.isin(train_keys)]
    validation_keys = remaining_keys.sample(n=n_val, random_state=seed)
    test_keys = remaining_keys[~remaining_keys.isin(validation_keys)]

    return set(train_keys), set(validation_keys), set(test_keys)




def split_pair_keys(
    labeled_dataframe: pd.DataFrame,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> tuple[set[str], set[str], set[str]]:
    """Split by exact pair key.

    This gives entity leakage but prevents exact-pair leakage:
    genes can appear across train/validation/test, but a pair_key can appear
    in only one split. Human and mouse versions of the same pair_key remain in
    the same split so the final test comparison can be merged.
    """

    if not 0 < val_frac < 1:
        raise ValueError("val_frac must be between 0 and 1.")

    if not 0 < test_frac < 1:
        raise ValueError("test_frac must be between 0 and 1.")

    if val_frac + test_frac >= 1:
        raise ValueError("val_frac + test_frac must be below 1.")

    if "pair_key" not in labeled_dataframe.columns:
        raise ValueError("Cannot split by pair_key because pair_key column is missing.")

    pair_keys = pd.Series(labeled_dataframe["pair_key"].dropna().astype(str).unique())
    n_pair_keys = len(pair_keys)

    n_validation = max(1, round(n_pair_keys * val_frac))
    n_test = max(1, round(n_pair_keys * test_frac))
    n_train = n_pair_keys - n_validation - n_test

    if n_train <= 0:
        raise ValueError(
            "Not enough pair keys to create train/validation/test split. "
            f"Observed pair keys: {n_pair_keys}. Split sizes: "
            f"train={n_train}, validation={n_validation}, test={n_test}."
        )

    train_pair_keys = pair_keys.sample(n=n_train, random_state=seed)
    remaining_pair_keys = pair_keys[~pair_keys.isin(train_pair_keys)]
    validation_pair_keys = remaining_pair_keys.sample(n=n_validation, random_state=seed)
    test_pair_keys = remaining_pair_keys[~remaining_pair_keys.isin(validation_pair_keys)]

    return set(train_pair_keys), set(validation_pair_keys), set(test_pair_keys)


def validate_pair_key_splits(
    train_pair_keys: set[str],
    validation_pair_keys: set[str],
    test_pair_keys: set[str],
) -> None:
    overlaps = {
        "train_validation": train_pair_keys & validation_pair_keys,
        "train_test": train_pair_keys & test_pair_keys,
        "validation_test": validation_pair_keys & test_pair_keys,
    }

    non_empty_overlaps = {
        name: values
        for name, values in overlaps.items()
        if values
    }

    if non_empty_overlaps:
        overlap_summary = {
            name: sorted(values)[:10]
            for name, values in non_empty_overlaps.items()
        }
        raise ValueError(f"Exact pair leakage detected across splits: {overlap_summary}")


def subset_by_pair_keys(
    dataframe: pd.DataFrame,
    pair_keys: set[str],
    labeled_only: bool,
    split_name: str,
    species: str,
) -> pd.DataFrame:
    context = f"{species} {split_name}"

    required_columns = {"pair_key", "label"}
    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        raise ValueError(f"{context} is missing columns: {sorted(missing_columns)}")

    result = dataframe[dataframe["pair_key"].astype(str).isin(pair_keys)].copy()

    if result.empty:
        raise ValueError(f"{context} split has zero pairs.")

    validate_labels(result, context)

    result["original_label"] = result["label"]

    if labeled_only:
        result = result[result["label"].isin(LABEL_MAP)].copy()

        if result.empty:
            raise ValueError(f"{context} split has no labeled positive or negative pairs.")

        result["label"] = result["label"].map(LABEL_MAP)

        if result["label"].isna().any():
            raise ValueError(f"{context} label conversion failed.")

        result["label"] = result["label"].astype(int)

        observed_labels = set(result["label"].unique())
        if observed_labels != {0, 1}:
            raise ValueError(
                f"{context} must contain both labels 0 and 1. "
                f"Observed labels: {sorted(observed_labels)}"
            )

    result["pair_row_id"] = np.arange(len(result), dtype=int)

    return result


def write_split_summary(
    output_path: Path,
    train_dataframe: pd.DataFrame,
    validation_dataframe: pd.DataFrame,
    test_dataframe: pd.DataFrame,
    split_mode: str,
    species: str,
) -> None:
    rows = []

    for split_name, split_dataframe in [
        ("train", train_dataframe),
        ("validation", validation_dataframe),
        ("test", test_dataframe),
    ]:
        rows.append(
            {
                "split_mode": split_mode,
                "species": species,
                "split": split_name,
                "n_rows": len(split_dataframe),
                "n_pair_keys": split_dataframe["pair_key"].nunique(),
                "n_gene_1": split_dataframe["gene_id_1"].nunique(),
                "n_gene_2": split_dataframe["gene_id_2"].nunique(),
                "label_counts": dict(split_dataframe["original_label"].value_counts())
                if "original_label" in split_dataframe.columns
                else dict(split_dataframe["label"].value_counts()),
            }
        )

    pd.DataFrame(rows).to_csv(output_path, sep="\t", index=False)

def validate_labels(dataframe: pd.DataFrame, context: str) -> None:
    if "label" not in dataframe.columns:
        raise ValueError(f"{context} is missing label column.")

    invalid_values = [
        value
        for value in dataframe["label"].dropna().unique().tolist()
        if value not in set(LABEL_MAP) | {"U"}
    ]

    if invalid_values:
        raise ValueError(f"{context} contains invalid labels: {format_values(invalid_values)}")

    if dataframe["label"].isna().any():
        raise ValueError(f"{context} contains missing labels.")


def subset_by_keys(
    dataframe: pd.DataFrame,
    keys: set[str],
    labeled_only: bool,
    split_name: str,
    species: str,
) -> pd.DataFrame:
    context = f"{species} {split_name}"

    required_columns = {"orthology_key_1", "orthology_key_2", "label"}
    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        raise ValueError(f"{context} is missing columns: {sorted(missing_columns)}")

    result = dataframe[
        dataframe["orthology_key_1"].astype(str).isin(keys)
        & dataframe["orthology_key_2"].astype(str).isin(keys)
    ].copy()

    if result.empty:
        raise ValueError(f"{context} split has zero pairs.")

    validate_labels(result, context)

    result["original_label"] = result["label"]

    if labeled_only:
        result = result[result["label"].isin(LABEL_MAP)].copy()

        if result.empty:
            raise ValueError(f"{context} split has no labeled positive or negative pairs.")

        result["label"] = result["label"].map(LABEL_MAP)

        if result["label"].isna().any():
            raise ValueError(f"{context} label conversion failed.")

        result["label"] = result["label"].astype(int)

        observed_labels = set(result["label"].unique())
        if observed_labels != {0, 1}:
            raise ValueError(
                f"{context} must contain both labels 0 and 1. "
                f"Observed labels: {sorted(observed_labels)}"
            )

    result["pair_row_id"] = np.arange(len(result), dtype=int)

    return result


def add_sequences(
    dataframe: pd.DataFrame,
    species: str,
    human_sequences: dict[str, str],
    mouse_sequences: dict[str, str],
) -> pd.DataFrame:
    dataframe = dataframe.copy()
    sequence_map = human_sequences if species == "human" else mouse_sequences

    dataframe["seq_1"] = dataframe["gene_id_1"].astype(str).map(sequence_map)
    dataframe["seq_2"] = dataframe["gene_id_2"].astype(str).map(sequence_map)

    missing_sequence_rows = dataframe[dataframe["seq_1"].isna() | dataframe["seq_2"].isna()]

    if not missing_sequence_rows.empty:
        preview_columns = ["gene_id_1", "gene_id_2", "source_species"]
        raise ValueError(
            f"Missing {species} promoter sequences for {len(missing_sequence_rows)} pairs. "
            "Example rows:\n"
            f"{missing_sequence_rows[preview_columns].head(10).to_string(index=False)}"
        )

    return dataframe


def validate_training_dataframe(
    dataframe: pd.DataFrame,
    split_name: str,
    species: str,
) -> None:
    context = f"{species} {split_name}"

    if dataframe.empty:
        raise ValueError(f"{context} dataframe is empty.")

    required_columns = {"gene_id_1", "gene_id_2", "seq_1", "seq_2", "label", "pair_row_id"}
    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        raise ValueError(f"{context} is missing columns: {sorted(missing_columns)}")

    observed_labels = set(dataframe["label"].unique())
    if observed_labels != {0, 1}:
        raise ValueError(
            f"{context} must contain both labels 0 and 1. "
            f"Observed labels: {sorted(observed_labels)}"
        )


class SameSpeciesPredictionDataset(Dataset):
    """Prediction-only dataset for same-species pairs."""

    def __init__(self, dataframe: pd.DataFrame) -> None:
        self.dataframe = dataframe.reset_index(drop=True).copy()

        required_columns = {"seq_1", "seq_2", "pair_key"}
        missing_columns = required_columns - set(self.dataframe.columns)

        if missing_columns:
            raise ValueError(f"Missing prediction columns: {sorted(missing_columns)}")

        if self.dataframe[["seq_1", "seq_2", "pair_key"]].isna().any().any():
            raise ValueError("Prediction dataframe contains missing seq_1, seq_2, or pair_key values.")

        if not self.dataframe["pair_key"].is_unique:
            duplicated_pair_keys = self.dataframe.loc[
                self.dataframe["pair_key"].duplicated(keep=False),
                "pair_key",
            ].drop_duplicates().tolist()

            raise ValueError(
                "pair_key must be unique during prediction. "
                f"Duplicated examples: {format_values(duplicated_pair_keys)}"
            )

        self.encoded_1 = [one_hot_encode(sequence) for sequence in self.dataframe["seq_1"]]
        self.encoded_2 = [one_hot_encode(sequence) for sequence in self.dataframe["seq_2"]]

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int):
        row = self.dataframe.iloc[index]

        return (
            self.encoded_1[index],
            self.encoded_2[index],
            row["pair_key"],
        )


def train_species_model(
    train_dataframe: pd.DataFrame,
    validation_dataframe: pd.DataFrame,
    output_dir: Path,
    args: argparse.Namespace,
    species: str,
) -> torch.nn.Module:
    validate_training_dataframe(train_dataframe, "train", species)
    validate_training_dataframe(validation_dataframe, "validation", species)

    return run_training(
        df_train=train_dataframe,
        df_val=validation_dataframe,
        model_output_dir=output_dir,
        batch_size=args.batch_size,
        margin=args.margin,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        dropout_rate=args.dropout,
        weight_decay=args.weight_decay,
        optimize_hparams=False,
        metric_column=metric_column(args.profiling),
    )


def predict_embedding_similarity(
    model: SiameseCNN,
    test_dataframe: pd.DataFrame,
    batch_size: int,
    expression_column: str,
) -> pd.DataFrame:
    if expression_column not in test_dataframe.columns:
        raise ValueError(f"Test dataframe is missing expression column: {expression_column}")

    loader = DataLoader(
        SameSpeciesPredictionDataset(test_dataframe),
        batch_size=batch_size,
        shuffle=False,
    )

    model.eval()
    rows = []

    with torch.no_grad():
        for sequence_1, sequence_2, pair_keys in loader:
            output_1, output_2 = model(sequence_1, sequence_2)
            similarities = torch_functional.cosine_similarity(output_1, output_2).cpu().numpy()

            for pair_key, similarity in zip(pair_keys, similarities):
                rows.append({"pair_key": pair_key, "embedding_similarity": float(similarity)})

    predictions = pd.DataFrame(rows)

    metadata_columns = [
        "pair_key",
        "pair_name",
        "gene_name_1",
        "gene_name_2",
        "orthology_key_1",
        "orthology_key_2",
        expression_column,
        "original_label",
    ]

    missing_metadata_columns = set(metadata_columns) - set(test_dataframe.columns)
    if missing_metadata_columns:
        raise ValueError(f"Test dataframe is missing metadata columns: {sorted(missing_metadata_columns)}")

    metadata = test_dataframe[metadata_columns].copy()
    metadata = metadata.rename(columns={expression_column: "expression_similarity"})

    output_dataframe = predictions.merge(metadata, on="pair_key", how="inner")

    if len(output_dataframe) != len(predictions):
        raise ValueError("Prediction metadata merge changed the number of rows.")

    return output_dataframe


def build_final_table(
    human_predictions: pd.DataFrame,
    mouse_predictions: pd.DataFrame,
    normalization_name: str,
) -> pd.DataFrame:
    human = human_predictions.rename(
        columns={
            "expression_similarity": "expression_similarity_human",
            "embedding_similarity": "embedding_similarity_human",
            "original_label": "label_human",
        }
    )
    mouse = mouse_predictions.rename(
        columns={
            "expression_similarity": "expression_similarity_mouse",
            "embedding_similarity": "embedding_similarity_mouse",
            "original_label": "label_mouse",
        }
    )

    final = human.merge(
        mouse[
            [
                "pair_key",
                "expression_similarity_mouse",
                "embedding_similarity_mouse",
                "label_mouse",
            ]
        ],
        on="pair_key",
        how="inner",
    )

    if final.empty:
        raise ValueError("Human and mouse test predictions have no shared pair_key values.")

    final["normalization"] = normalization_name
    final["label"] = np.where(
        final["label_human"].astype(str) == final["label_mouse"].astype(str),
        final["label_human"].astype(str),
        "discordant",
    )

    columns = [
        "normalization",
        "pair_key",
        "pair_name",
        "gene_name_1",
        "gene_name_2",
        "expression_similarity_human",
        "expression_similarity_mouse",
        "label",
        "label_human",
        "label_mouse",
        "embedding_similarity_human",
        "embedding_similarity_mouse",
    ]

    return final[columns].copy()


def plot_correlation_outputs(
    final_table: pd.DataFrame,
    output_dir: Path,
    normalization_name: str,
) -> None:
    if len(final_table) < 2:
        raise ValueError("Need at least two shared test pairs to compute correlations.")

    value_columns = [
        "expression_similarity_human",
        "expression_similarity_mouse",
        "embedding_similarity_human",
        "embedding_similarity_mouse",
    ]

    if final_table[value_columns].isna().any().any():
        raise ValueError("Final table contains missing values in correlation columns.")

    short_names = {
        "expression_similarity_human": "Expr H",
        "expression_similarity_mouse": "Expr M",
        "embedding_similarity_human": "Emb H",
        "embedding_similarity_mouse": "Emb M",
    }

    plot_table = final_table[value_columns].rename(columns=short_names)
    correlation_matrix = plot_table.corr(method="pearson")
    correlation_matrix.to_csv(output_dir / f"{normalization_name}_correlation_matrix.tsv", sep="	")

    plt.figure(figsize=(3.6, 3.0))
    sns.heatmap(
        correlation_matrix,
        annot=True,
        annot_kws={"fontsize": 6},
        fmt=".2f",
        cmap="vlag",
        center=0,
        square=True,
        linewidths=0.3,
        cbar_kws={"label": "Pearson r"},
    )
    plt.title(f"Correlation matrix ({normalization_name})", fontsize=8)
    plt.xticks(rotation=45, ha="right", fontsize=6)
    plt.yticks(rotation=0, fontsize=6)
    plt.tight_layout()
    plt.savefig(output_dir / f"{normalization_name}_correlation_heatmap.png", dpi=500)
    plt.close()

    grid = sns.PairGrid(plot_table, height=1.25, corner=False)
    grid.map_lower(sns.scatterplot, s=3, alpha=0.25, linewidth=0)
    grid.map_diag(sns.histplot, bins=25, edgecolor=None)

    def annotate_correlation(x, y, **kwargs):
        axis = plt.gca()
        if len(x) > 1 and len(y) > 1:
            r_value, _ = pearsonr(x, y)
            axis.annotate(
                f"r={r_value:.2f}",
                xy=(0.5, 0.5),
                xycoords="axes fraction",
                ha="center",
                va="center",
                fontsize=6,
            )
        axis.set_axis_off()

    grid.map_upper(annotate_correlation)
    grid.figure.suptitle(f"Pairwise relationships ({normalization_name})", y=1.02, fontsize=8)
    grid.figure.tight_layout()
    grid.figure.savefig(
        output_dir / f"{normalization_name}_pairwise_correlation_plot.png",
        dpi=500,
        bbox_inches="tight",
    )
    plt.close(grid.figure)


def _normalise_label_values(series: pd.Series) -> pd.Series:
    return series.replace(
        {
            "1": "P",
            1: "P",
            1.0: "P",
            "0": "N",
            0: "N",
            0.0: "N",
        }
    )


def filter_species_predictions_by_scope(
    predictions: pd.DataFrame,
    scope: str,
) -> pd.DataFrame:
    if scope == "all":
        return predictions.copy()

    if scope != "labeled":
        raise ValueError(f"Unknown evaluation scope: {scope}")

    labels = _normalise_label_values(predictions["original_label"])
    filtered = predictions[labels.isin(["P", "N"])].copy()

    if filtered.empty:
        raise ValueError("No labeled P/N rows available for species-level test evaluation.")

    return filtered


def filter_final_table_by_scope(
    final_table: pd.DataFrame,
    scope: str,
) -> pd.DataFrame:
    if scope == "all":
        return final_table.copy()

    if scope != "labeled":
        raise ValueError(f"Unknown evaluation scope: {scope}")

    label_human = _normalise_label_values(final_table["label_human"])
    label_mouse = _normalise_label_values(final_table["label_mouse"])

    filtered = final_table[
        label_human.isin(["P", "N"])
        & label_mouse.isin(["P", "N"])
    ].copy()

    if filtered.empty:
        raise ValueError("No shared test pair has labeled P/N rows in both human and mouse.")

    return filtered


def compute_embedding_expression_stats(
    dataframe: pd.DataFrame,
    expression_column: str,
    embedding_column: str,
    context: str,
) -> dict[str, float | int | str]:
    if len(dataframe) < 2:
        raise ValueError(f"Need at least two rows to evaluate {context}.")

    if dataframe[[expression_column, embedding_column]].isna().any().any():
        raise ValueError(f"Missing values found while evaluating {context}.")

    expression_values = dataframe[expression_column].to_numpy(dtype=float)
    embedding_values = dataframe[embedding_column].to_numpy(dtype=float)

    if np.unique(expression_values).size < 2:
        raise ValueError(f"Expression values are constant while evaluating {context}.")

    if np.unique(embedding_values).size < 2:
        raise ValueError(f"Embedding values are constant while evaluating {context}.")

    pearson_value, pearson_p_value = pearsonr(embedding_values, expression_values)

    return {
        "context": context,
        "n_pairs": len(dataframe),
        "pearson_r": float(pearson_value),
        "pearson_p_value": float(pearson_p_value),
        "embedding_mean": float(np.mean(embedding_values)),
        "embedding_std": float(np.std(embedding_values, ddof=1)),
        "expression_mean": float(np.mean(expression_values)),
        "expression_std": float(np.std(expression_values, ddof=1)),
    }


def plot_embedding_vs_expression_scatter(
    dataframe: pd.DataFrame,
    expression_column: str,
    embedding_column: str,
    output_path: Path,
    title: str,
    label_column: str | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats = compute_embedding_expression_stats(
        dataframe=dataframe,
        expression_column=expression_column,
        embedding_column=embedding_column,
        context=title,
    )

    plt.figure(figsize=(4.2, 3.4))

    if label_column is not None and label_column in dataframe.columns:
        labels = _normalise_label_values(dataframe[label_column])
        for label_value, label_name in [("P", "P"), ("N", "N"), ("U", "U")]:
            subset = dataframe[labels == label_value]
            if subset.empty:
                continue
            plt.scatter(
                subset[embedding_column],
                subset[expression_column],
                alpha=0.25,
                s=5,
                label=f"{label_name} (n={len(subset)})",
            )
        plt.legend(fontsize=5, frameon=False)
    else:
        plt.scatter(
            dataframe[embedding_column],
            dataframe[expression_column],
            alpha=0.25,
            s=5,
        )

    plt.xlabel("Embedding similarity", fontsize=7)
    plt.ylabel("Expression similarity", fontsize=7)
    plt.title(f"{title}\nr={stats['pearson_r']:.3f}, n={stats['n_pairs']}", fontsize=8)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=500, bbox_inches="tight")
    plt.close()


def evaluate_species_test_predictions(
    predictions: pd.DataFrame,
    species: str,
    output_dir: Path,
    normalization_name: str,
    scopes: list[str],
) -> pd.DataFrame:
    rows = []

    for scope in scopes:
        scoped_predictions = filter_species_predictions_by_scope(predictions, scope)

        stats = compute_embedding_expression_stats(
            dataframe=scoped_predictions,
            expression_column="expression_similarity",
            embedding_column="embedding_similarity",
            context=f"{normalization_name}_{species}_{scope}",
        )
        stats.update(
            {
                "normalization": normalization_name,
                "species": species,
                "scope": scope,
            }
        )
        rows.append(stats)

        scoped_predictions.to_csv(
            output_dir / f"test_embedding_vs_expression_{scope}.tsv",
            sep="\t",
            index=False,
        )

        plot_embedding_vs_expression_scatter(
            dataframe=scoped_predictions,
            expression_column="expression_similarity",
            embedding_column="embedding_similarity",
            output_path=output_dir / f"test_embedding_vs_expression_{scope}.png",
            title=f"{species} test ({normalization_name}, {scope})",
            label_column="original_label",
        )

    stats_dataframe = pd.DataFrame(rows)
    stats_dataframe.to_csv(
        output_dir / "test_embedding_vs_expression_stats.tsv",
        sep="\t",
        index=False,
    )

    return stats_dataframe


def build_final_evaluation_long_table(
    final_table: pd.DataFrame,
    scope: str,
) -> pd.DataFrame:
    scoped_table = filter_final_table_by_scope(final_table, scope)

    human = scoped_table[
        [
            "normalization",
            "pair_key",
            "pair_name",
            "expression_similarity_human",
            "embedding_similarity_human",
            "label_human",
        ]
    ].rename(
        columns={
            "expression_similarity_human": "expression_similarity",
            "embedding_similarity_human": "embedding_similarity",
            "label_human": "label",
        }
    )
    human["species"] = "human"

    mouse = scoped_table[
        [
            "normalization",
            "pair_key",
            "pair_name",
            "expression_similarity_mouse",
            "embedding_similarity_mouse",
            "label_mouse",
        ]
    ].rename(
        columns={
            "expression_similarity_mouse": "expression_similarity",
            "embedding_similarity_mouse": "embedding_similarity",
            "label_mouse": "label",
        }
    )
    mouse["species"] = "mouse"

    long_table = pd.concat([human, mouse], ignore_index=True)
    long_table["scope"] = scope

    return long_table[
        [
            "normalization",
            "scope",
            "species",
            "pair_key",
            "pair_name",
            "expression_similarity",
            "embedding_similarity",
            "label",
        ]
    ]


def evaluate_final_test_table(
    final_table: pd.DataFrame,
    output_dir: Path,
    normalization_name: str,
    scopes: list[str],
) -> pd.DataFrame:
    all_long_tables = []
    stats_rows = []

    for scope in scopes:
        long_table = build_final_evaluation_long_table(final_table, scope)
        all_long_tables.append(long_table)

        for species in ["human", "mouse"]:
            species_table = long_table[long_table["species"] == species].copy()
            stats = compute_embedding_expression_stats(
                dataframe=species_table,
                expression_column="expression_similarity",
                embedding_column="embedding_similarity",
                context=f"{normalization_name}_final_{species}_{scope}",
            )
            stats.update(
                {
                    "normalization": normalization_name,
                    "species": species,
                    "scope": scope,
                }
            )
            stats_rows.append(stats)

        combined_stats = compute_embedding_expression_stats(
            dataframe=long_table,
            expression_column="expression_similarity",
            embedding_column="embedding_similarity",
            context=f"{normalization_name}_final_combined_{scope}",
        )
        combined_stats.update(
            {
                "normalization": normalization_name,
                "species": "combined",
                "scope": scope,
            }
        )
        stats_rows.append(combined_stats)

        plot_embedding_vs_expression_scatter(
            dataframe=long_table,
            expression_column="expression_similarity",
            embedding_column="embedding_similarity",
            output_path=output_dir / f"{normalization_name}_final_embedding_vs_expression_{scope}.png",
            title=f"Final test ({normalization_name}, {scope})",
            label_column="label",
        )

    final_long_table = pd.concat(all_long_tables, ignore_index=True)
    final_long_table.to_csv(
        output_dir / f"{normalization_name}_final_embedding_vs_expression_test.tsv",
        sep="\t",
        index=False,
    )

    stats_dataframe = pd.DataFrame(stats_rows)
    stats_dataframe.to_csv(
        output_dir / f"{normalization_name}_final_embedding_vs_expression_stats.tsv",
        sep="\t",
        index=False,
    )

    return stats_dataframe

def run_one_normalization(
    raw_dataframe: pd.DataFrame,
    normalization_name: str,
    args: argparse.Namespace,
    feature_columns: list[str],
    tissues: list[str],
    human_sequences: dict[str, str],
    mouse_sequences: dict[str, str],
) -> pd.DataFrame:
    logger.info("Running normalization branch: %s", normalization_name)

    branch_dir = args.output_dir / args.name / "SameSpecies_Model_Comparison" / normalization_name
    branch_dir.mkdir(parents=True, exist_ok=True)

    normalized_dataframe = normalize_dataframe(raw_dataframe, feature_columns, normalization_name)

    human_pairs = build_same_species_pairs(
        normalized_dataframe,
        "human",
        tissues,
        max_pairs=args.max_same_species_pairs,
    )
    mouse_pairs = build_same_species_pairs(
        normalized_dataframe,
        "mouse",
        tissues,
        max_pairs=args.max_same_species_pairs,
    )

    combined_pairs = pd.concat([human_pairs, mouse_pairs], ignore_index=True)
    combined_pairs.to_csv(branch_dir / "same_species_pairs.tsv", sep="	", index=False)

    labeled_dataframe = profile_and_label_pairs(
        pairs_dataframe=combined_pairs,
        profiling_method=args.profiling,
        labeling_method=args.labeling,
        tissues=tissues,
        n_pos=args.n_pos,
        n_neg=args.n_neg,
        seed=args.seed,
    )
    labeled_dataframe.to_csv(branch_dir / "same_species_profiled_labeled.tsv", sep="	", index=False)

    if args.split_mode in {"entity_leakage", "leaked"}:
        train_split_keys, validation_split_keys, test_split_keys = split_pair_keys(
            labeled_dataframe=labeled_dataframe,
            val_frac=args.val_frac,
            test_frac=args.test_frac,
            seed=args.seed,
        )
        validate_pair_key_splits(train_split_keys, validation_split_keys, test_split_keys)
        subset_function = subset_by_pair_keys
        split_key_type = "pair_key"
    elif args.split_mode == "orthology_key":
        train_split_keys, validation_split_keys, test_split_keys = split_orthology_keys(
            raw_dataframe["orthology_key"],
            val_frac=args.val_frac,
            test_frac=args.test_frac,
            seed=args.seed,
        )
        subset_function = subset_by_keys
        split_key_type = "orthology_key"
    else:
        raise ValueError(f"Unknown split mode: {args.split_mode}")

    logger.info(
        "Split mode %s uses %s values: train=%d | validation=%d | test=%d",
        args.split_mode,
        split_key_type,
        len(train_split_keys),
        len(validation_split_keys),
        len(test_split_keys),
    )

    expression_column = metric_column(args.profiling)

    predictions = {}
    species_evaluation_stats = []

    for species in ["human", "mouse"]:
        species_dataframe = labeled_dataframe[labeled_dataframe["source_species"] == species].copy()

        train_dataframe = subset_function(
            species_dataframe,
            train_split_keys,
            labeled_only=True,
            split_name="train",
            species=species,
        )
        validation_dataframe = subset_function(
            species_dataframe,
            validation_split_keys,
            labeled_only=True,
            split_name="validation",
            species=species,
        )
        test_dataframe = subset_function(
            species_dataframe,
            test_split_keys,
            labeled_only=False,
            split_name="test",
            species=species,
        )

        train_dataframe = add_sequences(train_dataframe, species, human_sequences, mouse_sequences)
        validation_dataframe = add_sequences(validation_dataframe, species, human_sequences, mouse_sequences)
        test_dataframe = add_sequences(test_dataframe, species, human_sequences, mouse_sequences)

        split_dir = branch_dir / species / "splits"
        split_dir.mkdir(parents=True, exist_ok=True)
        train_dataframe.to_csv(split_dir / "train.tsv", sep="	", index=False)
        validation_dataframe.to_csv(split_dir / "validation.tsv", sep="	", index=False)
        test_dataframe.to_csv(split_dir / "test.tsv", sep="	", index=False)
        write_split_summary(
            output_path=split_dir / "split_summary.tsv",
            train_dataframe=train_dataframe,
            validation_dataframe=validation_dataframe,
            test_dataframe=test_dataframe,
            split_mode=args.split_mode,
            species=species,
        )

        model = train_species_model(
            train_dataframe=train_dataframe,
            validation_dataframe=validation_dataframe,
            output_dir=branch_dir / species / "Model",
            args=args,
            species=species,
        )

        model_dir = branch_dir / species / "Model"
        predictions[species] = predict_embedding_similarity(
            model=model,
            test_dataframe=test_dataframe,
            batch_size=args.batch_size,
            expression_column=expression_column,
        )
        predictions[species].to_csv(
            model_dir / "test_embedding_predictions.tsv",
            sep="	",
            index=False,
        )
        species_evaluation_stats.append(
            evaluate_species_test_predictions(
                predictions=predictions[species],
                species=species,
                output_dir=model_dir,
                normalization_name=normalization_name,
                scopes=args.evaluation_scopes,
            )
        )

    pd.concat(species_evaluation_stats, ignore_index=True).to_csv(
        branch_dir / f"{normalization_name}_species_test_embedding_vs_expression_stats.tsv",
        sep="	",
        index=False,
    )

    final_table = build_final_table(
        human_predictions=predictions["human"],
        mouse_predictions=predictions["mouse"],
        normalization_name=normalization_name,
    )

    final_table.to_csv(branch_dir / f"{normalization_name}_final_comparison_table.tsv", sep="	", index=False)
    evaluate_final_test_table(
        final_table=final_table,
        output_dir=branch_dir,
        normalization_name=normalization_name,
        scopes=args.evaluation_scopes,
    )
    plot_correlation_outputs(final_table, branch_dir, normalization_name)

    logger.info("Final table for %s has %d shared test pairs.", normalization_name, len(final_table))
    return final_table


def main() -> Path:
    args = parse_args()

    raw_dataframe = pd.read_csv(args.input, sep="	")
    raw_dataframe = add_orthology_keys(raw_dataframe)

    feature_columns = get_feature_columns(raw_dataframe)
    tissues = get_tissues(feature_columns)

    human_sequences = load_fasta_sequences(args.human_fasta)
    mouse_sequences = load_fasta_sequences(args.mouse_fasta)

    final_tables = []
    for normalization_name in args.normalizations:
        final_tables.append(
            run_one_normalization(
                raw_dataframe=raw_dataframe,
                normalization_name=normalization_name,
                args=args,
                feature_columns=feature_columns,
                tissues=tissues,
                human_sequences=human_sequences,
                mouse_sequences=mouse_sequences,
            )
        )

    all_results = pd.concat(final_tables, ignore_index=True)
    output_dir = args.output_dir / args.name / "SameSpecies_Model_Comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_path = output_dir / "all_normalizations_final_comparison_table.tsv"
    all_results.to_csv(all_path, sep="	", index=False)

    logger.info("All results saved to %s", all_path)
    return all_path


if __name__ == "__main__":
    main()
