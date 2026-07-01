#!/usr/bin/env python3
"""Nextflow wrapper for labeled pair generation.

This script reuses the project labeling science modules:
- lib.normalization
- lib.modify_dataset
- lib.profiling_functions
- lib.labeling

It only changes what is required for Nextflow:
- process-local outputs
- stable output folders declared by Nextflow
- explicit command-line inputs
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib import labeling as lbl
from lib import modify_dataset as mod
from lib import normalization as normfx
from lib import profiling_functions as pf

logging.basicConfig(
    level=logging.INFO,
    format="\033[1;32m%(levelname)s\033[0m | \033[1;36m%(name)s\033[0m | %(message)s",
)

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(".")


# -----------------------------
# Helpers
# -----------------------------
def get_feature_columns(dataframe: pd.DataFrame) -> list[str]:
    """Extract TPM feature columns from dataframe."""

    return [
        column
        for column in dataframe.columns
        if len(column.split("_"))>2 and column.split("_")[-2] == "tpm"
    ]


def get_tissues(features: list[str]) -> list[str]:
    """Extract tissue names from TPM columns."""

    return sorted({
        column.rsplit("_", 2)[0]
        for column in features
    })


def get_normalization_function(method_name: str):
    """Retrieve normalization function by name."""

    methods = {
        function.__name__: function
        for function in normfx.NORMALIZATION_METHODS
    }

    if method_name not in methods:
        raise ValueError(f"Unknown normalization method: {method_name}")

    return methods[method_name]


def get_labeling_function(method_name: str):
    """Retrieve labeling function by name."""

    methods = {
        function.__name__: function
        for function in lbl.LABELING_FUNCTIONS
    }

    if method_name not in methods:
        raise ValueError(f"Unknown labeling method: {method_name}")

    return methods[method_name]


def profile_dataframe(
    pair_dataframe: pd.DataFrame,
    profiling_method: str,
    tissues: list[str],
    output_path: Path,
) -> pd.DataFrame:
    """Apply one profiling method and save the profiled dataframe."""

    profiled_dataframe = pf.apply_metric(
        df=pair_dataframe,
        metric_name=profiling_method,
        tissues=tissues,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    profiled_dataframe.to_csv(output_path, sep="\t", index=False)

    return profiled_dataframe


def label_dataframe(
    profiled_dataframe: pd.DataFrame,
    profiling_method: str,
    labeling_method: str,
    n_pos: int,
    n_neg: int,
    seed: int,
) -> tuple[pd.DataFrame, str]:
    """Apply one labeling function and return dataframe plus output name."""

    labeling_function = get_labeling_function(labeling_method)

    labeled_dataframe, output_name = labeling_function(
        df=profiled_dataframe,
        profiling_method=profiling_method,
        n_pos=n_pos,
        n_neg=n_neg,
        random_state=seed,
    )

    output_name = Path(output_name).with_suffix(".tsv").name

    return labeled_dataframe, output_name


def build_orthology_key_map(dataframe: pd.DataFrame) -> dict[str, str]:
    """Build a species-independent gene key map from the original input dataframe."""

    required_columns = {"gene_id_human", "gene_id_mouse"}
    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        raise ValueError(
            "Cannot build same-species pair keys. "
            f"Missing columns in input dataframe: {sorted(missing_columns)}"
        )

    key_column = "gene_name" if "gene_name" in dataframe.columns else None

    gene_key_map: dict[str, str] = {}

    for row_index, row in dataframe.reset_index(drop=True).iterrows():
        shared_key = (
            str(row[key_column])
            if key_column is not None and pd.notna(row[key_column])
            else f"ortholog_group_{row_index}"
        )

        gene_key_map[str(row["gene_id_human"])] = shared_key
        gene_key_map[str(row["gene_id_mouse"])] = shared_key

    return gene_key_map


def add_same_species_pair_key(
    pair_dataframe: pd.DataFrame,
    gene_key_map: dict[str, str],
) -> pd.DataFrame:
    """Add a species-independent pair key to same-species pair rows."""

    required_columns = {"gene_id_1", "gene_id_2"}
    missing_columns = required_columns - set(pair_dataframe.columns)

    if missing_columns:
        raise ValueError(
            "Cannot add same-species pair keys. "
            f"Missing columns in same-species pairs: {sorted(missing_columns)}"
        )

    output_dataframe = pair_dataframe.copy()

    missing_gene_ids = sorted(
        (
            set(output_dataframe["gene_id_1"].astype(str))
            | set(output_dataframe["gene_id_2"].astype(str))
        )
        - set(gene_key_map)
    )

    if missing_gene_ids:
        raise ValueError(
            "Some same-species genes cannot be mapped to an orthology key. "
            f"Examples: {missing_gene_ids[:10]}"
        )

    def make_pair_key(row: pd.Series) -> str:
        first_key = gene_key_map[str(row["gene_id_1"])]
        second_key = gene_key_map[str(row["gene_id_2"])]

        return "__".join(sorted([first_key, second_key]))

    output_dataframe["orthology_key_1"] = output_dataframe["gene_id_1"].astype(str).map(gene_key_map)
    output_dataframe["orthology_key_2"] = output_dataframe["gene_id_2"].astype(str).map(gene_key_map)
    output_dataframe["pair_key"] = output_dataframe.apply(make_pair_key, axis=1)

    return output_dataframe


def copy_directory_contents(source_dir: Path, target_dir: Path) -> None:
    """Replace target_dir with a copy of source_dir."""

    if target_dir.exists():
        shutil.rmtree(target_dir)

    if source_dir.exists():
        shutil.copytree(source_dir, target_dir)
    else:
        target_dir.mkdir(parents=True, exist_ok=True)


def stage_outputs_for_nextflow(dataset_name: str) -> None:
    """Expose stable process-local outputs at paths declared in the Nextflow module."""

    dataset_dir = OUTPUT_DIR / dataset_name

    copy_directory_contents(dataset_dir / "Intermediate_Datasets", Path("Intermediate_Datasets"))
    copy_directory_contents(dataset_dir / "Labeling", Path("Labeling"))


# -----------------------------
# Command-line interface
# -----------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--dataset-name", "--name", dest="dataset_name", required=True)

    parser.add_argument(
        "--normalization",
        required=True,
        choices=[function.__name__ for function in normfx.NORMALIZATION_METHODS],
    )

    parser.add_argument(
        "--pairing",
        default="cross_species",
        choices=["cross_species", "same_species"],
    )

    parser.add_argument(
        "--profiling",
        required=True,
        choices=[function.__name__ for function in pf.ALL_METRICS],
    )

    parser.add_argument(
        "--labeling",
        required=True,
        choices=[function.__name__ for function in lbl.LABELING_FUNCTIONS],
    )

    parser.add_argument("--n-pos", type=int, default=10000)
    parser.add_argument("--n-neg", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


# -----------------------------
# Main workflow
# -----------------------------
def main() -> Path:
    args = parse_args()

    logger.info("Loading raw dataset: %s", args.input)
    raw_dataframe = pd.read_csv(args.input, sep="\t")

    features = get_feature_columns(raw_dataframe)
    tissues = get_tissues(features)

    logger.info("Applying normalization: %s", args.normalization)
    normalization_function = get_normalization_function(args.normalization)

    normalized_dataframe = normalization_function(
        raw_dataframe.copy(),
        numeric_cols=features,
    )

    intermediate_dir = OUTPUT_DIR / args.dataset_name / "Intermediate_Datasets"
    intermediate_dir.mkdir(parents=True, exist_ok=True)

    normalized_path = intermediate_dir / "normalized.tsv"
    normalized_dataframe.to_csv(normalized_path, sep="\t", index=False)

    logger.info("Pairing mode: %s", args.pairing)

    if args.pairing == "cross_species":
        logger.info("Generating all human x mouse pairs.")

        pair_dataframe = mod.all_pairs(
            df_path=normalized_path,
            df_name=args.dataset_name,
            output_dir=OUTPUT_DIR,
        )

        logger.info("Generated %d total cross-species pairs.", len(pair_dataframe))

        profiled_dataframe = profile_dataframe(
            pair_dataframe=pair_dataframe,
            profiling_method=args.profiling,
            tissues=tissues,
            output_path=intermediate_dir / "profiled.tsv",
        )

        labeled_dataframe, output_name = label_dataframe(
            profiled_dataframe=profiled_dataframe,
            profiling_method=args.profiling,
            labeling_method=args.labeling,
            n_pos=args.n_pos,
            n_neg=args.n_neg,
            seed=args.seed,
        )

    else:
        pair_parts: list[pd.DataFrame] = []
        gene_key_map = build_orthology_key_map(raw_dataframe)

        for species in ["human", "mouse"]:
            logger.info("Generating %s x %s pairs.", species, species)

            pair_dataframe_species = mod.same_species_pairs(
                df=normalized_dataframe,
                species=species,
                df_name=args.dataset_name,
                output_dir=OUTPUT_DIR,
            )

            logger.info(
                "Generated %d %s x %s pairs.",
                len(pair_dataframe_species),
                species,
                species,
            )

            pair_dataframe_species["source_species"] = species
            pair_dataframe_species = add_same_species_pair_key(
                pair_dataframe=pair_dataframe_species,
                gene_key_map=gene_key_map,
            )
            pair_parts.append(pair_dataframe_species)

        pair_dataframe = pd.concat(
            pair_parts,
            ignore_index=True,
        )

        logger.info("Generated %d total same-species pairs.", len(pair_dataframe))

        profiled_dataframe = profile_dataframe(
            pair_dataframe=pair_dataframe,
            profiling_method=args.profiling,
            tissues=tissues,
            output_path=intermediate_dir / "profiled_same_species.tsv",
        )

        labeled_dataframe, output_name = label_dataframe(
            profiled_dataframe=profiled_dataframe,
            profiling_method=args.profiling,
            labeling_method=args.labeling,
            n_pos=args.n_pos,
            n_neg=args.n_neg,
            seed=args.seed,
        )

        output_name = output_name.replace(".tsv", "_same_species.tsv")

    labeling_dir = OUTPUT_DIR / args.dataset_name / "Labeling"
    labeling_dir.mkdir(parents=True, exist_ok=True)

    output_path = labeling_dir / output_name
    labeled_dataframe.to_csv(output_path, sep="\t", index=False)

    logger.info("Labeled dataset saved to %s", output_path)

    lbl.sanity_check_boxplot(
        df=labeled_dataframe,
        profiling_method=args.profiling,
        output_path=labeling_dir / f"boxplot_{output_name}.svg",
    )

    manifest = {
        "input": str(args.input),
        "dataset_name": args.dataset_name,
        "normalization": args.normalization,
        "pairing": args.pairing,
        "profiling": args.profiling,
        "labeling": args.labeling,
        "n_pos": args.n_pos,
        "n_neg": args.n_neg,
        "seed": args.seed,
        "labeled_output": output_name,
        "nextflow_output_policy": "stable process-local output folders are declared by the Nextflow process",
    }

    manifest_path = labeling_dir / "labeling_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    stage_outputs_for_nextflow(args.dataset_name)

    logger.info("Labeling outputs generated in process working directory.")

    return output_path


if __name__ == "__main__":
    main()
