#!/usr/bin/env python3
"""Generate labeled datasets from raw expression data.

Workflow:
    Raw TPM data
        -> normalization
        -> pair generation (cross_species or same_species)
        -> profiling
        -> labeling
        -> labeled dataset

Pairing modes:
    cross_species:
        Build human x mouse pairs, then profile and label them.
    same_species:
        Build human x human and mouse x mouse pairs separately after
        normalization. Both species are concatenated, profiled, and labeled
        together in one global ranking. The final file keeps ``source_species``
        and ``pair_key`` columns so human-human and mouse-mouse equivalent
        pairs can be compared later.
"""

import argparse
import logging
import sys
from pathlib import Path

# allow direct execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from lib import labeling as lbl
from lib import modify_dataset as mod
from lib import normalization as normfx
from lib import profiling_functions as pf

logging.basicConfig(
    level=logging.INFO,
    format="\033[1;32m%(levelname)s\033[0m | \033[1;36m%(name)s\033[0m | %(message)s",
)

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("results")


# -----------------------------
# Helpers
# -----------------------------
def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Extract TPM feature columns from dataframe."""

    return [
        col
        for col in df.columns
        if col.split("_")[-2] == "tpm"
    ]


def get_tissues(features: list[str]) -> list[str]:
    """Extract tissue names from TPM columns."""

    return sorted({
        col.rsplit("_", 2)[0]
        for col in features
    })


def get_normalization_function(method_name: str):
    """Retrieve normalization function by name."""

    methods = {
        fx.__name__: fx
        for fx in normfx.NORMALIZATION_METHODS
    }

    if method_name not in methods:
        raise ValueError(f"Unknown normalization method: {method_name}")

    return methods[method_name]


def profile_dataframe(
    df_pairs: pd.DataFrame,
    profiling_method: str,
    tissues: list[str],
    output_path: Path,
) -> pd.DataFrame:
    """Apply one profiling method and save the profiled dataframe."""

    df_profiled = pf.apply_metric(
        df=df_pairs,
        metric_name=profiling_method,
        tissues=tissues,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_profiled.to_csv(output_path, sep="\t", index=False)

    return df_profiled


def get_labeling_function(method_name: str):
    """Retrieve labeling function by name."""

    methods = {
        fx.__name__: fx
        for fx in lbl.LABELING_FUNCTIONS
    }

    if method_name not in methods:
        raise ValueError(f"Unknown labeling method: {method_name}")

    return methods[method_name]


def label_dataframe(
    df_profiled: pd.DataFrame,
    profiling_method: str,
    labeling_method: str,
    n_pos: int,
    n_neg: int,
    seed: int,
) -> tuple[pd.DataFrame, str]:
    """Apply one labeling function and return dataframe plus output name."""

    labeling_function = get_labeling_function(labeling_method)

    df_labeled, output_name = labeling_function(
        df=df_profiled,
        profiling_method=profiling_method,
        n_pos=n_pos,
        n_neg=n_neg,
        random_state=seed,
    )

    return df_labeled, output_name


def build_orthology_key_map(df: pd.DataFrame) -> dict[str, str]:
    """Build a species-independent gene key map from the original input dataframe."""

    required_columns = {"gene_id_human", "gene_id_mouse"}

    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            "Cannot build same-species pair keys. "
            f"Missing columns in input dataframe: {sorted(missing_columns)}"
        )

    key_column = "gene_name" if "gene_name" in df.columns else None

    gene_key_map: dict[str, str] = {}

    for row_index, row in df.reset_index(drop=True).iterrows():
        shared_key = (
            str(row[key_column])
            if key_column is not None and pd.notna(row[key_column])
            else f"ortholog_group_{row_index}"
        )

        gene_key_map[str(row["gene_id_human"])] = shared_key
        gene_key_map[str(row["gene_id_mouse"])] = shared_key

    return gene_key_map


def add_same_species_pair_key(
    df_pairs: pd.DataFrame,
    gene_key_map: dict[str, str],
) -> pd.DataFrame:
    """Add a species-independent pair key to same-species pair rows."""

    required_columns = {"gene_id_1", "gene_id_2"}

    missing_columns = required_columns - set(df_pairs.columns)
    if missing_columns:
        raise ValueError(
            "Cannot add same-species pair keys. "
            f"Missing columns in same-species pairs: {sorted(missing_columns)}"
        )

    df_out = df_pairs.copy()

    missing_gene_ids = sorted(
        (
            set(df_out["gene_id_1"].astype(str))
            | set(df_out["gene_id_2"].astype(str))
        )
        - set(gene_key_map)
    )

    if missing_gene_ids:
        raise ValueError(
            "Some same-species genes cannot be mapped to an orthology key. "
            f"Examples: {missing_gene_ids[:10]}"
        )

    def _make_pair_key(row: pd.Series) -> str:
        key_1 = gene_key_map[str(row["gene_id_1"])]
        key_2 = gene_key_map[str(row["gene_id_2"])]

        return "__".join(sorted([key_1, key_2]))

    df_out["orthology_key_1"] = df_out["gene_id_1"].astype(str).map(gene_key_map)
    df_out["orthology_key_2"] = df_out["gene_id_2"].astype(str).map(gene_key_map)
    df_out["pair_key"] = df_out.apply(_make_pair_key, axis=1)

    return df_out


# -----------------------------
# CLI
# -----------------------------
def parse_args(cli_args: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to raw TPM TSV dataset.",
    )

    parser.add_argument(
        "--name",
        required=True,
        help="Dataset name used for output naming.",
    )

    parser.add_argument(
        "--normalization",
        required=True,
        choices=[fx.__name__ for fx in normfx.NORMALIZATION_METHODS],
        help="Normalization method selected from normalization benchmarking.",
    )

    parser.add_argument(
        "--pairing",
        default="cross_species",
        choices=["cross_species", "same_species"],
        help=(
            "Pair-generation mode after normalization. "
            "cross_species builds human x mouse pairs. "
            "same_species builds human x human and mouse x mouse pairs separately."
        ),
    )

    parser.add_argument(
        "--profiling",
        required=True,
        choices=[fx.__name__ for fx in pf.ALL_METRICS],
        help="Profiling metric selected from profiling benchmarking.",
    )

    parser.add_argument(
        "--labeling",
        required=True,
        choices=[fx.__name__ for fx in lbl.LABELING_FUNCTIONS],
        help="Labeling strategy.",
    )

    parser.add_argument(
        "--n-pos",
        type=int,
        default=10000,
        help=(
            "Number of positive pairs. In same_species mode this "
            "is applied once after concatenating human-human and mouse-mouse pairs."
        ),
    )

    parser.add_argument(
        "--n-neg",
        type=int,
        default=10000,
        help=(
            "Number of negative pairs. In same_species mode this "
            "is applied once after concatenating human-human and mouse-mouse pairs."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    return parser.parse_args(cli_args)


# -----------------------------
# Main workflow
# -----------------------------
def main(cli_args: list[str] | None = None) -> Path:
    """Run labeling workflow."""

    args = parse_args() if cli_args is None else parse_args(cli_args)

    logger.info("Loading raw dataset.")

    # Prepare data
    df = pd.read_csv(args.input, sep="\t")

    features = get_feature_columns(df)
    tissues = get_tissues(features)

    # Normalize
    logger.info("Applying normalization: %s", args.normalization)
    normalization_function = get_normalization_function(args.normalization)

    df_norm = normalization_function(
        df.copy(),
        numeric_cols=features,
    )

    temp_dir = OUTPUT_DIR / args.name / "Intermediate_Datasets"
    temp_dir.mkdir(parents=True, exist_ok=True)

    normalized_path = temp_dir / "normalized.tsv"
    df_norm.to_csv(normalized_path, sep="\t", index=False)

    # Generate pairs
    logger.info("Pairing mode: %s", args.pairing)

    if args.pairing == "cross_species":
        logger.info("Generating all human x mouse pairs.")

        df_pairs = mod.all_pairs(
            df_path=normalized_path,
            df_name=args.name,
            output_dir=OUTPUT_DIR,
        )

        logger.info("Generated %d total cross-species pairs.", len(df_pairs))

        # Generate expression profile similarities
        df_profiled = profile_dataframe(
            df_pairs=df_pairs,
            profiling_method=args.profiling,
            tissues=tissues,
            output_path=temp_dir / "profiled.tsv",
        )

        # Label pairs in positive, negative and undefined
        df_labeled, output_name = label_dataframe(
            df_profiled=df_profiled,
            profiling_method=args.profiling,
            labeling_method=args.labeling,
            n_pos=args.n_pos,
            n_neg=args.n_neg,
            seed=args.seed,
        )

    else:
        pair_parts: list[pd.DataFrame] = []
        gene_key_map = build_orthology_key_map(df)

        for species in ["human", "mouse"]:
            logger.info("Generating %s x %s pairs.", species, species)

            df_pairs_species = mod.same_species_pairs(
                df=df_norm,
                species=species,
                df_name=args.name,
                output_dir=OUTPUT_DIR,
            )

            logger.info(
                "Generated %d %s x %s pairs.",
                len(df_pairs_species),
                species,
                species,
            )

            df_pairs_species["source_species"] = species
            df_pairs_species = add_same_species_pair_key(
                df_pairs=df_pairs_species,
                gene_key_map=gene_key_map,
            )
            pair_parts.append(df_pairs_species)

        df_pairs = pd.concat(
            pair_parts,
            ignore_index=True,
        )

        logger.info(
            "Generated %d total same-species pairs.",
            len(df_pairs),
        )

        # Generate expression profile similarities on the combined dataframe.
        df_profiled = profile_dataframe(
            df_pairs=df_pairs,
            profiling_method=args.profiling,
            tissues=tissues,
            output_path=temp_dir / "profiled_same_species.tsv",
        )

        # Label the combined human-human + mouse-mouse dataframe globally.
        df_labeled, output_name = label_dataframe(
            df_profiled=df_profiled,
            profiling_method=args.profiling,
            labeling_method=args.labeling,
            n_pos=args.n_pos,
            n_neg=args.n_neg,
            seed=args.seed,
        )

        output_name = output_name.replace(
            ".tsv",
            "_same_species.tsv",
        )

    out_dir = OUTPUT_DIR / args.name / "Labeling"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / output_name
    df_labeled.to_csv(out_path, sep="\t", index=False)

    logger.info("Labeled dataset saved to %s", out_path)

    lbl.sanity_check_boxplot(
        df=df_labeled,
        profiling_method=args.profiling,
        output_path=out_dir / f"boxplot_{output_name}.png",
    )

    logger.info("Sanity-check plot generated.")

    return out_path


if __name__ == "__main__":
    main()
