#!/usr/bin/env python3
"""
Full preprocessing pipeline:

1. Normalization (benchmark + selection OR manual)
2. Pairing mode selection
3. Profiling (manual or benchmark where meaningful)
4. Labeling

Pairing modes:
- cross_species: existing human x mouse workflow
- same_species: human x human and mouse x mouse after normalization

Arguments: 
    python bin/preprocess.py \
        --input \
        --name  \
        --normalization  \
        --pairing  \
        --profiling  \
        --labeling  \
        
        Optional:
        --n-pos  \
        --n-neg 
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bin.normalize import main as run_normalization
from bin.profiling import main as run_profiling
from bin.label import main as run_labeling

logging.basicConfig(
    level=logging.INFO,
    format="\033[1;32m%(levelname)s\033[0m | \033[1;36m%(name)s\033[0m | %(message)s",
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse pipeline arguments."""

    parser = argparse.ArgumentParser(description="Full preprocessing pipeline")

    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--name", required=True)

    parser.add_argument("--normalization", default=None)
    parser.add_argument(
        "--pairing",
        default="cross_species",
        choices=["cross_species", "same_species"],
        help="Pair-generation mode after normalization.",
    )
    parser.add_argument("--profiling", default=None)
    parser.add_argument("--labeling", default="rank_labeling_random")

    parser.add_argument("--n-pos", type=int, default=10000)
    parser.add_argument("--n-neg", type=int, default=10000)

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--expression-unit",
        default="tpm",
        choices=["tpm", "counts"],
    )

    return parser.parse_args()


def main():
    """Run full pipeline by chaining existing scripts."""

    args = parse_args()

    logger.info("STEP 1 — Normalization benchmark / selection")

    if args.normalization:
        norm_best = args.normalization
        logger.info("Normalization selected manually: %s", norm_best)
    else:
        norm_best = run_normalization(
            [
                "--input",
                str(args.input),
                "--name",
                args.name,
                "--expression-unit",
                args.expression_unit,
            ]
        )

        logger.info("Selected normalization: %s", norm_best)

    logger.info("STEP 2 — Pairing mode: %s", args.pairing)

    logger.info("STEP 3 — Profiling benchmark / selection")

    if args.profiling:
        prof_best = args.profiling
        logger.info("Profiling selected manually: %s", prof_best)
    else:
        if args.pairing == "same_species":
            raise ValueError(
                "Automatic profiling selection is still based on cross-species "
                "ortholog/non-ortholog benchmarking. For --pairing same_species, "
                "pass --profiling explicitly, e.g. --profiling cosine_sim."
            )

        prof_best = run_profiling(
            [
                "--input",
                str(args.input),
                "--name",
                args.name,
                "--expression-unit",
                args.expression_unit,
            ]
        )

        logger.info("Selected profiling: %s", prof_best)

    logger.info("STEP 4 — Labeling")

    labeling_args = [
        "--input",
        str(args.input),
        "--name",
        args.name,
        "--normalization",
        norm_best,
        "--pairing",
        args.pairing,
        "--profiling",
        prof_best,
        "--labeling",
        args.labeling,
        "--n-pos",
        str(args.n_pos),
        "--n-neg",
        str(args.n_neg),
        "--seed",
        str(args.seed),
        "--expression-unit",
        args.expression_unit,
    ]

    out_path = run_labeling(labeling_args)

    logger.info("Pipeline finished -> %s", out_path)

    return out_path


if __name__ == "__main__":
    main()
