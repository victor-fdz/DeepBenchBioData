import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Calculate promoter alignment identity difference between functional categories."
    )
    parser.add_argument(
        "--alignment", required=True, help="Path to the promoter alignment TSV file"
    )
    parser.add_argument(
        "--labeling", required=True, help="Path to the labeling/splits TSV file"
    )
    parser.add_argument(
        "--output",
        default="alignment_identity_difference.txt",
        help="Path to save the output text file (default: alignment_identity_difference.txt)",
    )

    args = parser.parse_args()

    # Load data
    df_alignment = pd.read_csv(args.alignment, sep="\t")
    df_labeling = pd.read_csv(args.labeling, sep="\t")

    # Merge dataframes
    df_all = pd.merge(
        df_alignment, df_labeling, on=["gene_id_human", "gene_id_mouse"], how="inner"
    )

    # Separate by label
    positive_pairs = df_all[df_all["label"] == 1]
    negative_pairs = df_all[df_all["label"] == 0]

    # Calculate means
    positive_means = positive_pairs.mean(numeric_only=True)
    negative_means = negative_pairs.mean(numeric_only=True)

    # Compute metric
    diff = positive_means - negative_means
    diff_val = float(diff["promoter_identity"])
    max_diff = 75
    diff_percentage = (diff_val / max_diff) * 100

    # Save results
    with open(args.output, "w") as f:
        f.write(
            f"""Positive pairs mean promoter identity: {positive_means['promoter_identity']:.2f}\n
Negative pairs mean promoter identity: {negative_means['promoter_identity']:.2f}\n
Alignment-based identity difference by functional categories: {diff_percentage:.2f}%\n"""
        )


if __name__ == "__main__":
    main()