# Normalization Pipeline

Cross-species gene expression normalization benchmarking pipeline.

## Installation

```bash
cd normalization_pipeline
pip install -e .
```

This installs the package in **editable mode** — changes to the code take effect immediately without reinstalling.

## Usage

After installation, run from anywhere:

```bash
normalize \
  --input /path/to/your/data.tsv \
  --name dataset_name \
  --tissue General \
  --r-exec /path/to/Rscript
```

### Arguments

- `--input`: Path to input TSV with TPM values (required)
- `--name`: Dataset name for output file naming (required)
- `--tissue`: Tissue for method ranking (default: `General`)
- `--r-exec`: Path to Rscript executable (default: `Rscript`)

### Example

```bash
normalize \
  --input data/kinases.tsv \
  --name kinases \
  --tissue heart
```

## Directory Structure

```
normalization_pipeline/
├── bin/
│   ├── normalize.py               # Main entry point
│   └── deseq2_edger_normalize.R   # R normalization script
├── lib/
│   ├── __init__.py
│   ├── modify_dataset.py          # Generate non-orthologous pairs
│   ├── normalization.py           # Normalization functions
│   ├── compute_stats.py           # Correlation statistics
│   └── plotting.py                # Visualization utilities
├── results/                       # Output directory (generated)
├── requirements.txt
├── setup.py
└── README.md
```

## Output

Results are saved to `results/<dataset_name>/`:
- `Orthologs/` — Normalized orthologous datasets
- `NonOrthologs/` — Normalized non-orthologous datasets  
- `Normalization/` — Plots and stats summary

## Development

To modify the code:

1. Edit files in `bin/` or `lib/`
2. Changes take effect immediately (editable install)
3. No need to reinstall

To run without installing:

```bash
cd normalization_pipeline
PYTHONPATH=. python bin/normalize.py --input ... --name ...
```
