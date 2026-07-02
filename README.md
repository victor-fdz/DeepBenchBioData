# DeepBenchBioData 🖥️🧬

DeepBenchBioData is a Nextflow-based pipeline for cross-species gene expression benchmarking, promoter-sequence profiling, gene-pair labeling, sequence encoding, and deep learning model training/evaluation.

## Pipeline

`main_full_pipeline.nf` runs the full workflow:

1. data exploration before normalization
2. normalization benchmarking
3. profiling benchmarking, with optional promoter-alignment comparison
4. labeled gene-pair generation
5. promoter-sequence encoding and train/validation/test splitting
6. model training and evaluation

## Requirements

- Nextflow
- A Micromamba/Conda environment named `environment_tfm`
- The input expression table, promoter alignment, and human/mouse promoter FASTA files
- Execution from the repository root

Before running the pipeline, load the environment with `micromamba activate environment_tfm`.

## Inputs

| Parameter | Description |
| --- | --- |
| `--input` | Tabular gene expression dataset. |
| `--name` | Run name used for metadata and output naming. |
| `--promoter` | Promoter alignment path supplied to the run. |
| `--human_fasta` | Human promoter FASTA file. |
| `--mouse_fasta` | Mouse promoter FASTA file. |
| `--labeling` | Strategy used to label gene pairs. |
| `--split_mode` | Train/validation/test split strategy. |
| `--val_frac`, `--test_frac` | Validation and test-set fractions. |
| `--n_pos`, `--n_neg` | Number of positive and negative gene pairs. |
| Model parameters | Training hyperparameters passed to the model step. |

## Runnable example

```bash
nextflow run main_full_pipeline.nf   -c conf/full_pipeline.config   -profile local   --input data/new_expression_data.txt   --name my_results --promoter data/kinases_promoter_alignment   --outdir results/my_results  --human_fasta data/promoter_kinases_human.fasta   --mouse_fasta data/promoter_kinases_mouse.fasta   --labeling rank_labeling  --split_mode anti_leakage   --val_frac 0.15   --test_frac 0.15   --n_pos 10000   --n_neg 8000   --epochs 40 --small_kernel_size 6 --medium_kernel_size 10 --large_kernel_size 20 --dropout 0.1 --learning_rate 0.0005 --weight_decay 0 
```

## Outputs

With the example above, outputs are written under `results/my_results/`:

- `data_exploration/`: exploratory outputs before and after normalization
- `normalization/`: normalization benchmark results and selected method
- `profiling/`: profiling benchmark results and selected metric
- `labeling/`: labeled gene-pair tables
- `encoding/`: encoded sequences and train/validation/test splits
- `model/`: trained model and evaluation outputs
- `nextflow_reports/`: trace, timeline, report, and DAG files

## Repository layout

| Path | Purpose |
| --- | --- |
| `main_full_pipeline.nf` | Full pipeline entry point. |
| `conf/full_pipeline.config` | Default parameters and execution profiles. |
| `workflows/` | Workflow definitions. |
| `modules/local/` | Nextflow process modules. |
| `bin/` | Python wrappers called by Nextflow. |
| `lib/` | Core Python implementation. |
| `envs/environment_tfm.yml` | Environment definition. |
| `data/` | Example input data and promoter files. |


## Adding New Metrics

The benchmarking framework is easily extensible.

To add a new normalization method:

1. Implement the normalization function in `lib/normalization.py`.
2. Add the function name to the `NORMALIZATION_METHODS` list.

To add a new profiling metric:

1. Implement the metric function in `lib/profiling_functions.py`.
2. Register the function in:
   - `INTERNAL_METRICS` if it is a **gene-wise** metric (computed independently for each species).
   - `EXTERNAL_METRICS` if it is a **pair-wise** metric (computed directly between species).