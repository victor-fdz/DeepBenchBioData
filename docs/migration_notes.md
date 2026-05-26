# Migration notes

This scaffold intentionally keeps the Python scripts as the computational engine.
Nextflow is used first as the orchestration layer.

Current working workflow:

1. LABEL_GENE_PAIRS wraps bin/label.py
2. ENCODE_AND_SPLIT_PAIRS wraps bin/encode_split.py
3. TRAIN_AND_EVALUATE_MODEL wraps bin/model_processing.py

The normalize and profile benchmark modules are included but not wired into the main workflow yet.
Reason: the current Python benchmark flow is not fully Nextflow friendly because normalization selection can be interactive and label.py already performs normalization and profiling internally when explicit methods are provided.

Recommended next refactor:

1. Make each Python script accept an explicit output directory.
2. Make normalization benchmarking non-interactive.
3. Split model_processing.py into independent train and evaluate Nextflow processes.
4. Move Python dependencies into a locked container or a fully pinned Conda environment.
