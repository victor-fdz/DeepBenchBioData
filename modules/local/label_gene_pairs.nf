process LABEL_GENE_PAIRS {
    tag { dataset_name }

    publishDir "${params.outdir}/labeling", mode: params.publish_mode

    input:
    tuple val(dataset_name), path(raw_expression_dataset), val(selected_normalization), val(selected_profiling)

    output:
    tuple val(dataset_name),
          path("labeled_pairs.tsv"),
          path("selected_normalization.txt"),
          path("selected_profiling.txt"),
          path("Labeling"),
          path("Intermediate_Datasets"),
          emit: labeled_pairs

    script:
    """
    selected_normalization_input='${selected_normalization}'
    selected_profiling_input='${selected_profiling}'

    if [ -f "\${selected_normalization_input}" ]; then
        selected_normalization=\$(cat "\${selected_normalization_input}" | tr -d '[:space:]')
    else
        selected_normalization="\${selected_normalization_input}"
    fi

    if [ -f "\${selected_profiling_input}" ]; then
        selected_profiling=\$(cat "\${selected_profiling_input}" | tr -d '[:space:]')
    else
        selected_profiling="\${selected_profiling_input}"
    fi

    if [ -z "\${selected_normalization}" ] || [ "\${selected_normalization}" = "null" ]; then
        echo "Missing normalization method. Either run the normalization benchmark before labeling or pass --normalization." >&2
        exit 1
    fi

    if [ -z "\${selected_profiling}" ] || [ "\${selected_profiling}" = "null" ]; then
        echo "Missing profiling method. Either run the profiling benchmark before labeling or pass --profiling." >&2
        exit 1
    fi

    echo "\${selected_normalization}" > selected_normalization.txt
    echo "\${selected_profiling}" > selected_profiling.txt

    python ${projectDir}/bin/label_nextflow.py \
        --input ${raw_expression_dataset} \
        --dataset-name ${dataset_name} \
        --normalization "\${selected_normalization}" \
        --pairing ${params.pairing} \
        --profiling "\${selected_profiling}" \
        --labeling ${params.labeling} \
        --n-pos ${params.n_pos} \
        --n-neg ${params.n_neg} \
        --seed ${params.seed}

    labeled_count=\$(find Labeling -maxdepth 1 -type f -name "*.tsv" | wc -l)
    if [ "\${labeled_count}" -ne 1 ]; then
        echo "Expected exactly one labeled TSV in Labeling/, found \${labeled_count}" >&2
        find Labeling -maxdepth 1 -type f -name "*.tsv" >&2
        exit 1
    fi

    cp \$(find Labeling -maxdepth 1 -type f -name "*.tsv") labeled_pairs.tsv
    """
}
