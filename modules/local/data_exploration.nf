process DATA_EXPLORATION_BEFORE_NORMALIZATION {
    tag { dataset_name }

    publishDir "${params.outdir}/data_exploration/before_normalization", mode: params.publish_mode

    input:
    tuple val(dataset_name), path(raw_expression_dataset)

    output:
    tuple val(dataset_name),
          path("Data_Exploration_Before_Normalization"),
          emit: before_outputs

    script:
    """
    python ${projectDir}/bin/data_exploration_nextflow.py \
        --input ${raw_expression_dataset} \
        --dataset-name ${dataset_name} \
        --normalization-method before_norm \
        --normalization-stage before_norm \
        --output-dir Data_Exploration_Before_Normalization
    """
}


process DATA_EXPLORATION_AFTER_NORMALIZATION {
    tag { dataset_name }

    publishDir "${params.outdir}/data_exploration/after_normalization", mode: params.publish_mode

    input:
    tuple val(dataset_name), path(best_normalization_file), path(orthologs_dir)

    output:
    tuple val(dataset_name),
          path("Data_Exploration_After_Normalization"),
          emit: after_outputs

    script:
    """
    selected_normalization=\$(cat ${best_normalization_file} | tr -d '[:space:]')

    if [ -z "\${selected_normalization}" ] || [ "\${selected_normalization}" = "null" ]; then
        echo "Missing selected normalization method in ${best_normalization_file}" >&2
        exit 1
    fi

    normalized_input="${orthologs_dir}/Orthologs_${dataset_name}_\${selected_normalization}.tsv"

    if [ ! -f "\${normalized_input}" ]; then
        echo "Expected selected normalized ortholog dataset not found: \${normalized_input}" >&2
        echo "Available files in ${orthologs_dir}:" >&2
        find ${orthologs_dir} -maxdepth 1 -type f -name "*.tsv" -print >&2
        exit 1
    fi

    python ${projectDir}/bin/data_exploration_nextflow.py \
        --input "\${normalized_input}" \
        --dataset-name ${dataset_name} \
        --normalization-method "\${selected_normalization}" \
        --normalization-stage after_norm \
        --output-dir Data_Exploration_After_Normalization
    """
}
