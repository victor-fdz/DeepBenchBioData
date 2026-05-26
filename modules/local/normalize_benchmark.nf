process NORMALIZE_BENCHMARK {
    tag "${dataset_name}"

    publishDir "${params.outdir}/${dataset_name}/Normalization", mode: params.publish_mode

    input:
    tuple val(dataset_name), path(raw_expression_dataset)

    output:
    tuple val(dataset_name), path("results/${dataset_name}/Normalization"), emit: normalization_directory

    script:
    """
    python ${projectDir}/bin/normalize.py \
        --input ${raw_expression_dataset} \
        --name ${dataset_name} \
        --tissue ${params.normalization_ranking_tissue}
    """
}
