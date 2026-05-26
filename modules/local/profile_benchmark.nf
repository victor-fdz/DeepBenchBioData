process PROFILE_BENCHMARK {
    tag "${dataset_name}"

    publishDir "${params.outdir}/${dataset_name}/Benchmarking", mode: params.publish_mode

    input:
    tuple val(dataset_name), path(raw_expression_dataset)

    output:
    tuple val(dataset_name), path("results/${dataset_name}/Benchmarking"), emit: profiling_benchmark_directory

    script:
    promoter_argument = params.promoter_alignment == null ? '' : "--promoter ${params.promoter_alignment}"

    """
    python ${projectDir}/bin/profiling.py \
        --input ${raw_expression_dataset} \
        --name ${dataset_name} \
        ${promoter_argument}
    """
}
