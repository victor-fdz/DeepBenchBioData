nextflow.enable.dsl = 2

include { PREPROCESSING_BENCHMARK_AND_LABELING } from './workflows/preprocessing'

workflow {
    if (!params.input) {
        error "Missing required parameter: --input"
    }

    if (!params.name) {
        error "Missing required parameter: --name"
    }

    Channel
        .fromPath(params.input, checkIfExists: true)
        .map { raw_expression_dataset -> tuple(params.name, raw_expression_dataset) }
        .set { raw_expression_dataset_channel }

    PREPROCESSING_BENCHMARK_AND_LABELING(raw_expression_dataset_channel)
}
