nextflow.enable.dsl = 2

include { MODEL_TRAINING_ONLY } from './workflows/model_training'

workflow {
    if (!params.name) {
        error "Missing required parameter: --name"
    }

    if (!params.train) {
        error "Missing required parameter: --train"
    }

    if (!params.validation) {
        error "Missing required parameter: --validation"
    }

    if (!params.test) {
        error "Missing required parameter: --test"
    }

    Channel
        .of(
            tuple(
                params.name,
                file(params.train, checkIfExists: true),
                file(params.validation, checkIfExists: true),
                file(params.test, checkIfExists: true)
            )
        )
        .set { split_files_channel }

    MODEL_TRAINING_ONLY(split_files_channel)
}
