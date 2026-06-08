include { LABEL_GENE_PAIRS } from '../modules/local/label_gene_pairs'

workflow LABELING_ONLY {
    take:
    raw_expression_dataset_channel

    main:
    if (!params.normalization) {
        error "Missing required parameter: --normalization"
    }

    if (!params.profiling) {
        error "Missing required parameter: --profiling"
    }

    labeling_input_channel = raw_expression_dataset_channel.map { dataset_name, raw_expression_dataset ->
        tuple(dataset_name, raw_expression_dataset, params.normalization, params.profiling)
    }

    LABEL_GENE_PAIRS(labeling_input_channel)

    emit:
    labeled_pairs = LABEL_GENE_PAIRS.out.labeled_pairs
}