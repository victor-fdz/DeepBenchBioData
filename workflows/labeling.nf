include { LABEL_GENE_PAIRS } from '../modules/local/label_gene_pairs'

workflow LABELING_ONLY {
    take:
    raw_expression_dataset_channel

    main:
    LABEL_GENE_PAIRS(raw_expression_dataset_channel)

    emit:
    labeled_pairs = LABEL_GENE_PAIRS.out.labeled_pairs
}
