nextflow.enable.dsl = 2

include { LABEL_GENE_PAIRS } from '../modules/local/label_gene_pairs.nf'
include { ENCODE_AND_SPLIT_PAIRS } from '../modules/local/encode_and_split_pairs.nf'
include { TRAIN_AND_EVALUATE_MODEL } from '../modules/local/train_and_evaluate_model.nf'

workflow PROMOTER_SEQUENCE_PIPELINE {

    if (params.input == null) {
        error "Missing required parameter: --input"
    }

    if (params.name == null) {
        error "Missing required parameter: --name"
    }

    if (params.human_fasta == null) {
        error "Missing required parameter: --human_fasta"
    }

    if (params.mouse_fasta == null) {
        error "Missing required parameter: --mouse_fasta"
    }

    raw_expression_dataset_channel = Channel
        .fromPath(params.input, checkIfExists: true)
        .map { raw_expression_dataset -> tuple(params.name, raw_expression_dataset) }

    human_promoter_fasta_channel = Channel.fromPath(params.human_fasta, checkIfExists: true)
    mouse_promoter_fasta_channel = Channel.fromPath(params.mouse_fasta, checkIfExists: true)
    original_gene_list_channel = Channel.fromPath(params.input, checkIfExists: true)

    LABEL_GENE_PAIRS(raw_expression_dataset_channel)

    ENCODE_AND_SPLIT_PAIRS(
        LABEL_GENE_PAIRS.out.labeled_pairs,
        human_promoter_fasta_channel,
        mouse_promoter_fasta_channel,
        original_gene_list_channel
    )

    TRAIN_AND_EVALUATE_MODEL(ENCODE_AND_SPLIT_PAIRS.out.split_directory)
}
