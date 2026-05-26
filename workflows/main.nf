include { LABEL_GENE_PAIRS } from '../modules/local/label_gene_pairs.nf'
include { ENCODE_AND_SPLIT_PAIRS } from '../modules/local/encode_and_split_pairs.nf'
include { TRAIN_AND_EVALUATE_MODEL } from '../modules/local/train_and_evaluate_model.nf'

workflow PROMOTER_SEQUENCE_PIPELINE {

    // --- Validate required parameters ---
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

    // --- Stage the lib/ directory so Python scripts can import from it ---
    ch_lib = Channel.value(file("${projectDir}/lib", checkIfExists: true))

    // --- Input channels ---
    // Queue channel: one tuple per dataset
    ch_expression = Channel
        .fromPath(params.input, checkIfExists: true)
        .map { f -> tuple(params.name, f) }

    // Value channels for single files (reusable across all items in the queue)
    ch_human_fasta = Channel.value(file(params.human_fasta, checkIfExists: true))
    ch_mouse_fasta = Channel.value(file(params.mouse_fasta, checkIfExists: true))
    ch_gene_list   = Channel.value(file(params.input, checkIfExists: true))

    // --- Step 1: Label gene pairs ---
    LABEL_GENE_PAIRS(
        ch_expression,
        ch_lib
    )

    // --- Step 2: Encode sequences and split into train/val/test ---
    ENCODE_AND_SPLIT_PAIRS(
        LABEL_GENE_PAIRS.out.labeled_pairs,
        ch_human_fasta,
        ch_mouse_fasta,
        ch_gene_list,
        ch_lib
    )

    // --- Step 3: Train and evaluate the model ---
    TRAIN_AND_EVALUATE_MODEL(
        ENCODE_AND_SPLIT_PAIRS.out.split_directory,
        ch_lib
    )
}
