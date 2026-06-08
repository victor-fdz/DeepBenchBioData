nextflow.enable.dsl = 2

include { FULL_PIPELINE } from './workflows/full_pipeline'

workflow {
    if (!params.input) {
        error "Missing required parameter: --input"
    }

    if (!params.name) {
        error "Missing required parameter: --name"
    }

    if (!params.human_fasta) {
        error "Missing required parameter: --human_fasta"
    }

    if (!params.mouse_fasta) {
        error "Missing required parameter: --mouse_fasta"
    }

    def gene_list_path = params.gene_list ?: params.input

    Channel
        .fromPath(params.input, checkIfExists: true)
        .map { raw_expression_dataset -> tuple(params.name, raw_expression_dataset) }
        .set { raw_expression_dataset_channel }

    Channel
        .fromPath(params.human_fasta, checkIfExists: true)
        .set { human_fasta_channel }

    Channel
        .fromPath(params.mouse_fasta, checkIfExists: true)
        .set { mouse_fasta_channel }

    Channel
        .fromPath(gene_list_path, checkIfExists: true)
        .set { original_gene_list_channel }

    FULL_PIPELINE(
        raw_expression_dataset_channel,
        human_fasta_channel,
        mouse_fasta_channel,
        original_gene_list_channel
    )
}