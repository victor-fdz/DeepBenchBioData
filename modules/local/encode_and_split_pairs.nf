process ENCODE_AND_SPLIT_PAIRS {
    tag { dataset_name }

    publishDir { "${params.outdir}/${dataset_name}/Encoded" }, mode: params.publish_mode

    input:
    tuple val(dataset_name), path(labeled_pairs)
    path human_fasta
    path mouse_fasta
    path original_gene_list

    output:
    tuple val(dataset_name), path("results/${dataset_name}/splits/${params.split_mode}"), emit: split_directory

    script:
    """
    python ${projectDir}/bin/encode_split.py \
        --input ${labeled_pairs} \
        --name ${dataset_name} \
        --human-fasta ${human_fasta} \
        --mouse-fasta ${mouse_fasta} \
        --profiling ${params.profiling} \
        --split-mode ${params.split_mode} \
        --gene-list ${original_gene_list} \
        --val-frac ${params.val_frac} \
        --test-frac ${params.test_frac} \
        --seed ${params.seed}
    """
}
