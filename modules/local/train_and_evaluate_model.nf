process TRAIN_AND_EVALUATE_MODEL {
    tag "${dataset_name}"

    publishDir {"${params.outdir}/${dataset_name}/Model"}, mode: params.publish_mode

    input:
    tuple val(dataset_name), path(split_directory)

    output:
    tuple val(dataset_name), path("results/${dataset_name}/Model"), emit: model_directory

    script:
    optimize_argument = params.optimize ? '--optimize' : ''

    """
    python ${projectDir}/bin/model_processing.py \
        --data ${split_directory} \
        --name ${dataset_name} \
        --metric ${params.profiling} \
        --batch-size ${params.batch_size} \
        --epochs ${params.epochs} \
        --learning-rate ${params.learning_rate} \
        --margin ${params.margin} \
        --dropout ${params.dropout} \
        --weight-decay ${params.weight_decay} \
        ${optimize_argument}
    """
}
