include { TRAIN_AND_EVALUATE_MODEL } from '../modules/local/train_and_evaluate_model'

workflow MODEL_TRAINING_ONLY {
    take:
    split_files_channel

    main:
    if (!params.metric) {
        error "Missing required parameter for standalone model training: --metric"
    }

    model_input_channel = split_files_channel.map { dataset_name, train_split, validation_split, test_split ->
        tuple(dataset_name, train_split, validation_split, test_split, params.metric)
    }

    TRAIN_AND_EVALUATE_MODEL(model_input_channel)

    emit:
    model_outputs = TRAIN_AND_EVALUATE_MODEL.out.model_outputs
}
