#!/bin/bash

# Check if .env file exists in the current directory
if [ -f ".env" ]; then
    export $(egrep -v '^#' .env | xargs)
else
    HUGGINGFACE_TOKEN=None
fi

model_path=google/flan-t5-small
model_name=${model_path#*/}

EPOCHS=15
SEED=42

# Define hyperparameters for simple_hypersearch
label_smoothing_factors="0.15 0.1"
learning_rates="1e-5 3e-5"
weight_decays="0.1 0.01 0.001 0.005 0.0001"
dropout_probs="0.8 0.2 0.4 0.6"
strategies="new_0.6 new_median new_0.5 new_avg"

# Command template for simple_hypersearch
command_template="python classifier/fine-tuning/run_update.py \
    --do_train \
    --do_eval \
    --do_predict \
    --model_name_or_path $model_path \
    --model_type AutoModelForSequenceClassification \
    --train_file classifier/ft_dataset/{strategy}/ft_dataset_train.json \
    --validation_file classifier/ft_dataset/{strategy}/ft_dataset_valid.json \
    --test_file classifier/ft_dataset/{strategy}/ft_dataset_test.json \
    --seed $SEED \
    --num_train_epochs $EPOCHS \
    --use_earlystopping \
    --early_stopping_patience 3 \
    --label_smoothing_factor {label_smoothing_factor} \
    --hidden_dropout_prob {dropout_prob} \
    --attention_probs_dropout_prob {dropout_prob} \
    --output_dir classifier/model/$model_name-qclassifier_{strategy}-droprob_{dropout_prob}-smooth_{label_smoothing_factor}-lr_{learning_rate}-dcy_{weight_decay} \
    --logging_strategy 'epoch' \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --learning_rate {learning_rate} \
    --weight_decay {weight_decay} \
    --evaluation_strategy 'epoch' \
    --save_strategy 'epoch' \
    --save_total_limit 2 \
    --load_best_model_at_end \
    --push_to_hub \
    --hub_model_id $model_name-qclassifier_{strategy}-droprob_{dropout_prob}-smooth_{label_smoothing_factor}-lr_{learning_rate}-dcy_{weight_decay} \
    --hub_token $HUGGINGFACE_TOKEN"

# Run hyperparameter search and GPU scheduling
simple_hypersearch "$command_template" \
    --sampling-mode grid \
    -p label_smoothing_factor $label_smoothing_factors \
    -p learning_rate $learning_rates \
    -p weight_decay $weight_decays \
    -p dropout_prob $dropout_probs \
    -p strategy $strategies \
    | simple_gpu_scheduler --gpus "$CUDA_VISIBLE_DEVICES"