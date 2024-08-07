#!/bin/bash

# Check if .env file exists in the current directory
if [ -f ".env" ]; then
    export $(egrep -v '^#' .env | xargs)
else
    HUGGINGFACE_TOKEN=None
fi

for model_path in google/flan-t5-small google/t5-v1_1-small google/flan-t5-base google/t5-v1_1-base google/flan-t5-large google/t5-v1_1-large google/flan-t5-xl google/t5-v1_1-xl
do
    model_name=${model_path#*/}

    EPOCHS=15
    REPOSITORY_ID=$model_name-query_level_estimator
    SEED=42

    python classifier/fine-tuning/run.py \
        --do_train \
        --do_eval \
        --do_predict \
        --model_name_or_path $model_path \
        --model_type AutoModelForSequenceClassification \
        --train_file classifier/ft_dataset/ft_dataset_train.json \
        --validation_file classifier/ft_dataset/ft_dataset_valid.json \
        --test_file classifier/ft_dataset/ft_dataset_test.json \
        --seed ${SEED} \
        --num_train_epochs ${EPOCHS} \
        --use_earlystopping \
        --early_stopping_patience 3 \
        --output_dir classifier/model/${REPOSITORY_ID} \
        --logging_strategy "epoch" \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 8 \
        --learning_rate 3e-4 \
        --evaluation_strategy "epoch" \
        --save_strategy "epoch" \
        --save_total_limit 3 \
        --load_best_model_at_end \
        --push_to_hub \
        --hub_model_id ${REPOSITORY_ID} \
        --hub_token ${HUGGINGFACE_TOKEN}
done