#!/bin/bash

# Check if .env file exists in the current directory
if [ -f ".env" ]; then
  export $(egrep -v '^#' .env | xargs)
fi

EPOCHS=15
REPOSITORY_ID=flan-t5-small-query_rewriting_classifier
SEED=42

python classifier/fine-tuning/run.py \
    --do_train \
    --model_name_or_path google/flan-t5-small \
    --model_type AutoModelForSequenceClassification \
    --train_file classifier/ft_dataset/ft_dataset.json \
    --seed ${SEED} \
    --num_train_epochs ${EPOCHS} \
    --output_dir classifier/model/${REPOSITORY_ID} \
    --logging_strategy steps \
    --logging_steps 1000 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --learning_rate 3e-4 \
    --save_strategy "epoch" \
    --save_total_limit 2 \
    --push_to_hub \
    --hub_model_id ${REPOSITORY_ID} \
    --hub_token ${HUGGINGFACE_TOKEN}