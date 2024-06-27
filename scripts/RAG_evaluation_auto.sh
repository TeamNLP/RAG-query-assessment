#!/bin/bash

evaluation_model_name="gpt-4o-2024-05-13" #`gpt-4o-2024-05-13`, `gpt-4-0125-preview`, or `gpt-3.5-turbo-0125`

for target_model_name in "meta-llama/Meta-Llama-3-8B-Instruct" "meta-llama/Meta-Llama-3-70B-Instruct"; do
    python classifier/ft_dataset/labeling/auto_evaluation/RAG_evaluation_auto.py \
        --dataset_path classifier/ft_dataset/generation/ms_marco.jsonl \
        --evaluation_model_name ${evaluation_model_name} \
        --target_model_name ${target_model_name} \
        --output_path classifier/ft_dataset/labeling/auto_evaluation/result
done