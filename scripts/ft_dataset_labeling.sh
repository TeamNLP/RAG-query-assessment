#!/bin/bash

python classifier/ft_dataset/labeling/query_labeling.py \
    --large_model_eval_result_path classifier/ft_dataset/labeling/auto_evaluation/result/meta-llama-3-70b-instruct_evaluated_by_gpt-3.5-turbo-0125.jsonl \
    --small_model_eval_result_path classifier/ft_dataset/labeling/auto_evaluation/result/meta-llama-3-8b-instruct_evaluated_by_gpt-3.5-turbo-0125.jsonl \
    --output_path classifier/ft_dataset/labeling/auto_evaluation/result

