#!/bin/bash

python classifier/ft_dataset/labeling/query_wise_label.py \
    --seed 13370 \
    --eval_result_path classifier/ft_dataset/evaluation/result \
    --large_performance_threshold 0.5 \
    --small_performance_threshold 0.5 \
    --result_dataset_path classifier/ft_dataset