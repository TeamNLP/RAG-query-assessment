#!/bin/bash

# 0.5
python classifier/ft_dataset/labeling/query_wise_label.py \
    --seed 13370 \
    --eval_result_path classifier/ft_dataset/evaluation/result \
    --threshold_strategy 0.5 \
    --train_ratio 80 \
    --valid_ratio 10 \
    --test_ratio 10 \
    --result_dataset_path classifier/ft_dataset/new_0.5

# 0.6
python classifier/ft_dataset/labeling/query_wise_label.py \
    --seed 13370 \
    --eval_result_path classifier/ft_dataset/evaluation/result \
    --threshold_strategy 0.6 \
    --train_ratio 80 \
    --valid_ratio 10 \
    --test_ratio 10 \
    --result_dataset_path classifier/ft_dataset/new_0.6


# average
python classifier/ft_dataset/labeling/query_wise_label.py \
    --seed 13370 \
    --eval_result_path classifier/ft_dataset/evaluation/result \
    --threshold_strategy mean \
    --train_ratio 80 \
    --valid_ratio 10 \
    --test_ratio 10 \
    --result_dataset_path classifier/ft_dataset/new_avg


# median
python classifier/ft_dataset/labeling/query_wise_label.py \
    --seed 13370 \
    --eval_result_path classifier/ft_dataset/evaluation/result \
    --threshold_strategy median \
    --train_ratio 80 \
    --valid_ratio 10 \
    --test_ratio 10 \
    --result_dataset_path classifier/ft_dataset/new_median