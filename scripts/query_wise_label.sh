#!/bin/bash

python classifier/ft_dataset/labeling/query_wise_label.py \
    --eval_result_path classifier/ft_dataset/evaluation/result \
    --performance_threshold 0.5 \
    --result_file_path classifier/ft_dataset/ft_dataset.json