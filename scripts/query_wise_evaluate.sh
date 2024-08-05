#!/bin/bash

python classifier/ft_dataset/evaluation/query_wise_evaluate.py \
    --bertscore_model_type microsoft/deberta-xlarge-mnli \
    --predictions_result_dir classifier/ft_dataset/predictions \
    --output_result_dir classifier/ft_dataset/evaluation/result