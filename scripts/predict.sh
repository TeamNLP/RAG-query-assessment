#!/bin/bash

for dataset in hotpotqa 2wikimultihopqa musique nq trivia squad
do
    python experiments/predict.py \
        --dataset $dataset \
        --dataset_type test_subsampled \
        --retrieval_top_n 3 \
        --generator_model_name meta-llama/Meta-Llama-3-8B
done