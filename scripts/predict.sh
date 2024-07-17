#!/bin/bash

for dataset in hotpotqa 2wikimultihopqa musique nq trivia squad
do
    python experiments/predict.py \
        --dataset $dataset \
        --dataset_type test_subsampled \
        --retrieval_top_n 3 \
        --generator_model_name google/gemma-1.1-2b-it
done