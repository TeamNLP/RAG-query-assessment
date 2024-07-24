#!/bin/bash

for model_path in meta-llama/Meta-Llama-3.1-8B meta-llama/Meta-Llama-3.1-8B-Instruct meta-llama/Meta-Llama-3-8B meta-llama/Meta-Llama-3-8B-Instruct
do
    model_suffix=${model_path#*/}
    for dataset in hotpotqa 2wikimultihopqa musique nq trivia squad
    do
        python experiments/predict.py \
            --output_directory "predictions_$model_suffix" \
            --dataset $dataset \
            --dataset_type test_subsampled \
            --retrieval_top_n 3 \
            --generator_model_name $model_path
    done
done