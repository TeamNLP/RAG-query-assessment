#!/bin/bash

for model_path in meta-llama/Meta-Llama-3.1-8B meta-llama/Meta-Llama-3.1-8B-Instruct meta-llama/Meta-Llama-3-8B meta-llama/Meta-Llama-3-8B-Instruct
do
    model_suffix=${model_path#*/}
    for dataset in hotpotqa 2wikimultihopqa musique nq trivia squad
    do
        python experiments/evaluate.py \
            --input_directory "predictions_$model_suffix" \
            --dataset $dataset \
            --dataset_type test_subsampled
    done
done

python experiments/extract_evaluation_results.py