#!/bin/bash

for output_directory_prefix in predictions predictions_wo predictions_chat
do
    for model_path in meta-llama/Meta-Llama-3.1-8B meta-llama/Meta-Llama-3.1-8B-Instruct meta-llama/Meta-Llama-3-8B meta-llama/Meta-Llama-3-8B-Instruct
    do
        model_suffix=${model_path#*/}
        for dataset in nq trivia
        # for dataset in hotpotqa 2wikimultihopqa musique nq trivia squad
        do
            python experiments/evaluate.py \
                --input_directory "$output_directory_prefix"_"$model_suffix" \
                --dataset $dataset \
                --dataset_type test_subsampled
        done
    done

    python experiments/extract_evaluation_results.py \
        --output_directory_prefix $output_directory_prefix
done