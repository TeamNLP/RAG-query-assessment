#!/bin/bash

for rewrite_method in method1 method2 method3 method4 method5
do
    # use template with instruction
    output_directory_prefix="$rewrite_method"_predictions
    for model_path in meta-llama/Meta-Llama-3.1-8B meta-llama/Meta-Llama-3.1-8B-Instruct meta-llama/Meta-Llama-3-8B meta-llama/Meta-Llama-3-8B-Instruct
    do
        model_suffix=${model_path#*/}
        for dataset in hotpotqa 2wikimultihopqa musique nq trivia squad
        do
            python experiments/evaluate.py \
                --input_directory "$output_directory_prefix"_"$model_suffix" \
                --dataset $dataset \
                --dataset_type test_subsampled
        done
    done
    python experiments/extract_evaluation_results.py \
        --output_directory_prefix $output_directory_prefix


    # use_template_wo_instruction
    output_directory_prefix="$rewrite_method"_predictions_wo
    for model_path in meta-llama/Meta-Llama-3.1-8B meta-llama/Meta-Llama-3-8B 
    do
        model_suffix=${model_path#*/}
        for dataset in hotpotqa 2wikimultihopqa musique nq trivia squad
        do
            python experiments/evaluate.py \
                --input_directory "$output_directory_prefix"_"$model_suffix" \
                --dataset $dataset \
                --dataset_type test_subsampled
        done
    done
    python experiments/extract_evaluation_results.py \
        --output_directory_prefix $output_directory_prefix


    # use_chat_template
    output_directory_prefix="$rewrite_method"_predictions_chat
    for model_path in meta-llama/Meta-Llama-3.1-8B-Instruct meta-llama/Meta-Llama-3-8B-Instruct gpt-4o-mini-2024-07-18
    do
        model_suffix=${model_path#*/}
        for dataset in hotpotqa 2wikimultihopqa musique nq trivia squad
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