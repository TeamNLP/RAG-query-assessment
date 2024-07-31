#!/bin/bash


for rewrite_method in method1 method2 method3 method4 method5
do
    base_directory="predictions/$rewrite_method"

    # use template with instruction
    output_directory_prefix=""
    for model_path in meta-llama/Meta-Llama-3.1-8B meta-llama/Meta-Llama-3.1-8B-Instruct meta-llama/Meta-Llama-3-8B meta-llama/Meta-Llama-3-8B-Instruct
    do
        model_suffix=${model_path#*/}
        for dataset in hotpotqa 2wikimultihopqa musique nq trivia squad
        do
            python experiments/evaluate.py \
                --input_directory "$base_directory"/"$output_directory_prefix""$model_suffix" \
                --dataset $dataset \
                --dataset_type test_subsampled
        done
    done
    python experiments/extract_evaluation_results.py \
        --base_directory $base_directory 

    # use_template_wo_instruction
    output_directory_prefix="wo_"
    for model_path in meta-llama/Meta-Llama-3.1-8B meta-llama/Meta-Llama-3-8B 
    do
        model_suffix=${model_path#*/}
        for dataset in hotpotqa 2wikimultihopqa musique nq trivia squad
        do
            python experiments/evaluate.py \
                --input_directory "$base_directory"/"$output_directory_prefix""$model_suffix" \
                --dataset $dataset \
                --dataset_type test_subsampled
        done
    done
    python experiments/extract_evaluation_results.py \
        --base_directory $base_directory \
        --output_directory_prefix $output_directory_prefix


    # use_chat_template
    output_directory_prefix="chat_"
    for model_path in meta-llama/Meta-Llama-3.1-8B-Instruct meta-llama/Meta-Llama-3-8B-Instruct gpt-4o-mini-2024-07-18
    do
        model_suffix=${model_path#*/}
        for dataset in hotpotqa 2wikimultihopqa musique nq trivia squad
        do
            python experiments/evaluate.py \
                --input_directory "$base_directory"/"$output_directory_prefix""$model_suffix" \
                --dataset $dataset \
                --dataset_type test_subsampled
        done
    done
    python experiments/extract_evaluation_results.py \
        --base_directory $base_directory \
        --output_directory_prefix $output_directory_prefix
done