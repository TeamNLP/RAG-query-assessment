#!/bin/bash

# use template with instruction
for model_path in meta-llama/Meta-Llama-3.1-8B meta-llama/Meta-Llama-3.1-8B-Instruct meta-llama/Meta-Llama-3-8B meta-llama/Meta-Llama-3-8B-Instruct
do
    model_suffix=${model_path#*/}
    for dataset in hotpotqa 2wikimultihopqa musique nq trivia squad
    do
        echo "model_path: $model_path, dataset: $dataset"
        python experiments/predict.py \
            --output_directory predictions/original_query/$model_suffix \
            --dataset $dataset \
            --dataset_type test_subsampled \
            --retrieval_top_n 5 \
            --batch_size 128 \
            --temperature 0 \
            --generator_model_name $model_path
    done
done

# use_template_wo_instruction
for model_path in meta-llama/Meta-Llama-3.1-8B meta-llama/Meta-Llama-3.1-8B-Instruct meta-llama/Meta-Llama-3-8B meta-llama/Meta-Llama-3-8B-Instruct
do
    model_suffix=${model_path#*/}
    for dataset in hotpotqa 2wikimultihopqa musique nq trivia squad
    do
        echo "model_path: $model_path, dataset: $dataset"
        python experiments/predict.py \
            --output_directory predictions/original_query/wo_$model_suffix \
            --dataset $dataset \
            --dataset_type test_subsampled \
            --retrieval_top_n 5 \
            --batch_size 128 \
            --temperature 0 \
            --use_template_wo_instruction \
            --generator_model_name $model_path
    done
done

# use_chat_template
for model_path in meta-llama/Meta-Llama-3.1-8B-Instruct meta-llama/Meta-Llama-3-8B-Instruct gpt-4o-mini-2024-07-18
do
    model_suffix=${model_path#*/}
    for dataset in hotpotqa 2wikimultihopqa musique nq trivia squad
    do
        echo "model_path: $model_path, dataset: $dataset"
        python experiments/predict.py \
            --output_directory predictions/original_query/chat_$model_suffix \
            --dataset $dataset \
            --dataset_type test_subsampled \
            --retrieval_top_n 5 \
            --batch_size 128 \
            --temperature 0 \
            --use_chat_template \
            --generator_model_name $model_path
    done
done