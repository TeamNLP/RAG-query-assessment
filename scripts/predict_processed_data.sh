#!/bin/bash

# use template with instruction
for rewrite_method in method1 method2 method3 method4 method5
do
    for model_path in meta-llama/Meta-Llama-3.1-8B meta-llama/Meta-Llama-3.1-8B-Instruct meta-llama/Meta-Llama-3-8B meta-llama/Meta-Llama-3-8B-Instruct
    do
        model_suffix=${model_path#*/}
        for dataset in nq trivia
        # for dataset in hotpotqa 2wikimultihopqa musique nq trivia squad
        do
            echo "model_path: $model_path, dataset: $dataset"
            python experiments/predict.py \
                --output_directory "$rewrite_method"_predictions_wo_"$model_suffix" \
                --dataset $dataset \
                --dataset_type test_subsampled \
                --retrieval_top_n 5 \
                --batch_size 128 \
                --temperature 0 \
                --generator_model_name $model_path
        done
    done
done

# use_template_wo_instruction
for rewrite_method in method1 method2 method3 method4 method5
do
    for model_path in meta-llama/Meta-Llama-3.1-8B meta-llama/Meta-Llama-3.1-8B-Instruct meta-llama/Meta-Llama-3-8B meta-llama/Meta-Llama-3-8B-Instruct
    do
        model_suffix=${model_path#*/}
        for dataset in nq trivia
        # for dataset in hotpotqa 2wikimultihopqa musique nq trivia squad
        do
            echo "model_path: $model_path, dataset: $dataset"
            python experiments/predict.py \
                --output_directory "$rewrite_method"_predictions_wo_"$model_suffix" \
                --dataset $dataset \
                --dataset_type test_subsampled \
                --retrieval_top_n 5 \
                --batch_size 128 \
                --temperature 0 \
                --use_template_wo_instruction \
                --generator_model_name $model_path
        done
    done
done

# use_chat_template
for rewrite_method in method1 method2 method3 method4 method5
do
    for model_path in meta-llama/Meta-Llama-3.1-8B meta-llama/Meta-Llama-3.1-8B-Instruct meta-llama/Meta-Llama-3-8B meta-llama/Meta-Llama-3-8B-Instruct
    do
        model_suffix=${model_path#*/}
        for dataset in nq trivia
        # for dataset in hotpotqa 2wikimultihopqa musique nq trivia squad
        do
            echo "model_path: $model_path, dataset: $dataset"
            python experiments/predict.py \
                --output_directory "$rewrite_method"_predictions_chat_"$model_suffix" \
                --dataset $dataset \
                --dataset_type test_subsampled \
                --retrieval_top_n 5 \
                --batch_size 128 \
                --temperature 0 \
                --use_chat_template \
                --generator_model_name $model_path
        done
    done
done