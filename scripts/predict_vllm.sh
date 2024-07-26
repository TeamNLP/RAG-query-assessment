#!/bin/bash
for model_path in meta-llama/Meta-Llama-3.1-8B meta-llama/Meta-Llama-3.1-8B-Instruct meta-llama/Meta-Llama-3-8B meta-llama/Meta-Llama-3-8B-Instruct
do
    model_suffix=${model_path#*/}
    for dataset in nq trivia
    # for dataset in hotpotqa 2wikimultihopqa musique nq trivia squad
    do
        echo "model_path: $model_path, dataset: $dataset"
        python experiments/predict_vllm.py \
            --output_directory "predictions_$model_suffix" \
            --dataset $dataset \
            --dataset_type test_subsampled \
            --retrieval_top_n 3 \
            --batch_size 128 \
            --temperature 0 \
            --generator_model_name $model_path
    done
done