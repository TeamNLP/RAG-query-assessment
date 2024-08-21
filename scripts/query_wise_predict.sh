#!/bin/bash

# for dataset in ms_marco nq trivia squad hotpotqa 2wikimultihopqa musique
for dataset in nq 
do
    if [ $dataset == "ms_marco" ]; then
        retrieval_top_n=3
    else
        retrieval_top_n=5
    fi

    generator_model_name=gpt-4o-mini-2024-07-18

    echo "dataset: $dataset, generator_model_name: $generator_model_name"
    python estimator/dataset_construction/query_wise_predict.py \
        --input_directory source_dataset \
        --output_directory predictions/$generator_model_name \
        --dataset $dataset \
        --dataset_type train_18000_subsampled \
        --retrieval_top_n $retrieval_top_n \
        --batch_size 1 \
        --temperature 0 \
        --use_chat_template \
        --generator_model_name $generator_model_name
done

