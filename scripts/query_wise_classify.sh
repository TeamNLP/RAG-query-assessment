#!/bin/bash

# for classifier_model_path in oneonlee/flan-t5-small-query_level_estimator_3e-5-new_0.5 oneonlee/flan-t5-small-query_level_estimator_3e-5-new_0.6
for classifier_model_path in oneonlee/flan-t5-xl-query_level_estimator_5e-5
# for classifier_model_path in oneonlee/flan-t5-small-query_level_estimator_3e-5 oneonlee/flan-t5-xl-query_level_estimator
do
    classifier_suffix=${classifier_model_path#*/}
    echo "classifier_model_path: $classifier_model_path"

    for dataset in nq trivia squad hotpotqa 2wikimultihopqa musique 
    do
        python experiments/classify.py \
            --classifier_model_name $classifier_model_path \
            --rewrite_method method3 \
            --output_directory "rewritten_data/$classifier_suffix" \
            --dataset $dataset \
            --dataset_type test_subsampled
    done
done