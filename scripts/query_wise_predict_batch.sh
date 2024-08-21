#!/bin/bash

# for dataset in musique
# do
#     if [ $dataset == "ms_marco" ]; then
#         retrieval_top_n=3
#     else
#         retrieval_top_n=5
#     fi

#     generator_model_name=gpt-4o-mini-2024-07-18

#     echo "dataset: $dataset"
#     python estimator/dataset_construction/query_wise_predict.py \
#         --input_directory source_dataset \
#         --output_directory predictions/$generator_model_name \
#         --dataset $dataset \
#         --dataset_type train_18000_subsampled \
#         --retrieval_top_n $retrieval_top_n \
#         --batch_size 1 \
#         --temperature 0 \
#         --use_chat_template \
#         --generator_model_name $generator_model_name \
#         --openai_batch_api upload
# done





generator_model_name=gpt-4o-mini-2024-07-18
echo "Check on $generator_model_name"
for batch_id in batch_A8SOq92taIiUhdg9GY2bgnTp batch_VvfQVC9cHOCA8QoV4sWVmFa0 batch_eAzVj9wI2tfg4b7SIJK7WfPn
do

    python estimator/dataset_construction/query_wise_predict.py \
        --output_directory predictions/$generator_model_name \
        --generator_model_name $generator_model_name \
        --dataset_type train_18000_subsampled \
        --openai_batch_api check \
        --openai_batch_id $batch_id
done






# for file_id in 
# do
#     generator_model_name=gpt-4o-mini-2024-07-18
#     dataset=trivia
#     python estimator/dataset_construction/query_wise_predict.py \
#         --output_directory predictions/$generator_model_name \
#         --dataset $dataset \
#         --dataset_type train_18000_subsampled \
#         --generator_model_name $generator_model_name \
#         --openai_batch_api analysis \
#         --openai_file_id $file_id
# done
# for file_id in
# do
#     generator_model_name=gpt-4o-mini-2024-07-18

#     python estimator/dataset_construction/query_wise_predict.py \
#         --output_directory predictions/$generator_model_name \
#         --dataset squad \
#         --dataset_type train_18000_subsampled \
#         --generator_model_name $generator_model_name \
#         --openai_batch_api analysis \
#         --openai_file_id $file_id
# done
 



