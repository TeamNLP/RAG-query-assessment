#!/bin/bash

for dataset in hotpotqa 2wikimultihopqa musique nq trivia squad
do
python experiments/evaluate.py \
    --dataset $dataset \
    --dataset_type test_subsampled
done