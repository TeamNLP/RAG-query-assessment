#!/bin/bash

python classifier/fine-tuning/run.py \
    --model_path google/flan-t5-xl \
    --dataset_path classifier/ft_dataset/ft_dataset.jsonl