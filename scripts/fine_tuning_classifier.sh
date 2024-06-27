#!/bin/bash

python classifier/fine-tuning/run.py \
    --model_name_or_path google/flan-t5-xl \
    --train_file classifier/ft_dataset/ft_dataset.json