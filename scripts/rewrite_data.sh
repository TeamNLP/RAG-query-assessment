#!/bin/bash

for rewrite_method in method1 method2 method3 method4 method5
do
    echo "Rewrite Method: $rewrite_method"
    for dataset in ms_marco ambig_qa
    do
        python experiments/rewriting/rewrite.py \
            --rewrite_method $rewrite_method \
            --rewriter_model_name gpt-4o-mini-2024-07-18 \
            --rewriter_max_new_tokens 200 \
            --output_directory "rewritten_data/$rewrite_method" \
            --dataset $dataset \
            --debug \
            --dataset_type test_subsampled
    done
done