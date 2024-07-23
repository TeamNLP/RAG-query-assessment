#!/bin/bash

for rewrite_method in method1 method2 method3 method4 method5
do
    python experiments/rewrite.py \
        --rewrite_method $rewrite_method \
        --rewriter_model_name gpt-4o-mini-2024-07-18 \
        --query "when was the first robot used in surgery" \
        --debug
done