#!/bin/bash
python experiments/main.py \
    --retrieval_corpus_name wiki \
    --retrieval_top_n 3 \
    --generator_model_name google/gemma-1.1-2b-it \
    --debug
