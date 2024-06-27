"""
Reference
https://github.com/huggingface/transformers/blob/main/examples/pytorch/question-answering/run_seq2seq_qa.py
https://github.com/starsuzi/Adaptive-RAG/blob/main/classifier/run_classifier.py
https://github.com/starsuzi/Adaptive-RAG/blob/main/classifier/utils.py
https://github.com/starsuzi/Adaptive-RAG/blob/main/classifier/run/run_large_train_gpt.sh
"""

import argparse

from loguru import logger
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from utils import load_model, load_ft_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a seq2seq based model")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models. e.g., `google/flan-t5-xl`",
        required=True,
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--train_file", 
        type=str, 
        default=None, 
        help="A json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", 
        type=str, 
        default=None, 
        help="A json file containing the validation data."
    )
    parser.add_argument(
        "--test_file", 
        type=str, 
        default=None, 
        help="A json file containing the test data."
    )
    parser.add_argument(
        "--do_train", 
        action="store_true", 
        help="Whether to run training."
    )
    parser.add_argument(
        "--do_eval", 
        action="store_true", 
        help="Whether to run eval on the dev set."
    )
    parser.add_argument(
        "--do_predict", 
        action="store_true", 
        help="Whether to run predictions on the test set."
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42, 
        help="A seed for reproducible training."
    )

    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the 🤗 datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the 🤗 datasets library).",
    )

    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    logger.info(args)

    # load model and tokenizer
    model, tokenizer = load_model(args)
    raw_datasets = load_ft_dataset(args)

    if args.do_train:
        pass

    if args.do_eval:
        pass
    
    if args.do_predict:
        pass


if __name__ == "__main__":
    main()