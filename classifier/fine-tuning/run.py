import argparse

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from utils import *

parser = argparse.ArgumentParser()
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
    '--dataset_path', 
    type=str, 
    default="../ft_dataset/ft_dataset.jsonl", 
    help="File Path of dataset for fine-tuning"
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

args = parser.parse_args()


def load_dataset(dataset_path):
    pass

    return dataset


if __name__ == "__main__":
    # load model and tokenizer
    model, tokenizer = load_model(args)
    dataset = load_dataset(args.dataset_path)


