import json

import numpy as np
import torch
from datasets import load_dataset
from loguru import logger
from sklearn.metrics import precision_recall_fscore_support
from transformers import (
    CONFIG_MAPPING,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    # SchedulerType,
    # get_scheduler,
)


label2id = {False: 0, True: 1}
id2label = {id: label for label, id in label2id.items()}

def load_model(args):
    """
    Load pretrained model and tokenizer

    In distributed training:
    The .from_pretrained methods guarantee that only one local process can concurrently download model & vocab.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.config_name:
        if args.model_type == "AutoModelForSequenceClassification":
            if args.do_train:
                config = AutoConfig.from_pretrained(
                    args.config_name, 
                    num_labels=len(label2id),
                    id2label=id2label,
                    label2id=label2id,
                    hidden_dropout_prob=args.hidden_dropout_prob,  # Dropout
                    attention_probs_dropout_prob=args.attention_probs_dropout_prob  # Attention Dropout
                )
            else:
                config = AutoConfig.from_pretrained(
                    args.config_name,
                    num_labels=2,
                )
        elif args.model_type == "AutoModelForSeq2SeqLM":
            config = AutoConfig.from_pretrained(args.config_name)
        else:
            raise ValueError("`model_type` should be `AutoModelForSequenceClassification` or `AutoModelForSeq2SeqLM`")
    elif args.model_name_or_path:
        if args.model_type == "AutoModelForSequenceClassification":
            if args.do_train:
                config = AutoConfig.from_pretrained(
                    args.model_name_or_path, 
                    num_labels=len(label2id),
                    id2label=id2label,
                    label2id=label2id,
                    hidden_dropout_prob=args.hidden_dropout_prob,  # Dropout
                    attention_probs_dropout_prob=args.attention_probs_dropout_prob  # Attention Dropout
                )
            else:
                config = AutoConfig.from_pretrained(
                    args.model_name_or_path,
                    num_labels=2,
                )
        elif args.model_type == "AutoModelForSeq2SeqLM":
            config = AutoConfig.from_pretrained(args.model_name_or_path)
        else:
            raise ValueError("`model_type` should be `AutoModelForSequenceClassification` or `AutoModelForSeq2SeqLM`")
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    global tokenizer
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer)
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if args.model_name_or_path:
        if args.model_type == "AutoModelForSequenceClassification":
            if args.do_train:
                model = AutoModelForSequenceClassification.from_pretrained(
                    args.model_name_or_path,
                    from_tf=bool(".ckpt" in args.model_name_or_path),
                    config=config,
                ).to(device)
            else:
                model = AutoModelForSequenceClassification.from_pretrained(
                    args.model_name_or_path,
                    from_tf=bool(".ckpt" in args.model_name_or_path),
                    config=config,
                ).to(device)
        elif args.model_type == "AutoModelForSeq2SeqLM":
            model = AutoModelForSeq2SeqLM.from_pretrained(
                args.model_name_or_path,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
            ).to(device)
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForSeq2SeqLM.from_config(config).to(device)
        
    return model, tokenizer


def tokenize_function(examples) -> dict:
    """
    Tokenize the `text` column in the dataset
    """
    return tokenizer(examples["text"], padding="max_length", truncation=True)


def one_hot_encode_labels(examples, num_labels):
    """
    One-hot encode the `label` column in the dataset.
    """
    # Convert labels to integer type before one-hot encoding
    labels = torch.tensor(examples['label'], dtype=torch.int64)
    one_hot_labels = torch.nn.functional.one_hot(labels, num_classes=num_labels).float()
    
    # Convert back to list format for compatibility with HF datasets
    examples['label'] = one_hot_labels.tolist()
    return examples


def load_ft_dataset(args):
    """
    Get the fine-tuning datasets: you can provide JSON training and evaluation files
    or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    (the dataset will be downloaded automatically from the datasets Hub).
    
    In distributed training:
    The load_dataset function guarantees that only one local process can concurrently download the dataset.
    """

    if args.model_type == "AutoModelForSequenceClassification":
        if args.dataset_name is not None:
            # Downloading and loading a dataset from the hub.
            if args.dataset_config_name is None:
                raw_datasets = load_dataset(args.dataset_name)
            else:
                raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
        else:
            data_files = {}
            if args.train_file is not None:
                extension = args.train_file.split(".")[-1].lower()
                assert extension == "json", "`train_file` should be a json file."
                data_files["train"] = args.train_file
            if args.validation_file is not None:
                extension = args.validation_file.split(".")[-1].lower()
                assert extension == "json", "`validation_file` should be a json file."
                data_files["validation"] = args.validation_file
            if args.test_file is not None:
                extension = args.test_file.split(".")[-1].lower()
                assert extension == "json", "`test_file` should be a json file."
                data_files["test"] = args.test_file
            raw_datasets = load_dataset(
                extension,
                data_files=data_files,
            )

        # Apply the tokenization
        tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
        
        if args.do_train:
            pass
        else:
            # Apply the one-hot encoding to the labels
            num_labels = 2  # Assuming binary classification
            tokenized_datasets = tokenized_datasets.map(lambda x: one_hot_encode_labels(x, num_labels), batched=True)
        
        return tokenized_datasets

def compute_metrics(eval_pred) -> dict:
    """
    Compute metrics for evaluation
    """

    logits, labels = eval_pred

    if isinstance(logits, tuple):  # if the model also returns hidden_states or attentions
        logits = logits[0]

    # Single label (binary or multiclass) classification
    predictions = np.argmax(logits, axis=-1)
    average = "binary"

    # Compute metrics with appropriate average method
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average=average
    )
    
    return {"precision": precision, "recall": recall, "f1": f1}


def write_json(instance, file_path):
    with open(file_path, "w") as file:
        json.dump(instance, file)


