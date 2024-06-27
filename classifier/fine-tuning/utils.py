from datasets import load_dataset
from transformers import (
    CONFIG_MAPPING,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    # SchedulerType,
    # get_scheduler,
)


def load_model(args):
    """
    Load pretrained model and tokenizer

    In distributed training:
    The .from_pretrained methods guarantee that only one local process can concurrently download model & vocab.
    """

    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

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
        model = AutoModelForSeq2SeqLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForSeq2SeqLM.from_config(config)
        
    return model, tokenizer


def load_ft_dataset(args):
    """
    Get the fine-tuning datasets: you can provide JSON training and evaluation files
    or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    (the dataset will be downloaded automatically from the datasets Hub).
    
    In distributed training:
    The load_dataset function guarantee that only one local process can concurrently download the dataset.
    """
    
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
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

    return raw_datasets