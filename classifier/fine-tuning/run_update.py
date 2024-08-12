"""
Reference
https://github.com/huggingface/transformers/blob/main/examples/pytorch/question-answering/run_seq2seq_qa.py
https://github.com/starsuzi/Adaptive-RAG/blob/main/classifier/run_classifier.py
https://github.com/starsuzi/Adaptive-RAG/blob/main/classifier/utils.py
https://github.com/starsuzi/Adaptive-RAG/blob/main/classifier/run/run_large_train_gpt.sh
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from huggingface_hub import HfFolder
from loguru import logger
from transformers import Trainer, TrainingArguments
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import EarlyStoppingCallback
from utils import load_model, load_ft_dataset, compute_metrics


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
        "--model_type",
        type=str,
        default="AutoModelForSequenceClassification",
        help="A type of model to fine-tune supported by 🤗 transformers. Now only supported for `AutoModelForSequenceClassification`."
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
        default=13370, 
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

    # training_args
    parser.add_argument(
        "--num_train_epochs", 
        type=int, 
        default=2, 
        help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--use_earlystopping",
        action="store_true",
        help="If passed, will use a EarlyStoppingCallback.",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=2,
        help="To stop training when the specified metric worsens for `early_stopping_patience` evaluation calls.",
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=None, 
        help="The output directory where the model predictions and checkpoints will be written."
    )
    parser.add_argument(
        "--logging_strategy", 
        type=str, 
        default="steps", 
        help="""The logging strategy to adopt during training. Possible values are:
- `"no"`: No logging is done during training.
- `"epoch"`: Logging is done at the end of each epoch.
- `"steps"`: Logging is done every `logging_steps`."""
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--use_fp16", 
        action="store_true", 
        help="Whether to use fp16 (mixed) precision instead of 32-bit."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0,  # L2 regularization
        help="Weight decay to apply (if any).",
    )
    parser.add_argument(
        "--label_smoothing_factor",
        type=float,
        default=0,  # Label smoothing
        help="The label smoothing factor to use (if any).",
    )
    parser.add_argument(
        "--hidden_dropout_prob",
        type=float,
        default=0,
        help="Dropout probability for fully connected layers in the transformer.",
    )
    parser.add_argument(
        "--attention_probs_dropout_prob",
        type=float,
        default=0,
        help="Dropout probability for attention probabilities.",
    )
    parser.add_argument(
        "--evaluation_strategy", 
        type=str, 
        default="epoch", 
        help="""The evaluation strategy to adopt during evaluation. Possible values are:
- `"no"`: No save is done during training.
- `"epoch"`: Save is done at the end of each epoch.
- `"steps"`: Save is done every `save_steps`."""
    )
    parser.add_argument(
        "--save_strategy", 
        type=str, 
        default="epoch", 
        help="""The checkpoint save strategy to adopt during training. Possible values are:
- `"no"`: No save is done during training.
- `"epoch"`: Save is done at the end of each epoch.
- `"steps"`: Save is done every `save_steps`.
If `"epoch"` or `"steps"` is chosen, saving will also be performed at the very end of training, always."""
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=2,
        help="If a value is passed, will limit the total amount of checkpoints. Deletes the older checkpoints in `output_dir`.",
    )
    parser.add_argument(
        "--load_best_model_at_end", 
        action="store_true", 
        help="Whether or not to load the best model found during training at the end of training. When this option is enabled, the best checkpoint will always be saved."
    )
    parser.add_argument(
        "--push_to_hub", 
        action="store_true", 
        help="Whether or not to push the model to the 🤗 Model Hub."
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument(
        "--hub_token", 
        type=str,
        default=None,
        help="The write token to use to push to the 🤗 Model Hub."
    )

    args = parser.parse_args()

    return args


def plot_metrics(log_history, output_dir):
    """
    Plot train loss, validation loss, and validation F1 score.
    """
    epochs = []
    train_losses = []
    eval_losses = []
    eval_f1 = []

    for entry in log_history:
        if 'epoch' in entry:
            epoch = entry['epoch']
            if 'loss' in entry:
                if epoch not in epochs:
                    epochs.append(epoch)
                train_losses.append(entry['loss'])
            if 'eval_loss' in entry:
                if epoch not in epochs:
                    epochs.append(epoch)
                eval_losses.append(entry['eval_loss'])
            if 'eval_f1' in entry:
                if epoch not in epochs:
                    epochs.append(epoch)
                eval_f1.append(entry['eval_f1'])

    # Ensure the lists are sorted by epoch
    sorted_indices = sorted(range(len(epochs)), key=lambda k: epochs[k])
    epochs = [epochs[i] for i in sorted_indices]
    train_losses = [train_losses[i] for i in sorted_indices]
    eval_losses = [eval_losses[i] for i in sorted_indices]
    eval_f1 = [eval_f1[i] for i in sorted_indices]

    # Create figure and axis
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot train loss
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss')
    ax1.plot(epochs, eval_losses, 'r--', label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend(loc='upper left')
    
    # Plot validation F1 on a separate figure
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    ax2.plot(epochs, eval_f1, 'g-', label='Validation F1')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation F1')
    ax2.legend(loc='upper left')

    # Save plots
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    fig.savefig(os.path.join(output_dir, 'losses.png'))
    fig2.savefig(os.path.join(output_dir, 'f1_scores.png'))
    plt.close(fig)
    plt.close(fig2)
    

def main():
    args = parse_args()
    logger.info(args)

    # load model and tokenizer
    model, tokenizer = load_model(args)
    tokenized_dataset = load_ft_dataset(args)

    if args.hub_token is not None and args.hub_token.lower() == "none":
        args.hub_token = None
    if args.hub_token is not None:
        user_hub_token = args.hub_token
    else:
        try:
            user_hub_token = HfFolder.get_token()
        except:
            user_hub_token = None
    
    if user_hub_token is None:
        args.hub_model_id = None
        args.push_to_hub = False

    if args.model_type == "AutoModelForSequenceClassification":
        # Define training args
        training_args = TrainingArguments(
            num_train_epochs=args.num_train_epochs,
            output_dir=args.output_dir,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            fp16=args.use_fp16,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,  # L2 regularization

            # logging & evaluation strategies
            logging_dir=f"{args.output_dir}/logs",
            logging_strategy=args.logging_strategy,  # "epoch"
            evaluation_strategy=args.evaluation_strategy,  # "epoch"
            save_strategy=args.save_strategy,
            save_total_limit=args.save_total_limit,
            load_best_model_at_end=args.load_best_model_at_end,
            metric_for_best_model="eval_f1",
            greater_is_better=True,

            label_smoothing_factor=args.label_smoothing_factor,  # Label smoothing
        )

        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=tokenized_dataset["train"] if args.do_train else None,
            eval_dataset=tokenized_dataset["validation"] if args.do_eval else None,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)] if args.use_earlystopping else None,
        )
    else:
        # Define Seq2Seq Training args
        training_args = Seq2SeqTrainingArguments(
            num_train_epochs=args.num_train_epochs,
            output_dir=args.output_dir,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            fp16=args.use_fp16,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,  # L2 regularization

            # logging & evaluation strategies
            logging_dir=f"{args.output_dir}/logs",
            logging_strategy=args.logging_strategy,  # "epoch"
            evaluation_strategy=args.evaluation_strategy,  # "epoch"
            save_strategy=args.save_strategy,
            save_total_limit=args.save_total_limit,
            load_best_model_at_end=args.load_best_model_at_end,
            metric_for_best_model="eval_f1",
            greater_is_better=True,

            predict_with_generate=True,
            generation_max_length=512,
            generation_num_beams=5,

            label_smoothing_factor=args.label_smoothing_factor,  # Label smoothing
        )

        trainer = Seq2SeqTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=tokenized_dataset["train"] if args.do_train else None,
            eval_dataset=tokenized_dataset["validation"] if args.do_eval else None,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)] if args.use_earlystopping else None,
        )

    if args.do_train:
        logger.info("*** Train ***")
        trainer.train()
        trainer.save_model(args.output_dir)
        if trainer.is_world_process_zero() and args.push_to_hub:
            trainer.push_to_hub()

    if args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        plot_metrics(trainer.state.log_history, args.output_dir)

    if args.do_predict:
        logger.info("*** Predict ***")
        predictions = trainer.predict(tokenized_dataset["test"])
        metrics = predictions.metrics
        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)


if __name__ == "__main__":
    main()