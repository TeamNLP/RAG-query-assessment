import argparse
import json, jsonlines
import os
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple, Union

from lib import write_jsonl
from metric.f1_bertscore import F1BertScoreMetric
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--bertscore_model_type', type=str, default="roberta-large", help="A name or a model path for BERT Score, used to load `transformers` pretrained model. Currently, the best model is `microsoft/deberta-xlarge-mnli`, please consider using it instead of the default `roberta-large` in order to have the best correlation with human evaluation. If you want the scores to correlate better with human scores, please consider using `microsoft/deberta-xlarge-mnli` or `microsoft/deberta-large-mnli` (faster).")
parser.add_argument('--predictions_result_dir', type=str, default="classifier/ft_dataset/predictions", help="A directory path of the model's prediction results")
parser.add_argument('--output_result_dir', type=str, default="classifier/ft_dataset/evaluation/result", help="A directory path to save evaluation results")

args = parser.parse_args()

DATASET_LIST = [
    "hotpotqa-train-distractor-stratified-v1.1",
    "msmarco-train-v2.1-stratified-v1.1",
    "squad-train-v2.0-stratified-v1.1",
    "asqa-train-stratified-v1.1"
]


def answer_extractor(potentially_cot: str) -> str:
    # In a few experiments I forgot the configuring the answer extractor part
    # and so the final answer is a cot chain instead. Instead of having to do
    # all those exps again, I'm just doing answer_extraction here. This needs
    # to be fixed later though.

    if potentially_cot.startswith('"') and potentially_cot.endswith('"'):
        potentially_cot = potentially_cot[1:-1]

    cot_regex = re.compile(".* answer is\s?:? ((.|\n)*)\\.?")
    match = cot_regex.match(potentially_cot)
    if match:
        output = match.group(1)
        if output.endswith("."):
            output = output[:-1]
    else:
        output = potentially_cot

    return output.strip()


def load_result_dicts(
    file_path: str,
    large_prediction_column: str,
    small_prediction_column: str,
    ground_truth_column: str = 'ground_truth',
    query_column: str = 'query',
    query_id_column: str = 'id'
) -> Tuple[Dict, Dict]:
    id_to_query = {}
    id_to_ground_truths = {}
    id_to_large_predictions = {}
    id_to_small_predictions = {}

    with jsonlines.open(file_path, 'r') as input_file:
        for line in input_file:
            qid = line[query_id_column]
            query = line[query_column]
            answer = line[ground_truth_column]
            large_pred = line[large_prediction_column]
            small_pred = line[small_prediction_column]

            id_to_query[qid] = query
            id_to_ground_truths[qid] = answer
            id_to_large_predictions[qid] = large_pred
            id_to_small_predictions[qid] = small_pred
    return id_to_query, id_to_ground_truths, id_to_large_predictions, id_to_small_predictions


def calculate_acc(prediction, ground_truth):
    for gt in ground_truth:
        if gt in prediction:
            return 1
    return 0


def query_wise_evaluate(
    data_name: str, 
    bertscore_model_type: str,
    predictions_result_dir: str,
    output_result_dir: str
) -> None:
    large_metrics = [F1BertScoreMetric(bertscore_model_type=bertscore_model_type)]
    small_metrics = [F1BertScoreMetric(bertscore_model_type=bertscore_model_type)]

    predictions_result_path = os.path.join(predictions_result_dir, f"predictions-{data_name}.jsonl")
    id_to_query, id_to_ground_truths, id_to_large_predictions, id_to_small_predictions = load_result_dicts(
        file_path = predictions_result_path,
        large_prediction_column="prediction_by_meta-llama-3-70b-instruct",
        small_prediction_column="prediction_by_meta-llama-3-8b-instruct"
    )

    evaluation_results = []
    for id_ in tqdm(set(id_to_ground_truths.keys()), desc=f"query-wise evaluation on {data_name}"):
        query = id_to_query[id_]
        ground_truth = id_to_ground_truths[id_]
        large_prediction = id_to_large_predictions[id_]
        small_prediction = id_to_small_predictions[id_]

        assert isinstance(large_prediction, (str, list)) and isinstance(small_prediction, (str, list))
        if isinstance(ground_truth, str):
            ground_truth = [ground_truth]
        if isinstance(large_prediction, str):
            large_prediction = [large_prediction]
        if isinstance(small_prediction, str):
            small_prediction = [small_prediction]

        assert isinstance(large_prediction, (list, tuple)) and isinstance(small_prediction, (list, tuple))
        large_prediction = [str(e) for e in large_prediction]
        small_prediction = [str(e) for e in small_prediction]

        large_prediction = [answer_extractor(_prediction) for _prediction in large_prediction]  # Temporary.
        small_prediction = [answer_extractor(_prediction) for _prediction in small_prediction]  # Temporary.

        assert len(large_prediction) == 1, f"len(large_prediction) == {len(large_prediction)}, large_prediction: {large_prediction}"
        assert len(small_prediction) == 1, f"len(small_prediction) == {len(small_prediction)}, small_prediction: {small_prediction}"

        large_f1_score, large_bertscore = large_metrics[0](large_prediction[0], ground_truth)
        small_f1_score, small_bertscore = small_metrics[0](small_prediction[0], ground_truth)

        evaluation_result = {
            "query_id": id_,
            "query": query,
            "large_evaluation": {
                "f1_score": large_f1_score,
                "bertscore": large_bertscore
            },
            "small_evaluation": {
                "f1_score": small_f1_score,
                "bertscore": small_bertscore
            },
            "ground_truth": ground_truth,
            "large_prediction": large_prediction,
            "small_prediction": small_prediction
        }
        evaluation_results.append(evaluation_result)

    write_jsonl(evaluation_results, os.path.join(output_result_dir, f"{data_name}_query-wise_eval_result.jsonl"))


def main(args):
    for data_name in DATASET_LIST:
        query_wise_evaluate(
            data_name=data_name,
            bertscore_model_type=args.bertscore_model_type,
            predictions_result_dir=args.predictions_result_dir,
            output_result_dir=args.output_result_dir
        )


if __name__ == "__main__":
    main(args)