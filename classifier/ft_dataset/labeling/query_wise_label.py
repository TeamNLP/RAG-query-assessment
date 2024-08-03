import argparse
import json
import os
from statistics import mean

from typing import List, Dict
from lib import read_jsonl, dump_json

parser = argparse.ArgumentParser()
parser.add_argument('--eval_result_path', type=str, default="classifier/ft_dataset/evaluation/result", help="A directory path where the evaluation results are stored")
parser.add_argument('--performance_threshold', type=float, default=0.5, help="A threshold for performance of the model")
parser.add_argument('--result_file_path', type=str, default="classifier/ft_dataset/ft_dataset.json", help="File Path to save query labeling results")

args = parser.parse_args()

DATASET_LIST = [
    "hotpotqa-train-distractor-stratified-v1.0",
    "msmarco-train-v2.1-stratified-v1.0",
    "squad-train-v2.0-stratified_v1.0"
]


def categorize_score(
    eval_result: List[Dict], 
    threshold: float
) -> List[Dict]:
    categorized_result = []
    for result_dict in eval_result:
        large_model_performance = mean([result_dict["large_evaluation"]["f1_score"], result_dict["large_evaluation"]["bertscore"]])
        small_model_performance = mean([result_dict["small_evaluation"]["f1_score"], result_dict["small_evaluation"]["bertscore"]])

        if large_model_performance >= threshold:
            result_dict["large_evaluation"]["performance"] = "High"
        else:
            result_dict["large_evaluation"]["performance"] = "Low"

        if small_model_performance >= threshold:
            result_dict["small_evaluation"]["performance"] = "High"
        else:
            result_dict["small_evaluation"]["performance"] = "Low"
        
        categorized_result.append(result_dict)
    
    return categorized_result


def query_wise_label(
    categorized_result: List[Dict]
) -> List[Dict]:
    labeled_dataset = []

    for result_dict in categorized_result:
        # TODO: Edit this function after labeling strategies have been set up
        if result_dict["large_evaluation"]["performance"] == "High" and result_dict["small_evaluation"]["performance"] == "High":
            rewriting_label = False
        else:
            rewriting_label = True

        datum = {}
        datum["id"] = str(result_dict["query_id"])
        datum["text"] = result_dict["query"]
        datum["label"] = rewriting_label
        labeled_dataset.append(datum)

    return labeled_dataset


def main(args):
    eval_result_instances = []
    for data_name in DATASET_LIST:
        instances = read_jsonl(os.path.join(args.eval_result_path, f"{data_name}_query-wise_eval_result.jsonl"))
        eval_result_instances.extend(instances)
    print(f"Total length of dataset: {len(eval_result_instances)}")

    categorized_result = categorize_score(eval_result_instances, args.performance_threshold)
    labeled_dataset = query_wise_label(categorized_result)
    dump_json(labeled_dataset, args.result_file_path)


if __name__ == "__main__":
    main(args)