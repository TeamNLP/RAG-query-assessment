import argparse
import json
import os
from statistics import mean

from typing import List, Dict
from lib import read_jsonl, dump_json
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=13370, help="A seed for reproducible training.")
parser.add_argument('--eval_result_path', type=str, default="classifier/ft_dataset/evaluation/result", help="A directory path where the evaluation results are stored")
parser.add_argument('--large_performance_threshold', type=float, default=0.5, help="A threshold for performance of the Large model")
parser.add_argument('--small_performance_threshold', type=float, default=0.5, help="A threshold for performance of the Small model")
parser.add_argument('--train_ratio', type=float, default=60, help="A ratio of train set")
parser.add_argument('--valid_ratio', type=float, default=20, help="A ratio of validation set")
parser.add_argument('--test_ratio', type=float, default=20, help="A ratio of test set")
parser.add_argument('--result_dataset_path', type=str, default="classifier/ft_dataset", help="A directory path to save query labeling results")

args = parser.parse_args()

DATASET_LIST = [
    "hotpotqa-train-distractor-stratified-v1.0",
    "msmarco-train-v2.1-stratified-v1.0",
    "squad-train-v2.0-stratified-v1.0",
    "asqa-train-stratified-v1.0"
]


def categorize_score(
    eval_result: List[Dict], 
    large_threshold: float,
    small_threshold: float
) -> List[Dict]:
    categorized_result = []
    for result_dict in eval_result:
        large_model_score = mean([result_dict["large_evaluation"]["f1_score"], result_dict["large_evaluation"]["bertscore"]])
        small_model_score = mean([result_dict["small_evaluation"]["f1_score"], result_dict["small_evaluation"]["bertscore"]])
        result_dict["large_evaluation"]["score"] = large_model_score
        result_dict["small_evaluation"]["score"] = small_model_score

        if large_model_score >= large_threshold:
            result_dict["large_evaluation"]["performance"] = "High"
        else:
            result_dict["large_evaluation"]["performance"] = "Low"

        if small_model_score >= small_threshold:
            result_dict["small_evaluation"]["performance"] = "High"
        else:
            result_dict["small_evaluation"]["performance"] = "Low"
        
        categorized_result.append(result_dict)
    
    return categorized_result


def query_wise_label_new(
    categorized_result: List[Dict]
) -> List[Dict]:
    labeled_dataset = []

    for result_dict in categorized_result:
        # TODO: Edit this function after labeling strategies have been set up
        rewriting_label = None
        if result_dict["large_evaluation"]["score"] > result_dict["small_evaluation"]["score"]:
            if result_dict["large_evaluation"]["performance"] == "High":
                rewriting_label = False
            elif result_dict["large_evaluation"]["performance"] == "Low":
                rewriting_label = True
        else:
            if result_dict["small_evaluation"]["performance"] == "High":
                rewriting_label = False
            elif result_dict["small_evaluation"]["performance"] == "Low":
                rewriting_label = True
        assert rewriting_label is not None
        
        datum = {}
        datum["id"] = str(result_dict["query_id"])
        datum["text"] = result_dict["query"]
        datum["label"] = rewriting_label
        labeled_dataset.append(datum)

    return labeled_dataset


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

    categorized_result = categorize_score(eval_result_instances, args.large_performance_threshold, args.small_performance_threshold)
    labeled_dataset = query_wise_label(categorized_result)

    result_file_path = os.path.join(args.result_dataset_path, "ft_dataset.json")
    dump_json(labeled_dataset, result_file_path)

    # train_test_split
    train_dataset, valid_and_test_dataset = train_test_split(labeled_dataset, test_size=((args.valid_ratio + args.test_ratio)/(args.train_ratio + args.valid_ratio + args.test_ratio)), shuffle=True, stratify=None, random_state=args.seed)
    valid_dataset, test_dataset = train_test_split(valid_and_test_dataset, test_size=(args.test_ratio/(args.valid_ratio + args.test_ratio)), shuffle=True, stratify=None, random_state=args.seed)

    train_result_file_path = os.path.join(args.result_dataset_path, "ft_dataset_train.json")
    valid_result_file_path = os.path.join(args.result_dataset_path, "ft_dataset_valid.json")
    test_result_file_path = os.path.join(args.result_dataset_path, "ft_dataset_test.json")
    
    dump_json(train_dataset, train_result_file_path)
    dump_json(valid_dataset, valid_result_file_path)
    dump_json(test_dataset, test_result_file_path)


if __name__ == "__main__":
    main(args)