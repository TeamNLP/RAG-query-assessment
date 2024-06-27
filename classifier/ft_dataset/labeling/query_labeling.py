import argparse
import json

from utils import load_jsonl, dump_jsonl

parser = argparse.ArgumentParser()
parser.add_argument('-l', '--large_model_eval_result_path', type=str, default="auto_evaluation/result/meta-llama-3-70b-instruct_evaluated_by_gpt-3.5-turbo-0125.jsonl", help="Path of the Large model evaluation results file")
parser.add_argument('-s', '--small_model_eval_result_path', type=str, default="auto_evaluation/result/meta-llama-3-8b-instruct_evaluated_by_gpt-3.5-turbo-0125.jsonl", help="Path of the Small model evaluation results file")
parser.add_argument('--large_model_threshold', type=float, default=0.5, help="Threshold for evaluation results of the large model")
parser.add_argument('--small_model_threshold', type=float, default=0.5, help="Threshold for evaluation results of the small model")
parser.add_argument('-r', '--result_file_path', type=str, default="../ft_dataset.jsonl", help="File Path to save query labeling results")

args = parser.parse_args()


def categorize_score(model_result, threshold):
    # !!! Temporary implementation !!!
    # TODO: Edit this function after the RAG evaluation is complete.
    # TODO: Edit default value of `threshold` 
    
    categorized_result = []
    for i in range(len(model_result)-1):
        result = model_result[i]
        score = result["exact_accurate"] + result["accurate"] + result["missing"] # TODO: Edit this part after the RAG evaluation is complete.
        
        if score >= threshold:
            result["performance"] = "High"
        else:
            result["performance"] = "Low"
        
        categorized_result.append(result)
    
    return categorized_result


def query_wise_label(large_model_result, small_model_result):
    # !!! Temporary implementation !!!

    assert len(large_model_result) == len(small_model_result)

    labeled_dataset = []

    for i in range(len(large_model_result)):
        assert large_model_result[i]["query"] == small_model_result[i]["query"]
        
        # TODO: Edit this function after labeling strategies have been set up
        if large_model_result[i]["performance"] == "High" and small_model_result[i]["performance"] == "High":
            rewriting_label = False
        else:
            rewriting_label = True

        datum = {}
        datum["query"] = large_model_result[i]["query"]
        datum["ground_truth"] = large_model_result[i]["ground_truth"]
        datum["rewriting_label"] = rewriting_label
        labeled_dataset.append(datum)

    return labeled_dataset


if __name__ == "__main__":
    large_model_result = load_jsonl(args.large_model_eval_result_path)
    small_model_result = load_jsonl(args.small_model_eval_result_path)

    large_model_result = categorize_score(large_model_result, args.large_model_threshold)
    small_model_result = categorize_score(small_model_result, args.small_model_threshold)

    labeled_dataset = query_wise_label(large_model_result, small_model_result)
    dump_jsonl(labeled_dataset, args.result_file_path)