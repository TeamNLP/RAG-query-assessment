import argparse
import json

from utils import load_jsonl

parser = argparse.ArgumentParser()
parser.add_argument('-l', '--large_model_eval_result_path', type=str, default="auto_evaluation/result/meta-llama-3-70b-instruct_evaluated_by_gpt-3.5-turbo-0125.jsonl", help="Path of the Large model evaluation results file")
parser.add_argument('-s', '--small_model_eval_result_path', type=str, default="auto_evaluation/result/meta-llama-3-8b-instruct_evaluated_by_gpt-3.5-turbo-0125.jsonl", help="Path of the Small model evaluation results file")
parser.add_argument('--output_path', type=str, default="auto_evaluation/result", help="Diretory Path to save query labeling results")

args = parser.parse_args()

def foo():


if __name__ == "__main__":
    large_result = load_jsonl(args.large_model_eval_result_path)
    small_result = load_jsonl(args.small_model_eval_result_path)


    # with open(dataset_path, "r") as f:
    #     for line in tqdm(f, desc=f"Getting Generated Predictions of {target_model_name}"):
    #         data = json.loads(line)
            
    #         query = data["query"]
    #         ground_truth = data["ground_truth"]
    #         prediction = data[f"prediction_by_{target_model_name.split('/')[-1].lower()}"]
    #         # references = data["references"]
            
    #         predictions.append(
    #             {
    #                 "query": query,
    #                 "ground_truth": str(data["ground_truth"]).strip().lower(),
    #                 "prediction": str(prediction).strip().lower(),
    #                 # "references": references,
    #             }
    #         )            
