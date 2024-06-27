# Reference: https://gitlab.aicrowd.com/aicrowd/challenges/meta-comprehensive-rag-benchmark-kdd-cup-2024/meta-comphrehensive-rag-benchmark-starter-kit/-/blob/32f82d9a2097bb77ae6cc8cdca5d3b0e028667fb/local_evaluation.py

import argparse
import bz2
import json
import os
from datetime import datetime

import dotenv
from loguru import logger
from openai import APIConnectionError, OpenAI, RateLimitError
from prompts.templates import IN_CONTEXT_EXAMPLES, INSTRUCTIONS
from tqdm.auto import tqdm
from transformers import LlamaTokenizerFast

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', type=str, default="ms_marco.jsonl", help="File path to the dataset file to evaluate")
parser.add_argument('--evaluation_model_name', type=str, default="gpt-3.5-turbo-0125", help="The name of the OpenAI model to use for evaluation. We recommend using `gpt-4o-2024-05-13`, `gpt-4-0125-preview`, or `gpt-3.5-turbo-0125`. See https://platform.openai.com/docs/models.")
parser.add_argument('--target_model_name', type=str, default=None, help="Name of the model being evaluated. e.g., `meta-llama/Meta-Llama-3-70B-Instruct` or `meta-llama/Meta-Llama-3-8B-Instruct`.")
parser.add_argument('--output_path', type=str, default="result", help="Diretory Path to save evaluation results")

args = parser.parse_args()


def get_system_message():
    """Returns the system message containing instructions and in context examples."""
    return INSTRUCTIONS + IN_CONTEXT_EXAMPLES


def attempt_api_call(client, model_name, messages, max_retries=10):
    """Attempt an API call with retries upon encountering specific errors."""
    # todo: add default response when all efforts fail
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                response_format={"type": "json_object"},
            )
            return response.choices[0].message.content
        except (APIConnectionError, RateLimitError):
            logger.warning(
                f"API call failed on attempt {attempt + 1}, retrying..."
            )
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            break
    return None


def log_response(messages, response, output_directory="api_responses"):
    """Save the response from the API to a file."""
    os.makedirs(output_directory, exist_ok=True)
    file_name = datetime.now().strftime("%d-%m-%Y-%H-%M-%S.json")
    file_path = os.path.join(output_directory, file_name)
    with open(file_path, "w") as f:
        json.dump({"messages": messages, "response": response}, f, indent=4)


def parse_response(resp: str):
    """Pass auto-eval output from the evaluator."""
    try:
        resp = resp.lower()
        model_resp = json.loads(resp)
        answer = -1
        if "accuracy" in model_resp and (
            (model_resp["accuracy"] is True)
            or (
                isinstance(model_resp["accuracy"], str)
                and model_resp["accuracy"].lower() == "true"
            )
        ):
            answer = 1
        else:
            raise ValueError(
                f"Could not parse answer from response: {model_resp}"
            )

        return answer
    except:
        return -1

# def trim_predictions_to_max_token_length(prediction):
#     """Trims prediction output to 75 tokens"""
#     max_token_length = 75
#     tokenized_prediction = tokenizer.encode(prediction)
#     trimmed_tokenized_prediction = tokenized_prediction[1: max_token_length+1]
#     trimmed_prediction = tokenizer.decode(trimmed_tokenized_prediction)
#     return trimmed_prediction

def get_generated_predictions(dataset_path, target_model_name):
    predictions = []
    data_list = []
    with open(dataset_path, "r") as f:
        for line in tqdm(f, desc=f"Getting Generated Predictions of {target_model_name}"):
            data = json.loads(line)
            
            query = data["query"]
            ground_truth = data["ground_truth"]
            prediction = data[f"prediction_by_{target_model_name.split('/')[-1].lower()}"]
            # references = data["references"]
            
            predictions.append(
                {
                    "query": query,
                    "ground_truth": str(data["ground_truth"]).strip().lower(),
                    "prediction": str(prediction).strip().lower(),
                    # "references": references,
                }
            )            

    return predictions


def get_eval_result_log(eval_type):
    eval_result_log = {
        "exact_accurate": int(eval_type == "exact_accurate"),
        "accurate": int(eval_type == "accurate"),
        "hallucination": int(eval_type == "hallucination"),
        "missing": int(eval_type == "missing"),
    }

    return eval_result_log


def evaluate_predictions(predictions, evaluation_model_name, openai_client):
    n_miss, n_correct, n_correct_exact = 0, 0, 0
    system_message = get_system_message()

    result_logs = []
    for prediction_dict in tqdm(predictions, total=len(predictions), desc="Evaluating Predictions"):
        query, ground_truth, prediction = (
            prediction_dict["query"],
            prediction_dict["ground_truth"],
            prediction_dict["prediction"],
        )

        messages = [
            {"role": "system", "content": system_message},
            {
                "role": "user",
                "content": f"Question: {query}\n Ground truth: {ground_truth}\n Prediction: {prediction}\n",
            },
        ]

        if prediction == ground_truth:
            n_correct_exact += 1
            n_correct += 1
            eval_result_log = get_eval_result_log("exact_accurate")
            result_logs.append(eval_result_log)
            continue
        elif prediction == "i don't know" or prediction == "i don't know." or prediction == "i do not know" or prediction == "i do not know.":
            n_miss += 1
            eval_result_log = get_eval_result_log("missing")
            result_logs.append(eval_result_log)
            continue

        response = attempt_api_call(
            openai_client, evaluation_model_name, messages
        )
        if response:
            log_response(messages, response)
            eval_res = parse_response(response)
            if eval_res == 1:
                n_correct += 1
                eval_result_log = get_eval_result_log("accurate")
                result_logs.append(eval_result_log)
            else:
                eval_result_log = get_eval_result_log("hallucination")
                result_logs.append(eval_result_log)

    n = len(predictions)
    results = {
        "score": (2 * n_correct + n_miss) / n - 1,
        "exact_accuracy": n_correct_exact / n,
        "accuracy": n_correct / n,
        "hallucination": (n - n_correct - n_miss) / n,
        "missing": n_miss / n,
        "n_miss": n_miss,
        "n_correct": n_correct,
        "n_correct_exact": n_correct_exact,
        "total": n,
    }
    logger.info(results)
    return results, result_logs


if __name__ == "__main__":
    dotenv.load_dotenv()

    DATASET_PATH = args.dataset_path
    EVALUATION_MODEL_NAME = args.evaluation_model_name
    TARGET_MODEL_NAME = args.target_model_name

    # Get generated predictions
    predictions = get_generated_predictions(DATASET_PATH, TARGET_MODEL_NAME)
    
    # Evaluate Predictions
    openai_client = OpenAI(api_key = os.environ.get('OPENAI_API_KEY'))

    evaluation_results, result_logs = evaluate_predictions(
        predictions, EVALUATION_MODEL_NAME, openai_client
    )
    assert len(predictions) == len(result_logs), f"len(predictions): {len(predictions)}, len(result_logs): {len(result_logs)}"
    
    for p, r in zip(predictions, result_logs):
        p.update(r)

    predictions.append(evaluation_results)

    # logging file directory, file name
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    with open(f"{args.output_path}/{TARGET_MODEL_NAME.split('/')[-1].lower()}_evaluated_by_{args.evaluation_model_name}.json", 'w') as f:
        for prediction in predictions:
            json_record = json.dumps(prediction, ensure_ascii=False)
            f.write(json_record + '\n')
