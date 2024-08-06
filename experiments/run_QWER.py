import argparse
import json
import os

import dotenv
import torch
from lib import CORPUS_NAME_DICT, read_jsonl, write_json
from loguru import logger
from openai import OpenAI
from model.QWER import make_qwer_framework
from tqdm import tqdm


dotenv.load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument('--input_directory', type=str, default="processed_data", help="`input_directory` to predict results")
parser.add_argument('--output_directory', type=str, default="predictions", help="`output_directory` to store the prediction results")
parser.add_argument('--classifier_model_name', type=str, default=None, required=False, help="`model_name` for Classifier.")
parser.add_argument('--rewriter_model_name', type=str, default="gpt-4o-mini-2024-07-18", help="OpenAI `model_name` for Rewriter")
parser.add_argument('--rewriter_max_new_tokens', type=int, default=200, help="`max_new_tokens` for Rewriter.")
parser.add_argument('--retrieval_corpus_name', type=str, default=None, required=False, help="`corpus_name` for ElasticSearch Retriever")
parser.add_argument('--retriever_api_url', type=str, default=None, help="`api_url` for ElasticSearch Retriever")
parser.add_argument('--retrieval_top_n', type=int, default=5, help="A number for how many results to retrieve")
parser.add_argument('--generator_model_name', type=str, default=None, help="`model_name` for Generator. Please refer to https://docs.vllm.ai/en/latest/models/supported_models.html.")
parser.add_argument('--generator_max_new_tokens', type=int, default=100, help="`max_new_tokens` for generator.")
parser.add_argument("--dataset", type=str, default=None, choices=("hotpotqa", "2wikimultihopqa", "musique", 'nq', 'trivia', 'squad'), help="")
parser.add_argument("--dataset_type", type=str, default=None, choices=("train", "dev", "test_subsampled", "dev_500_subsampled"), help="")
parser.add_argument('--do_sample', action="store_true", help="whether use sampling while generate responses")
parser.add_argument('--temperature', type=float, default=1.0, help="")
parser.add_argument('--top_k', type=int, default=50, help="")
parser.add_argument('--top_p', type=float, default=1.0, help="")
parser.add_argument('--vllm_tensor_parallel_size', type=int, default=1, help="TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.")
parser.add_argument('--vllm_gpu_memory_utilization', type=float, default=0.9, help="TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.")
parser.add_argument('--vllm_dtype', type=str, default="auto", help="Data type for model weights and activations. Possible choices: auto, half, float16, bfloat16, float, float32")
parser.add_argument('--batch_size', type=int, default=1, help="batch size")
parser.add_argument('--use_chat_template', action="store_true", help="whether use chat template")
parser.add_argument('--use_template_wo_instruction', action="store_true", help="whether use prompt template without instruction")
# parser.add_argument('--debug', action="store_true", help="whether use debug mode")

args = parser.parse_args()


def load_data_in_batches(dataset_path, batch_size):
    """
    Generator function that reads data from a compressed file and yields batches of data.
    Each batch is a dictionary containing lists of question_id, question_text and predictions.
    
    Args:
    dataset_path (str): Path to the dataset file.
    batch_size (int): Number of data items in each batch.
    
    Yields:
    dict: A batch of data.
    """
    def initialize_batch():
        """ Helper function to create an empty batch. """
        return {"question_id": [], "question_text": [], "prediction": []}

    try:
        with open(dataset_path, "r") as file:
            batch = initialize_batch()
            for line in file.readlines():
                try:
                    item = json.loads(line.strip())
                    batch["question_id"].append(item["question_id"])
                    batch["question_text"].append(item["question_text"])
                    
                    if len(batch["question_text"]) == batch_size:
                        yield batch
                        batch = initialize_batch()
                except json.JSONDecodeError:
                    logger.warn("Warning: Failed to decode a line.")
            # Yield any remaining data as the last batch
            if batch["question_text"]:
                yield batch
    except FileNotFoundError as e:
        logger.error(f"Error: The file {dataset_path} was not found.")
        raise e
    except IOError as e:
        logger.error(f"Error: An error occurred while reading the file {dataset_path}.")
        raise e


def generate_hf_predictions(dataset_path, qwer_framework, batch_size=2):
    """
    Processes batches of data from a dataset to generate predictions using a model.
    
    Args:
    dataset_path (str): Path to the dataset.
    qwer_framework (object): QWERFramework that provides `run_hf()` interfaces.
    batch_size (int)
    
    Returns:
    dict: A dictionary with question_id as key and predictions as value.
    """
    question_ids, queries, predictions = [], [], []

    if batch_size == 1:
        raise NotImplementedError

    elif batch_size > 1:
        for batch in tqdm(load_data_in_batches(dataset_path, batch_size), desc=f"Generating predictions on {dataset_path}"):
            batch_predictions = qwer_framework.run_framework(batch)
            
            question_ids.extend(batch["question_id"])
            queries.extend(batch["question_text"])
            predictions.extend(batch_predictions)
        
        assert len(question_ids) == len(queries) and len(queries) == len(predictions)

        output_instance = {}
        for i in range(len(queries)):
            question_id = question_ids[i]
            prediction = predictions[i]

            output_instance[question_id] = prediction

        return output_instance


def generate_gpt_predictions(
    dataset_path, 
    qwer_framework, 
    # debug=False
):
    """
    Generate predictions using an OpenAI GPT model.
    
    Args:
    dataset_path (str): Path to the dataset.
    qwer_framework (object): QWERFramework that provides `run_gpt()` interfaces.
    
    Returns:
    dict: A dictionary with question_id as key and predictions as value.
    """
    
    input_instance = read_jsonl(dataset_path)
    output_instance = {}

    for datum in tqdm(input_instance, desc=f"Generating predictions on {dataset_path}"):
        question_id = datum["question_id"]
        question_text = datum["question_text"]

        response = qwer_framework.run_framework(
            query=question_text, 
            # debug=debug
        )
        output_instance[question_id] = response
    
    return output_instance


def main(args):
    if args.retriever_api_url is None:
        try:
            args.retriever_api_url = os.environ.get('RETRIEVER_API_URL')
        except KeyError:
            raise KeyError("`retriever_api_url` required!")
    
    if args.retrieval_corpus_name is None:
        args.retrieval_corpus_name = CORPUS_NAME_DICT[args.dataset]        

    qwer_framework = make_qwer_framework(args)

    input_directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), args.input_directory, args.dataset)
    input_filepath = os.path.join(input_directory, f"{args.dataset_type}.jsonl")

    output_directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), args.output_directory, args.dataset)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    output_filepath = os.path.join(output_directory, f"prediction_{args.dataset_type}.json")

    print(f"run RAG on {args.dataset_type}.jsonl of {args.dataset}")
    if "gpt-3.5" in args.generator_model_name.lower() or "gpt-4" in args.generator_model_name.lower():
        output_instance = generate_gpt_predictions(
            input_filepath, 
            qwer_framework, 
            # debug=args.debug
        )
    else:
        output_instance = generate_hf_predictions(
            input_filepath, 
            qwer_framework, 
            batch_size=args.batch_size
        )

    write_json(output_instance, output_filepath)


if __name__ == "__main__":
    main(args)