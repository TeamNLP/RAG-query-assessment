import argparse
import json
import os

import dotenv
from lib import make_request_dict, read_jsonl, write_jsonl
from openai import OpenAI
from tqdm import tqdm


dotenv.load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument('--generator_model_name', type=str, default="gpt-4o-2024-08-06", help="`model_name` for Generator. Please refer to https://docs.vllm.ai/en/latest/models/supported_models.html.")
parser.add_argument('--input_directory', type=str, default="source_dataset", help="`input_directory` to predict results")
parser.add_argument('--output_directory', type=str, default="predictions", help="`output_directory` to store the prediction results")
parser.add_argument("--dataset_type", type=str, default="train_18000_subsampled", choices=("train", "train_18000_subsampled", "dev", "test_subsampled", "dev_500_subsampled", "dev_7_subsampled"), help="")
parser.add_argument('--openai_batch_api', type=str, default=None, choices=("upload", "check", "analysis"), help="OpenAI Batch API")
parser.add_argument('--openai_batch_id', type=str, default=None, help="ID of OpenAI Batch object")
parser.add_argument('--openai_file_id', type=str, default=None, help="ID of OpenAI Files API")
parser.add_argument('--temperature', type=float, default=0.1, help="")
parser.add_argument('--generator_max_new_tokens', type=int, default=100, help="`max_new_tokens` for generator.")

args = parser.parse_args()






def main(args):
    assert args.generator_model_name == "gpt-4o-2024-08-06"
    assert args.temperature == 0.1
    assert args.generator_max_new_tokens == 100

    openai_client = None
    if "gpt-3.5" in args.generator_model_name.lower() or "gpt-4" in args.generator_model_name.lower():
        api_key = os.environ.get('OPENAI_API_KEY')
        openai_client = OpenAI(api_key=api_key)

    if args.openai_batch_api == "upload":
        for dataset in ["ms_marco", "nq", "trivia", "squad", "hotpotqa", "2wikimultihopqa", "musique"]:
                
            input_directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), args.input_directory, dataset)
            input_filepath = os.path.join(input_directory, f"{args.dataset_type}.jsonl")
            input_instance = read_jsonl(input_filepath)

            batch_file_directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), args.output_directory, "BatchAPI", dataset)
            if not os.path.exists(batch_file_directory):
                os.makedirs(batch_file_directory)

            for evaluation_type in ["Uncertainty", "Complexity", "Redundancy", "Subjectivity"]:
                batch_input_filepath = os.path.join(batch_file_directory, f"{evaluation_type}_batch_input_{args.dataset_type}.jsonl")

                batch_instances = []
                for datum in tqdm(input_instance, desc=f"Making an OpenAI Batch File on {input_filepath} - dataset: {dataset}, evaluation_type: {evaluation_type}"):
                    question_id = datum["question_id"]
                    question_text = datum["question_text"]

                    batch_instance, tok_len = make_request_dict(question_id, question_text, evaluation_type=evaluation_type)
                    batch_instances.append(batch_instance)
                write_jsonl(batch_instances, batch_input_filepath)

                batch_input_file = openai_client.files.create(
                    file=open(batch_input_filepath, "rb"),
                    purpose="batch"
                )

                batch_input_file_id = batch_input_file.id

                batch_obj = openai_client.batches.create(
                    input_file_id=batch_input_file_id,
                    endpoint="/v1/chat/completions",
                    completion_window="24h",
                    metadata={
                        "description": f"{evaluation_type}_batch_{dataset}"
                    }
                )
                batch_id = batch_obj.id

                print(f"{evaluation_type}_batch_{dataset} - batch_obj.id: {batch_id}")
    
    if args.openai_batch_api == "check":
        if args.openai_batch_id is not None:
            batch_obj = openai_client.batches.retrieve(args.openai_batch_id)
            status = batch_obj.status
            description = batch_obj.metadata["description"]
            print(f"{args.generator_model_name} - {description} ({args.openai_batch_id}): {status}")

            if status == "completed":
                print(f"Result of {description} ({args.openai_batch_id}) will be analyzed now :)")
                args.openai_file_id = batch_obj.output_file_id
                evaluation_type = description.split("_batch_")[0]
                dataset = description.split("_batch_")[1]
                print(f"Output File ID of {description} ({args.openai_batch_id}): {args.openai_file_id}")
                args.openai_batch_api = "analysis"
            else:
                return
        else:
            raise Exception(f"args.openai_batch_id is None!")

    if args.openai_batch_api == "analysis":
        if args.openai_file_id is not None:
            # Preparing Batch File Results
            file_response = openai_client.files.content(args.openai_file_id)
            batch_str_list = file_response.text.split("\n")[:-1]
            
            batch_output_instances = []
            for batch_str in tqdm(batch_str_list):
                batch_output_instance = json.loads(batch_str)
                batch_output_instances.append(batch_output_instance)

            batch_file_directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), args.output_directory, "BatchAPI", args.dataset)
            if not os.path.exists(batch_file_directory):
                os.makedirs(batch_file_directory)

            if "LLM" in batch_output_instance["custom_id"][0:3]:
                batch_output_filepath = os.path.join(batch_file_directory, f"LLM_batch_output_{args.dataset_type}.jsonl")
                generation_output_filepath = generator_output_filepath
            elif "RAG" in batch_output_instance["custom_id"][0:3]:
                batch_output_filepath = os.path.join(batch_file_directory, f"RAG_batch_output_{args.dataset_type}.jsonl")
                generation_output_filepath = rag_output_filepath
            else:
                raise Exception
            write_jsonl(batch_output_instances, batch_output_filepath)
            
            generation_output_instance = []
            for idx, batch_output_instance in enumerate(batch_output_instances):
                datum = input_instance[idx]
                question_id = datum["question_id"]
                question_text = datum["question_text"]
                answers_objects = datum["answers_objects"]
                generation_result = batch_output_instance["response"]["body"]["choices"][0]["message"]["content"]
                
                generation_output_dict = {}
                generation_output_dict["question_id"] = question_id
                generation_output_dict["question_text"] = question_text
                generation_output_dict["answers_objects"] = answers_objects
                generation_output_dict["result"] = generation_result
                generation_output_instance.append(generation_output_dict)
            write_jsonl(generation_output_instance, generation_output_filepath)
            
            return
        else:
            raise Exception(f"args.openai_batch_id is None!")



if __name__ == "__main__":
    main(args)