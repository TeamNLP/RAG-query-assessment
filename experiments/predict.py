import argparse
import json
import os
import requests
from typing import List, Tuple, Dict, Optional, Union, Any

import dotenv
import torch
import vllm
from lib import read_jsonl, write_json
from loguru import logger
from openai import OpenAI
from prompt_templates import RAG_SYS_PROMPT, RAG_PROMPT_TEMPLATE, RAG_PROMPT_TEMPLATE_WO_INST, RAG_PROMPT_TEMPLATE_SUFFIX
from tqdm import tqdm

CORPUS_NAME_DICT = {
    "hotpotqa":"hotpotqa",
    "2wikimultihopqa":"2wikimultihopqa",
    "musique":"musique",
    'nq':'wiki',
    'trivia':'wiki',
    'squad':'wiki'
}

dotenv.load_dotenv()
api_key = os.environ.get('OPENAI_API_KEY')

client = OpenAI(api_key=api_key)

parser = argparse.ArgumentParser()
parser.add_argument('--input_directory', type=str, default="processed_data", help="`input_directory` to predict results")
parser.add_argument('--output_directory', type=str, default="predictions", help="`output_directory` to store the prediction results")
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
parser.add_argument('--debug', action="store_true", help="whether use debug mode")

args = parser.parse_args()


class Retriever:
    def __init__(self, corpus_name, top_n=5, method="retrieve_from_elasticsearch", api_url="http://localhost:8000/"):
        assert method == "retrieve_from_elasticsearch"

        self.corpus_name = corpus_name
        self.top_n = top_n
        self.api_url = api_url
        
    def retrieve(
        self, 
        query: str
    ) -> Tuple[List[Dict], float]:
        payload = {
            "query_text": query,
            "retrieval_method": "retrieve_from_elasticsearch",
            "max_hits_count": self.top_n,
            "corpus_name": self.corpus_name,
        }

        response = requests.post(self.api_url, json=payload)

        if response.status_code == 200:
            results = response.json()
            retrieval_results = results['retrieval']
            retrieval_time = results['time_in_seconds']
            return retrieval_results, retrieval_time
        else:
            raise Exception(f"Retriever Request Failed! (Status Code: {response.status_code}, Text: {response.text})")

    def get_passages_as_list(
        self,
        retrieval_results: List[Dict],
    ) -> List[str]:
        passage_list = [result["paragraph_text"] for result in retrieval_results]
        return passage_list

    def get_passages_as_str(
        self,
        retrieval_results: List[Dict],
    ) -> str:
        passage_list = [result["paragraph_text"] for result in retrieval_results]
        passages = "\n".join(passage_list)
        return passages


class Generator:
    def __init__(self, model_name, generator_config):
        self.generator_config = generator_config
        self.use_chat_template = generator_config["use_chat_template"]
        self.max_new_tokens = generator_config["max_new_tokens"]
        self.do_sample = generator_config["do_sample"]
        self.temperature = generator_config["temperature"]
        self.top_k = generator_config["top_k"]
        self.top_p = generator_config["top_p"]
        self.batch_size = generator_config["batch_size"]

        self.model_name = model_name
        if "gpt-3.5" in self.model_name.lower() or "gpt-4" in self.model_name.lower():
            self.use_hf = False
            self.model = self.model_name
        else: # Load models at HuggingFace Hub
            self.use_hf = True 

            # initialize the model with vllm
            self.llm = vllm.LLM(
                self.model_name,
                worker_use_ray=True,
                tensor_parallel_size=generator_config["vllm_tensor_parallel_size"], 
                gpu_memory_utilization=generator_config["vllm_gpu_memory_utilization"],
                trust_remote_code=True,
                dtype=generator_config["vllm_dtype"],
                enforce_eager=True
            )
            self.tokenizer = self.llm.get_tokenizer()
        
        self.system_prompt = RAG_SYS_PROMPT
        self.prompt_template = RAG_PROMPT_TEMPLATE
        self.prompt_template_without_instruction = RAG_PROMPT_TEMPLATE_WO_INST
        self.template_suffix = RAG_PROMPT_TEMPLATE_SUFFIX

    def make_rag_prompt(
        self,
        query: str, 
        passages: str
    ) -> Union[str, List]:
        if self.use_chat_template:
            if self.use_hf:
                # Formats queries and retrieval results using the chat_template of the model.
                return self.tokenizer.apply_chat_template(
                        [
                            {"role": "system", "content": RAG_SYS_PROMPT},
                            {"role": "user", "content": self.prompt_template_without_instruction.format(query=query, passages=passages)},
                        ],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
            else: # use OpenAI API
                formatted_chat_template = [
                    {"role": "system", "content": RAG_SYS_PROMPT},
                    {"role": "user", "content": self.prompt_template_without_instruction.format(query=query, passages=passages)},
                ]
                return formatted_chat_template
        else:
            if args.use_template_wo_instruction:
                return self.prompt_template_without_instruction.format(query=query, passages=passages)
            else:
                return self.prompt_template.format(instruction=RAG_SYS_PROMPT, query=query, passages=passages)

    def make_rag_prompt_batch(
        self,
        queries: List[str], 
        batch_passages: List[str] = []
    ) -> List[str]:
        """
        Parameters:
        - queries (List[str]): A list of queries to be formatted into prompts.
        - batch_passages (List[str])
        """
        formatted_prompts = []
        for _idx, query in enumerate(queries):
            passages = batch_passages[_idx]
            formatted_prompts.append(self.make_rag_prompt(query, passages))

        return formatted_prompts

    def generate_gpt_response(
        self, 
        query: str, 
        passages: str
    ) -> Optional[str]:
        assert self.use_hf == False
        assert args.use_chat_template == True
        formatted_chat_template = self.make_rag_prompt(query, passages)

        try:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=formatted_chat_template,
                temperature=self.temperature,
                max_tokens=args.generator_max_new_tokens
            )
            output = response.choices[0].message.content.strip()
            return output

        except Exception as e:
            print(e)
            return None


class RAGFramework:
    def __init__(self, retriever, generator):
        self.retriever = retriever
        self.generator = generator

    def run_gpt(self, query, debug=False):
        assert self.generator.use_hf == False

        retrieval_results, retrieval_time = self.retriever.retrieve(query)
        retrieved_passages = self.retriever.get_passages_as_str(retrieval_results)
        if debug: print("\nRetrieved passages: ", retrieved_passages)
        
        input_prompt = self.generator.make_rag_prompt(query, retrieved_passages)
        if debug: print("\nInput prompt: ", input_prompt)
        
        # Generate Response
        response = self.generator.generate_gpt_response(
            query=query, 
            passages=retrieved_passages
        )
        
        if debug: print("Generated response: ", response)
        
        return response

    def batch_generate_prediction(self, batch: Dict[str, Any]) -> List[str]:
        """
        Generates predictions for a batch of queries using associated (pre-cached) search results and query times.

        Parameters:
            batch (Dict[str, Any]): A dictionary containing a batch of input queries with the following keys:
                - 'question_text' (List[str]): List of user queries.

        Returns:
            List[str]: A list of plain text responses for each query in the batch. Each response is limited to given max_new_tokens.
            If the generated response exceeds max_new_tokens, it will be truncated to fit within this limit.
        """
        queries = batch["question_text"]

        # Retrieve top matches for the whole batch
        batch_retrieval_passages = []
        for _idx, query in enumerate(queries):
            retrieval_results, _ = self.retriever.retrieve(query)
            retrieval_passages = self.retriever.get_passages_as_list(retrieval_results)
            batch_retrieval_passages.append(retrieval_passages)
            
        # Prepare formatted prompts from the LLM        
        formatted_prompts = self.generator.make_rag_prompt_batch(queries, batch_retrieval_passages)

        # Generate responses via vllm
        responses = self.generator.llm.generate(
            formatted_prompts,
            vllm.SamplingParams(
                n=1,  # Number of output sequences to return for each prompt.
                top_p=self.generator.top_p,  # Float that controls the cumulative probability of the top tokens to consider.
                top_k=self.generator.top_k,
                temperature=self.generator.temperature,  # Randomness of the sampling
                skip_special_tokens=True,  # Whether to skip special tokens in the output.
                stop_token_ids=[self.generator.tokenizer.eos_token_id, self.generator.tokenizer.convert_tokens_to_ids("<|eot_id|>")],
                max_tokens=self.generator.max_new_tokens  # Maximum number of tokens to generate per output sequence.
            ),
            use_tqdm=False # you might consider setting this to True during local development
        )

        # Aggregate predictions into List[str]
        predictions = []
        for response in responses:
            predictions.append(response.outputs[0].text)
        
        return predictions


def make_rag(args):
    retriever = Retriever(
        corpus_name=args.retrieval_corpus_name, 
        top_n=args.retrieval_top_n, 
        api_url=args.retriever_api_url
    )
    generator = Generator(
        model_name=args.generator_model_name, 
        generator_config={
            "use_chat_template":args.use_chat_template,
            "max_new_tokens":args.generator_max_new_tokens,
            "do_sample": args.do_sample,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "vllm_tensor_parallel_size": args.vllm_tensor_parallel_size,
            "vllm_gpu_memory_utilization": args.vllm_gpu_memory_utilization,
            "vllm_dtype": args.vllm_dtype,
            "batch_size": args.batch_size
        },
    )

    # RAG framework 
    rag = RAGFramework(retriever, generator)
    return rag


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


def generate_hf_predictions(dataset_path, rag_model, batch_size=2):
    """
    Processes batches of data from a dataset to generate predictions using a model.
    
    Args:
    dataset_path (str): Path to the dataset.
    rag_model (object): RAGFramework that provides `batch_generate_prediction()` interfaces.
    batch_size (int)
    
    Returns:
    dict: A dictionary with question_id as key and predictions as value.
    """
    question_ids, queries, predictions = [], [], []

    for batch in tqdm(load_data_in_batches(dataset_path, batch_size), desc=f"Generating predictions on {dataset_path}"):
        batch_predictions = rag_model.batch_generate_prediction(batch)
        
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

def generate_gpt_predictions(dataset_path, rag_model, debug=False):
    """
    Generate predictions using an OpenAI GPT model.
    
    Args:
    dataset_path (str): Path to the dataset.
    rag_model (object): RAGFramework that provides `batch_generate_prediction()` interfaces.
    
    Returns:
    dict: A dictionary with question_id as key and predictions as value.
    """
    
    input_instance = read_jsonl(dataset_path)
    output_instance = {}

    for datum in tqdm(input_instance, desc=f"Generating predictions on {dataset_path}"):
        question_id = datum["question_id"]
        question_text = datum["question_text"]

        response = rag_model.run_gpt(query=question_text, debug=debug)
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

    rag = make_rag(args)

    input_directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), args.input_directory, args.dataset)
    input_filepath = os.path.join(input_directory, f"{args.dataset_type}.jsonl")

    output_directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), args.output_directory, args.dataset)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    output_filepath = os.path.join(output_directory, f"prediction_{args.dataset_type}.json")

    print(f"run RAG on {args.dataset_type}.jsonl of {args.dataset}")
    if "gpt-3.5" in args.generator_model_name.lower() or "gpt-4" in args.generator_model_name.lower():
        output_instance = generate_gpt_predictions(input_filepath, rag, debug=args.debug)
    else:
        output_instance = generate_hf_predictions(input_filepath, rag, batch_size=args.batch_size)

    write_json(output_instance, output_filepath)


if __name__ == "__main__":
    main(args)