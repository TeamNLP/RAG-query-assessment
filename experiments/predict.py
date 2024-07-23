import argparse
import json
import os
import requests
from typing import List, Tuple, Dict, Optional

import dotenv
import torch
from lib import read_jsonl, write_json, load_data
from openai import OpenAI
from prompt_templates import RAG_SYS_PROMPT, RAG_PROMPT_TEMPLATE, RAG_PROMPT_TEMPLATE_SUFFIX
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)
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
parser.add_argument('--retrieval_corpus_name', type=str, default=None, required=False, help="`corpus_name` for ElasticSearch Retriever")
parser.add_argument('--retriever_api_url', type=str, default=None, help="`api_url` for ElasticSearch Retriever")
parser.add_argument('--retrieval_top_n', type=int, default=3, help="A number for how many results to retrieve")
parser.add_argument('--generator_model_name', type=str, required=True, help="`model_name` for Generator")
parser.add_argument('--load_generator_in_4bit', action="store_true", help="whether load generator in 4bit. Only for HuggingFace models.")
parser.add_argument('--generator_max_new_tokens', type=int, default=100, help="`max_new_tokens` for generator.")
parser.add_argument("--dataset", type=str, required=True, choices=("hotpotqa", "2wikimultihopqa", "musique", 'nq', 'trivia', 'squad'), help="")
parser.add_argument("--dataset_type", type=str, required=True, choices=("train", "dev", "test_subsampled", "dev_500_subsampled"), help="")
parser.add_argument('--do_sample', action="store_true", help="whether use sampling while generate responses")
parser.add_argument('--temperature', type=float, default=1.0, help="")
parser.add_argument('--top_k', type=int, default=50, help="")
parser.add_argument('--top_p', type=float, default=1.0, help="")
parser.add_argument('--batch_size', type=int, default=2, help="batch size")
parser.add_argument('--debug', action="store_true", help="whether use debug mode")

args = parser.parse_args()


class Retriever:
    def __init__(self, corpus_name, top_n=3, method="retrieve_from_elasticsearch", api_url="http://localhost:8000/"):
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

    def get_passages(
        self,
        retrieval_results: List[Dict],
    ) -> str:
        passage_list = [result["paragraph_text"] for result in retrieval_results]
        passages = "\n".join(passage_list)
        return passages


class Generator:
    def __init__(self, model_name, generator_config, load_in_4bit=False):
        self.generator_config = generator_config
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

            # Load the tokenizer for the specified model.
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            if load_in_4bit:
                # Load the large language model with the specified quantization configuration.
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=False,
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    device_map="auto",
                    quantization_config=bnb_config,
                    torch_dtype=torch.float16,
                )

            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    device_map="auto",
                    torch_dtype=torch.float16,
                )
            
            # Initialize a text generation pipeline with the loaded model and tokenizer.
            self.generation_pipe = pipeline(
                task="text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=self.max_new_tokens,
            )

            self.terminators = [
                self.generation_pipe.tokenizer.eos_token_id,
                self.generation_pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]
        
        self.system_prompt = RAG_SYS_PROMPT
        self.prompt_template = RAG_PROMPT_TEMPLATE
        self.template_suffix = RAG_PROMPT_TEMPLATE_SUFFIX

    def make_rag_prompt(
        self,
        query: str, 
        passages: str
    ) -> str:
        return self.prompt_template.format(query=query, passages=passages)

    def generate_response(
        self, 
        query: str, 
        passages: str
    ) -> Optional[str]:
        input_prompt = self.make_rag_prompt(query, passages)

        if self.use_hf: # Load models at HuggingFace Hub
            response = self.generation_pipe(
                input_prompt,
                max_new_tokens=self.max_new_tokens,
                eos_token_id=self.terminators,
                do_sample=self.do_sample,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                pad_token_id=self.generation_pipe.tokenizer.eos_token_id
            )
            output = response[0]["generated_text"].split(self.template_suffix)[-1].strip()
            return output

        else: # Use OpenAI API
            try:
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": input_prompt},
                    ],
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

    def run(self, query, debug=False):
        retrieval_results, retrieval_time = self.retriever.retrieve(query)
        retrieved_passages = self.retriever.get_passages(retrieval_results)
        if debug: print("\nRetrieved passages: ", retrieved_passages)
        
        input_prompt = self.generator.make_rag_prompt(query, retrieved_passages)
        if debug: print("\nInput prompt: ", input_prompt)
        
        # Generate Response
        response = self.generator.generate_response(
            query=query, 
            passages=retrieved_passages
        )
        
        if debug: print("Generated response: ", response)
        
        return response


def make_rag(args):
    retriever = Retriever(
        corpus_name=args.retrieval_corpus_name, 
        top_n=args.retrieval_top_n, 
        api_url=args.retriever_api_url
    )
    generator = Generator(
        model_name=args.generator_model_name, 
        generator_config={
            "max_new_tokens":args.generator_max_new_tokens,
            "do_sample": args.do_sample,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "batch_size": args.batch_size
        },
        load_in_4bit=args.load_generator_in_4bit, 
    )

    # RAG framework 
    rag = RAGFramework(retriever, generator)
    return rag


def main(args):
    if args.retriever_api_url is None:
        try:
            args.retriever_api_url = os.environ.get('RETRIEVER_API_URL')
        except KeyError:
            raise KeyError("`retriever_api_url` required!")
    
    if args.retrieval_corpus_name is None:
        args.retrieval_corpus_name = CORPUS_NAME_DICT[args.dataset]        

    rag = make_rag(args)

    input_directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), "processed_data", args.dataset)
    input_filepath = os.path.join(input_directory, f"{args.dataset_type}.jsonl")
    input_instance = read_jsonl(input_filepath)
    dataset = load_data(input_filepath)

    output_directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), "predictions", args.dataset)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    output_filepath = os.path.join(output_directory, f"{args.dataset_type}.json")
    output_instance = {}
    
    for datum in tqdm(input_instance, desc=f"Run RAG on {args.dataset_type}.jsonl of {args.dataset}"):
        question_id = datum["question_id"]
        question_text = datum["question_text"]

        response = rag.run(query=question_text, debug=args.debug)
        output_instance[question_id] = response

    write_json(output_instance, output_filepath)


if __name__ == "__main__":
    main(args)