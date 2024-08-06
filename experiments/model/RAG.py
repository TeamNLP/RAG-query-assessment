import requests
from typing import List, Tuple, Dict, Optional, Union, Any

import vllm
from openai import OpenAI
from prompt_templates import RAG_SYS_PROMPT, RAG_PROMPT_TEMPLATE, RAG_PROMPT_TEMPLATE_WO_INST, RAG_PROMPT_TEMPLATE_SUFFIX
from transformers import AutoTokenizer, AutoModelForSequenceClassification


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
    def __init__(self, model_name, generator_config, openai_client=None):
        self.generator_config = generator_config
        self.use_chat_template = generator_config["use_chat_template"]
        self.use_template_wo_instruction = generator_config["use_template_wo_instruction"]
        self.max_new_tokens = generator_config["max_new_tokens"]
        self.do_sample = generator_config["do_sample"]
        self.temperature = generator_config["temperature"]
        self.top_k = generator_config["top_k"]
        self.top_p = generator_config["top_p"]
        self.batch_size = generator_config["batch_size"]

        self.model_name = model_name
        if "gpt-3.5" in self.model_name.lower() or "gpt-4" in self.model_name.lower():
            assert openai_client is not None
            self.use_hf = False
            self.model = self.model_name
            self.openai_client = openai_client
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
            if self.use_template_wo_instruction:
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
        assert self.use_chat_template == True
        formatted_chat_template = self.make_rag_prompt(query, passages)

        try:
            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=formatted_chat_template,
                temperature=self.temperature,
                max_tokens=self.max_tokens
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
    openai_client = None
    if "gpt-3.5" in args.generator_model_name.lower() or "gpt-4" in args.generator_model_name.lower():
        api_key = os.environ.get('OPENAI_API_KEY')
        openai_client = OpenAI(api_key=api_key)

    retriever = Retriever(
        corpus_name=args.retrieval_corpus_name, 
        top_n=args.retrieval_top_n, 
        api_url=args.retriever_api_url
    )
    generator = Generator(
        model_name=args.generator_model_name, 
        generator_config={
            "use_chat_template":args.use_chat_template,
            "use_template_wo_instruction":args.use_template_wo_instruction,
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
        openai_client=openai_client
    )

    # RAG framework 
    rag = RAGFramework(retriever, generator)
    return rag