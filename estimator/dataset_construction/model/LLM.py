from typing import List, Tuple, Dict, Optional, Union, Any

import vllm
from model.prompt_templates import LLM_SYS_PROMPT, LLM_PROMPT_TEMPLATE, LLM_PROMPT_TEMPLATE_WO_INST, LLM_PROMPT_TEMPLATE_SUFFIX
from openai import OpenAI


class GeneratorLLM:
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
        
        self.system_prompt = LLM_SYS_PROMPT
        self.prompt_template = LLM_PROMPT_TEMPLATE
        self.prompt_template_without_instruction = LLM_PROMPT_TEMPLATE_WO_INST
        self.template_suffix = LLM_PROMPT_TEMPLATE_SUFFIX

    def make_llm_prompt(
        self,
        query: str
    ) -> Union[str, List]:
        if self.use_chat_template:
            if self.use_hf:
                # Formats queries and retrieval results using the chat_template of the model.
                return self.tokenizer.apply_chat_template(
                        [
                            {"role": "system", "content": LLM_SYS_PROMPT},
                            {"role": "user", "content": self.prompt_template_without_instruction.format(query=query)},
                        ],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
            else: # use OpenAI API
                formatted_chat_template = [
                    {"role": "system", "content": LLM_SYS_PROMPT},
                    {"role": "user", "content": self.prompt_template_without_instruction.format(query=query)},
                ]
                return formatted_chat_template
        else:
            if self.use_template_wo_instruction:
                return self.prompt_template_without_instruction.format(query=query)
            else:
                return self.prompt_template.format(instruction=LLM_SYS_PROMPT, query=query)

    def make_llm_prompt_batch(
        self,
        queries: List[str], 
    ) -> List[str]:
        """
        Parameters:
        - queries (List[str]): A list of queries to be formatted into prompts.
        """
        formatted_prompts = []
        for _idx, query in enumerate(queries):
            formatted_prompts.append(self.make_llm_prompt(query))

        return formatted_prompts

    def generate_gpt_response(
        self, 
        query: str, 
    ) -> Optional[str]:
        assert self.use_hf == False
        assert self.use_chat_template == True
        formatted_chat_template = self.make_llm_prompt(query)

        try:
            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=formatted_chat_template,
                temperature=self.temperature,
                max_tokens=self.max_new_tokens
            )
            output = response.choices[0].message.content.strip()
            return output

        except Exception as e:
            print(e)
            return None

    def generate_hf_response(
        self, 
        query: Union[str, Dict[str, Any]]
    ) -> Union[str, List[str]]:
        """
        Generates predictions for a single query or a batch of queries.

        Parameters:
            (A single) query (str): A input query
                or
            (A batch of) query (Dict[str, Any]): A dictionary containing a batch of input queries with the following keys:
                - 'question_text' (List[str]): List of user queries.

        Returns:
                str: A plain text response for a single query. The response is limited to given max_new_tokens.
                    If the generated response exceeds max_new_tokens, it will be truncated to fit within this limit.

            or
                List[str]: A list of plain text responses for each query in the batch. Each response is limited to given max_new_tokens.
                    If the generated response exceeds max_new_tokens, it will be truncated to fit within this limit.
        """
        assert self.use_hf == True

        if isinstance(query, str): # A single query
            # Prepare a formatted prompt from the generator  
            formatted_prompt = self.make_llm_prompt(query)

            # Generate responses via vllm
            response = self.llm.generate(
                formatted_prompt,
                vllm.SamplingParams(
                    n=1,  # Number of output sequences to return for each prompt.
                    top_p=self.top_p,  # Float that controls the cumulative probability of the top tokens to consider.
                    top_k=self.top_k,
                    temperature=self.temperature,  # Randomness of the sampling
                    skip_special_tokens=True,  # Whether to skip special tokens in the output.
                    stop_token_ids=[self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")],
                    max_tokens=self.max_new_tokens  # Maximum number of tokens to generate per output sequence.
                ),
                use_tqdm=False # you might consider setting this to True during local development
            )

            return response.outputs[0].text

        elif isinstance(query, dict): # A batch of queries
            queries = query["question_text"]

            # Prepare formatted prompts from the generator
            formatted_prompts = self.make_llm_prompt_batch(queries)

            # Generate responses via vllm
            responses = self.llm.generate(
                formatted_prompts,
                vllm.SamplingParams(
                    n=1,  # Number of output sequences to return for each prompt.
                    top_p=self.top_p,  # Float that controls the cumulative probability of the top tokens to consider.
                    top_k=self.top_k,
                    temperature=self.temperature,  # Randomness of the sampling
                    skip_special_tokens=True,  # Whether to skip special tokens in the output.
                    stop_token_ids=[self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")],
                    max_tokens=self.max_new_tokens  # Maximum number of tokens to generate per output sequence.
                ),
                use_tqdm=False # you might consider setting this to True during local development
            )

            # Aggregate predictions into List[str]
            predictions = []
            for response in responses:
                predictions.append(response.outputs[0].text)
            
            return predictions

