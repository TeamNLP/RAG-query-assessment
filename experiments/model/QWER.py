import os
from typing import Union, Any, Dict, List

import torch
from model.classifier import Classifier
from model.prompt_templates import REWRITER_GENERAL_SYSTEM_PROMPT, REWRITE_PROMPT_METHOD_3, INPUT_TEMPLATE_METHOD_3
from model.RAG import RAGFramework, Retriever, Generator
from model.rewriter import Rewriter
from openai import OpenAI
from tqdm import tqdm

REWRITE_PROMPT = REWRITE_PROMPT_METHOD_3
INPUT_PROMPT = INPUT_TEMPLATE_METHOD_3
SYSTEM_PROMPT = REWRITER_GENERAL_SYSTEM_PROMPT

class QWERFramework(RAGFramework):
    def __init__(self, classifier, rewriter, retriever, generator):
        self.classifier = classifier
        self.rewriter = rewriter
        self.retriever = retriever
        self.generator = generator

    def run_framework(
        self, 
        query: Union[str, Dict[str, Any]]
    ) -> Union[str, List[str]]:

        if isinstance(query, str): # A single query
            label, score = self.classifier.classify(query)
            if label:
                response = self.rewriter.rewrite_query(query)
                rewritten_query = response.split("Rewritten Query:")[-1].strip()
                query = rewritten_query
            else:
                pass
        elif isinstance(query, dict): # A batch of queries
            queries = query["question_text"]
            for _idx, _query in tqdm(enumerate(queries), desc="Running Query Classifier and Rewriter"):
                label, score = self.classifier.classify(_query)
                if label:
                    response = self.rewriter.rewrite_query(_query)
                    rewritten_query = response.split("Rewritten Query:")[-1].strip()
                    queries[_idx] = rewritten_query
                else:
                    pass

        if self.generator.use_hf == True:
            output = self.run_hf(query)
        elif self.generator.use_hf == False:
            output = self.run_gpt(query)

        return output


def make_qwer_framework(args):
    openai_api_key = os.environ.get('OPENAI_API_KEY')
    openai_client = OpenAI(api_key=openai_api_key)

    # Classifier
    classifier = Classifier(args.classifier_model_name)
    
    # Rewriter
    rewriter = Rewriter(
        model_name=args.rewriter_model_name, 
        system_prompt=SYSTEM_PROMPT,
        rewrite_prompt=REWRITE_PROMPT, 
        input_prompt=INPUT_PROMPT, 
        max_new_tokens=args.rewriter_max_new_tokens, 
        openai_client=openai_client
    )

    # Retriever
    retriever = Retriever(
        corpus_name=args.retrieval_corpus_name, 
        top_n=args.retrieval_top_n, 
        api_url=args.retriever_api_url
    )

    # Generator
    if "gpt-3.5" in args.generator_model_name.lower() or "gpt-4" in args.generator_model_name.lower():
        pass
    else:
        openai_client = None

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

    # QWER framework 
    qwer_framework = QWERFramework(classifier, rewriter, retriever, generator)
    return qwer_framework