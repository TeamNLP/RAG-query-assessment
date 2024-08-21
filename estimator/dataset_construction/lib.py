import os
import json
from typing import List, Dict

import tiktoken
from model.prompt_templates import Uncertainty_SYS_PROMPT, Complexity_SYS_PROMPT, Redundancy_SYS_PROMPT, Subjectivity_SYS_PROMPT
from model.prompt_templates import Uncertainty_USER_PROMPT, Complexity_USER_PROMPT, Redundancy_USER_PROMPT, Subjectivity_USER_PROMPT


def make_request_dict(question_id, question_text, generator=None, passages=None, evaluation_type=None):
    encoding = tiktoken.get_encoding("o200k_base")

    if generator is None:
        assert evaluation_type is not None
        custom_id_prefix = f"{evaluation_type} Evaluation Assistant_"

        if evaluation_type.lower() == "uncertainty":
            system_prompt = Uncertainty_SYS_PROMPT
            user_prompt = Uncertainty_USER_PROMPT
        elif evaluation_type.lower() == "complexity":
            system_prompt = Complexity_SYS_PROMPT
            user_prompt = Complexity_USER_PROMPT
        elif evaluation_type.lower() == "redundancy":
            system_prompt = Redundancy_SYS_PROMPT
            user_prompt = Redundancy_USER_PROMPT
        elif evaluation_type.lower() == "subjectivity":
            system_prompt = Subjectivity_SYS_PROMPT
            user_prompt = Subjectivity_USER_PROMPT
        else:
            raise Exception(f"Unexpected evaluation_type: {evaluation_type}")

        formatted_chat_template = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt.format(question=question_text)
            }
        ]

        request_dict = {
            "custom_id": f"{custom_id_prefix}{question_id}", 
            "method": "POST", 
            "url": "/v1/chat/completions", 
            "body": {
                "model": "gpt-4o-2024-08-06", 
                "messages": formatted_chat_template,
                "max_tokens": 100,
                "temperature": 0.1,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "reasoning_schema",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "reasoning_steps": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    },
                                    "description": "The reasoning steps leading to the final conclusion."
                                },
                                "answer": {
                                    "type": "string",
                                    "description": "The final answer, taking into account the reasoning steps."
                                }
                            },
                            "required": ["reasoning_steps", "answer"],
                            "additionalProperties": False
                        }
                    }
                }
            }
        }
    else:
        if passages is None:
            custom_id_prefix = "LLM_"
            formatted_chat_template = generator.make_llm_prompt(question_text)
        else:
            custom_id_prefix = "RAG_"
            if isinstance(passages, list):
                passages = "\n".join(passages)
            formatted_chat_template = generator.make_rag_prompt(question_text, passages)
            
        request_dict = {
            "custom_id": f"{custom_id_prefix}{question_id}", 
            "method": "POST", 
            "url": "/v1/chat/completions", 
            "body": {
                "model": generator.model_name, 
                "messages": formatted_chat_template,
                "max_tokens": generator.max_new_tokens,
                "temperature": generator.temperature
            }
        }

    return request_dict, len(encoding.encode(formatted_chat_template[0]["content"]))+len(encoding.encode(formatted_chat_template[1]["content"]))
    

def read_json(file_path: str) -> Dict:
    with open(file_path, "r", encoding="utf8", errors='ignore') as file:
        instance = json.load(file)
    return instance


def read_jsonl(file_path: str) -> List[Dict]:
    with open(file_path, "r") as file:
        instances = [json.loads(line.strip()) for line in file.readlines() if line.strip()]
    return instances


def write_json(instance: Dict, file_path: str):
    with open(file_path, "w") as file:
        json.dump(instance, file)


def write_jsonl(instances: List[Dict], file_path: str):
    with open(file_path, "w") as file:
        for instance in instances:
            file.write(json.dumps(instance) + "\n")


CORPUS_NAME_DICT = {
    "hotpotqa":"hotpotqa",
    "2wikimultihopqa":"2wikimultihopqa",
    "musique":"musique",
    'nq':'wiki',
    'trivia':'wiki',
    'squad':'wiki',
    'ms_marco':None
}
