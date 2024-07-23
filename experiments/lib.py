import os
import json
from typing import List, Dict
# from pathlib import Path

# import _jsonnet
# import requests
from datasets import load_dataset


def load_data(input_filepath):
    dataset = load_dataset("json", data_files=input_filepath)
    return dataset


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


# def get_retriever_address(suffix: str = ""):
#     retriever_address_config_filepath = ".retriever_address.jsonnet"
#     if not os.path.exists(retriever_address_config_filepath):
#         raise Exception(f"Retriver address filepath ({retriever_address_config_filepath}) not available.")
#     retriever_address_config_ = json.loads(_jsonnet.evaluate_file(retriever_address_config_filepath))
#     retriever_address_config = {
#         "host": retriever_address_config_["host" + suffix],
#         "port": retriever_address_config_["port" + suffix],
#     }
#     return retriever_address_config


# def get_llm_server_address(llm_port_num : str):
#     llm_server_address_config_filepath = ".llm_server_address.jsonnet"
#     if not os.path.exists(llm_server_address_config_filepath):
#         raise Exception(f"LLM Server address filepath ({llm_server_address_config_filepath}) not available.")
#     llm_server_address_config = json.loads(_jsonnet.evaluate_file(llm_server_address_config_filepath))
#     llm_server_address_config = {key: str(value) for key, value in llm_server_address_config.items()}
#     # TODO
#     #import pdb; pdb.set_trace()
#     llm_server_address_config['port'] = llm_port_num
#     return llm_server_address_config


# def get_config_file_path_from_name_or_path(experiment_name_or_path: str) -> str:
#     if not experiment_name_or_path.endswith(".jsonnet"):
#         # It's a name
#         assert (
#             len(experiment_name_or_path.split(os.path.sep)) == 1
#         ), "Experiment name shouldn't contain any path separators."
#         matching_result = list(Path(".").rglob("**/*" + experiment_name_or_path + ".jsonnet"))
#         matching_result = [
#             _result
#             for _result in matching_result
#             if os.path.splitext(os.path.basename(_result))[0] == experiment_name_or_path
#         ]
#         matching_result = [i for i in matching_result if 'backup' not in str(i)]
#         #import pdb; pdb.set_trace()
#         assert len(matching_result) == 1 
        
#         if len(matching_result) != 1:
#             #import pdb; pdb.set_trace()
#             exit(f"Couldn't find one matching path with the given name ({experiment_name_or_path}).")
#         config_filepath = matching_result[0]
#     else:
#         # It's a path
#         config_filepath = experiment_name_or_path
#     return config_filepath