import os
import json
from typing import List, Dict
from pathlib import Path

from rapidfuzz import fuzz
import requests


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


def find_matching_paragraph_text(corpus_name: str, original_paragraph_text: str) -> str:

    retriever_address_config = get_retriever_address()
    retriever_host = str(retriever_address_config["host"])
    retriever_port = str(retriever_address_config["port"])

    params = {
        "query_text": original_paragraph_text,
        "retrieval_method": "retrieve_from_elasticsearch",
        "max_hits_count": 1,
        "corpus_name": corpus_name,
    }

    url = retriever_host.rstrip("/") + ":" + str(retriever_port) + "/retrieve"
    result = requests.post(url, json=params)

    if not result.ok:
        print("WARNING: Something went wrong in the retrieval. Skiping this mapping.")
        return None

    result = result.json()
    retrieval = result["retrieval"]

    for item in retrieval:
        assert item["corpus_name"] == corpus_name

    retrieved_title = retrieval[0]["title"]
    retrieved_paragraph_text = retrieval[0]["paragraph_text"]

    match_ratio = fuzz.partial_ratio(original_paragraph_text, retrieved_paragraph_text)
    #import pdb; pdb.set_trace()
    if match_ratio > 95:
        return {"title": retrieved_title, "paragraph_text": retrieved_paragraph_text}
    else:
        print(f"WARNING: Couldn't map the original paragraph text to retrieved one ({match_ratio}).")
        return None
