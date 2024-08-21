import argparse
import random
import os

from tqdm import tqdm
from lib import read_jsonl, write_jsonl, find_matching_paragraph_text

random.seed(13370)  # Don't change this.

def main():

    parser = argparse.ArgumentParser(description="Save and sample data")
    parser.add_argument(
        "dataset_name", type=str, help="dataset name.", choices=("hotpotqa", "2wikimultihopqa", "musique", 'nq', 'trivia', 'squad', "ms_marco")
    )
    parser.add_argument("set_name", type=str, help="set name.", choices=("train_diff_size", "dev", "test", "dev_diff_size"))
    parser.add_argument("sample_size", type=int, help="sample_size")
    args = parser.parse_args()

    avoid_question_ids_file_path = None
    secondary_avoid_question_ids_file_path = None
    sample_size = 500
    if args.set_name == "test":
        input_file_path = os.path.join("processed_data", args.dataset_name, "dev.jsonl")
        dev_file_path = os.path.join("processed_data", args.dataset_name, "dev_subsampled.jsonl")
        avoid_question_ids_file_path = dev_file_path if os.path.exists(dev_file_path) else None
        sample_size = 500
    elif args.set_name == "dev_diff_size":
        input_file_path = os.path.join("processed_data", args.dataset_name, "dev.jsonl")
        avoid_question_ids_file_path = os.path.join("processed_data", args.dataset_name, "test_subsampled.jsonl")
        sample_size = args.sample_size
    elif args.set_name == "train_diff_size":
        input_file_path = os.path.join("processed_data", args.dataset_name, "train.jsonl")
        avoid_question_ids_file_path = os.path.join("processed_data", args.dataset_name, "test_subsampled.jsonl")
        dev_file_path = os.path.join("processed_data", args.dataset_name, "dev_subsampled.jsonl")
        secondary_avoid_question_ids_file_path = dev_file_path if os.path.exists(dev_file_path) else None
        sample_size = args.sample_size

    instances = read_jsonl(input_file_path)

    if avoid_question_ids_file_path:
        avoid_ids = set([avoid_instance["question_id"] for avoid_instance in read_jsonl(avoid_question_ids_file_path)])
        if secondary_avoid_question_ids_file_path:
            secondary_avoid_ids = set([avoid_instance["question_id"] for avoid_instance in read_jsonl(secondary_avoid_question_ids_file_path)])
            avoid_ids = avoid_ids | secondary_avoid_ids
        instances = [instance for instance in instances if instance["question_id"] not in avoid_ids]

    if args.set_name == "train_diff_size":
        instances = [instance for instance in instances if sum([context_dict["is_supporting"] for context_dict in instance["contexts"]]) > 0]

    instances = random.sample(instances, sample_size)

    for instance in tqdm(instances):
        for context in instance["contexts"]:

            if context in instance.get("pinned_contexts", []):
                # pinned contexts (iirc main) aren't in the associated wikipedia corpus.
                continue
            if args.set_name == "dev_diff_size":
                continue
            if args.set_name == "train_diff_size":
                continue

            if args.dataset_name in ['nq', 'trivia', 'squad', "ms_marco"]:
                # retrieved_result = find_matching_paragraph_text('wiki', context["paragraph_text"])
                continue
            else:
                retrieved_result = find_matching_paragraph_text(args.dataset_name, context["paragraph_text"])

            if retrieved_result is None:
                continue

            context["title"] = retrieved_result["title"]
            context["paragraph_text"] = retrieved_result["paragraph_text"]

    if args.set_name == "dev_diff_size":
        output_file_path = os.path.join("processed_data", args.dataset_name, f"dev_{args.sample_size}_subsampled.jsonl")
        write_jsonl(instances, output_file_path)
    if args.set_name == "train_diff_size":
        output_file_path = os.path.join("processed_data", args.dataset_name, f"train_{args.sample_size}_subsampled.jsonl")
        write_jsonl(instances, output_file_path)
    else:
        output_file_path = os.path.join("processed_data", args.dataset_name, f"{args.set_name}_subsampled.jsonl")
        write_jsonl(instances, output_file_path)


if __name__ == "__main__":
    main()
