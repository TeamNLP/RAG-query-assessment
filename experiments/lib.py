import os
import json
from typing import List, Dict

from sklearn.model_selection import StratifiedShuffleSplit

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


def hf_stratified_sampling(dataset, criterion, num_sample, seed=13370, verbose=True):
    stratified_sample = dataset.class_encode_column(criterion).train_test_split(
        train_size=num_sample,
        stratify_by_column=criterion,
        shuffle=True,
        seed=seed
    )['train']

    # To verify that the result is well stratified
    # Calculate the distribution of 'level_type' in the original dataset
    stratification_criterion = dataset.select_columns([criterion])[:][criterion]
    original_distribution = collections.Counter(stratification_criterion)
    original_total = sum(original_distribution.values())
    original_percentage = {k: (v, v / original_total) for k, v in original_distribution.items()}
    original_percentage = sorted(original_percentage.items(), key=lambda x:x[0])

    # Calculate the distribution of 'level_type' in the sampled dataset
    # sampled_criterion = [entry[criterion] for entry in stratified_sample]
    sampled_criterion = stratified_sample.select_columns([criterion])[:][criterion]
    sampled_distribution = collections.Counter(sampled_criterion)
    sampled_total = sum(sampled_distribution.values())
    sampled_percentage = {k: (v, v / sampled_total) for k, v in sampled_distribution.items()}
    sampled_percentage = sorted(sampled_percentage.items(), key=lambda x:x[0])

    if verbose:
        # Print the original and sampled distributions
        print("Original distribution (percent):")
        for k, (v, v_perc) in original_percentage:
            print(f"{k}: {v_perc*100:.2f}% ({v})")

        print("\nSampled distribution (percent):")
        for k, (v, v_perc) in sampled_percentage:
            print(f"{k}: {v_perc*100:.2f}% ({v})")

    return stratified_sample


def stratified_sampling(list_of_data, criterion, num_sample, seed=13370, verbose=True):

    # Extract stratification_criterion (e.g., 'level_type') for stratification
    stratification_criterion = [el[criterion] for el in list_of_data]

    # Define the stratified shuffle split
    # The 'n_splits=1' parameter indicates we want a single split,
    # and 'train_size=1000' specifies the desired sample size.
    split = StratifiedShuffleSplit(n_splits=1, train_size=num_sample, random_state=seed)

    # Perform the split and get the indices
    for train_index, _ in split.split(list_of_data, stratification_criterion):
        stratified_sample = [list_of_data[idx] for idx in train_index]

    if verbose:
        print("Sampled Size :", len(stratified_sample) )
        print("Sampled check :", [i for i in stratified_sample[:3]] )

    # To verify that the result is well stratified
    # Calculate the distribution of 'level_type' in the original dataset
    original_distribution = collections.Counter(stratification_criterion)
    original_total = sum(original_distribution.values())
    original_percentage = {k: (v, v / original_total) for k, v in original_distribution.items()}
    original_percentage = sorted(original_percentage.items(), key=lambda x:x[0])

    # Calculate the distribution of 'level_type' in the sampled dataset
    sampled_criterion = [entry[criterion] for entry in stratified_sample]
    sampled_distribution = collections.Counter(sampled_criterion)
    sampled_total = sum(sampled_distribution.values())
    sampled_percentage = {k: (v, v / sampled_total) for k, v in sampled_distribution.items()}
    sampled_percentage = sorted(sampled_percentage.items(), key=lambda x:x[0])

    if verbose:
        # Print the original and sampled distributions
        print("Original distribution (percent):")
        for k, (v, v_perc) in original_percentage:
            print(f"{k}: {v_perc*100:.2f}% ({v})")

        print("\nSampled distribution (percent):")
        for k, (v, v_perc) in sampled_percentage:
            print(f"{k}: {v_perc*100:.2f}% ({v})")

    return stratified_sample