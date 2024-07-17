cd experiments

# Process raw data files in a single standard format
python processing/process_nq.py
python processing/process_trivia.py
python processing/process_squad.py

# Subsample the processed datasets
for dataset_name in nq trivia squad
do
    python processing/subsample_dataset_and_remap_paras.py $dataset_name test 500
    python processing/subsample_dataset_and_remap_paras.py $dataset_name dev_diff_size 500
done


# The resulting experiments/processed_data/ directory should look like:
# .
# ├── nq
# │   ├── dev.jsonl
# │   ├── dev_500_subsampled.jsonl
# │   ├── test_subsampled.jsonl
# │   └── train.jsonl
# ├── squad
# │   ├── dev.jsonl
# │   ├── dev_500_subsampled.jsonl
# │   ├── test_subsampled.jsonl
# │   └── train.jsonl
# └── trivia
#     ├── dev.jsonl
#     ├── dev_500_subsampled.jsonl
#     ├── test_subsampled.jsonl
#     └── train.jsonl