cd experiments

# Process raw data files in a single standard format
python processing/process_nq.py
python processing/process_trivia.py
python processing/process_squad.py
python processing/process_ms_marco.py

# Subsample the processed datasets
for dataset_name in nq trivia squad ms_marco
do
    python processing/subsample_dataset_and_remap_paras.py $dataset_name test 500
    python processing/subsample_dataset_and_remap_paras.py $dataset_name dev_diff_size 500
done

# Subsample the processed datasets
for dataset_name in hotpotqa 2wikimultihopqa musique nq trivia squad ms_marco
do
    python processing/subsample_dataset_and_remap_paras.py $dataset_name train_diff_size 18000
done

cd ..

for dataset_name in hotpotqa 2wikimultihopqa musique nq trivia squad ms_marco
do
    mkdir estimator/dataset_construction/source_dataset/$dataset_name
    mv experiments/processed_data/$dataset_name/train_18000_subsampled.jsonl estimator/dataset_construction/source_dataset/$dataset_name/train_18000_subsampled.jsonl
done