cd experiments

# Subsample the processed datasets
for dataset_name in hotpotqa 2wikimultihopqa musique nq trivia squad
do
    python processing/subsample_dataset_and_remap_paras.py $dataset_name train_diff_size 18000
done