#!/bin/bash

generator_model_name=gpt-4o-2024-08-06

# echo "Upload"
# python estimator/dataset_construction/query_wise_label_by_evaluation_assistant.py \
#     --generator_model_name $generator_model_name \
#     --input_directory source_dataset\
#     --output_directory predictions/$generator_model_name \
#     --dataset_type train_18000_subsampled \
#     --openai_batch_api upload

echo "Check"
for batch_id in batch_hvIvXG0RFtTf3lyz6WygCZac batch_9iri4d4fSF5LyrSRw6qhhggg batch_uYUiOy5Memn0tO1xROJcqkHu batch_2nmhGKgVLlRAbqbDMP4lETV8 batch_oOAkrg3h3fEgokyL7ayG3552 batch_jkSF4ZsuEqKfdWWLqaQ65ni0 batch_QOWvgQjq8E4kwl7fq5h82JxU batch_0haaPhzYn7sEvij5omtBxCEf batch_f4VrXTAeakSuQ5LoLkbHabae batch_NlfLhpunnQIEMllKrD8DvAKb batch_amx7qSWvRqztQAKrPLbX0Tv1 batch_d4wj9BxhnurYOMOR0IcNfkWb batch_FfnsXKtcoMkR6jSg5wvi9Mj9 batch_nb6CjolRoeoBCg289DIwWDuO batch_4NdUuUQxvPQlQ6mWra5bCcDY batch_urLXnqZZbIGne4DXUtarAhde batch_n6U9IhUlV8F4stAHCvXg15eN batch_rwVXzJL0O7Qx7MSxYUwGN0Es batch_BAHyaEZU1yBn2J2CfD3nz7qI batch_2Y3FaWryIVFZ2Iz9ENCXvrt0 batch_ralegpZBp0IWNPKrnzvIcEPJ batch_cehIlmQhjD11IficpVU0IqFP batch_cZYFEyTTymOUL9YbpkO4eYcm batch_DPcbJN4xdcGDt5uRSCnFehJV batch_cofHjL8wklrlqaUVFxjTWTIW batch_yRMGNf4caPw8LgBXtWqYqjTT batch_tNZ40J90RHF02zIPuYCnFohv batch_Cc7ziMTFDW00kB7DsuyQTBnP
do
    python estimator/dataset_construction/query_wise_label_by_evaluation_assistant.py \
        --generator_model_name $generator_model_name \
        --input_directory source_dataset\
        --output_directory predictions/$generator_model_name \
        --dataset_type train_18000_subsampled \
        --openai_batch_api check \
        --openai_batch_id $batch_id
done