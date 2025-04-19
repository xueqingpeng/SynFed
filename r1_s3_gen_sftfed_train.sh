#!/bin/bash

# Directories
input_base_dir="syned_datasets/syn1_csv"
output_base_dir="syned_datasets/syn1_sft_fed"

mkdir -p "$output_base_dir"

# Dataset configurations (dataset name and number of clients)
declare -A datasets_config=(
    # ["va"]=1
    ["cleveland"]=1
    ["hungarian"]=1
    ["switzerland"]=1
    # ["switzerland_healthy"]=1
    # ["va_healthy"]=1
    # ["german"]=1
    # ["lendingclub"]=1
    # ["travel"]=1
)

# declare -A datasets_config=(
#     ["german"]=1
# )



split_type="train" 
split_method="none"

# Iterate through datasets and generate federated datasets
for ds_name in "${!datasets_config[@]}"; do
    num_clients=${datasets_config[$ds_name]}

    # Define input file path
    input_file="${input_base_dir}/${ds_name}.train.csv"

    # Check if input file exists
    if [[ ! -f "$input_file" ]]; then
        echo "Input file not found: $input_file. Skipping dataset: $ds_name."
        continue
    fi

    # Execute the Python script for the current dataset
    python -m src.data.sft_federated \
        --ds_name "$ds_name" \
        --num_client "$num_clients" \
        --input_file "$input_file" \
        --output_dir "$output_base_dir" \
        --split_type "$split_type" \
        --split_method "$split_method"

    echo "Federated ${split_type} dataset generated for: $ds_name with $num_clients clients."
done

echo "All ${split_type} datasets processed."
