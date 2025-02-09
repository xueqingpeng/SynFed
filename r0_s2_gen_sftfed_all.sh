#!/bin/bash

# Directories
raw_dataset_dir="datasets"
fed_dataset_dir="syned_datasets/original_0_sft_fed"
mkdir -p "$fed_dataset_dir"

# Dataset configurations (dataset name and number of clients)
# Modify this to control datasets and number of clients
declare -A datasets_config=(
    ["va"]=1
    ["cleveland"]=1
    ["hungarian"]=1
    ["switzerland"]=1
    # ["german"]=1
    # ["lendingclub"]=1
    # ["travel"]=1
)

# Define split types to process
split_types=("train" "test" "val")
split_method="none"

# Iterate over split types
for split_type in "${split_types[@]}"; do
    echo "Processing split type: $split_type"

    # Iterate through datasets and generate federated datasets
    for ds_name in "${!datasets_config[@]}"; do
        num_clients=${datasets_config[$ds_name]}

        # Define input file path
        input_file="${raw_dataset_dir}/${ds_name}/raw/${ds_name}_${split_type}.csv"

        # Define output base path
        output_dir="${fed_dataset_dir}"

        # Check if input file exists
        if [[ ! -f "$input_file" ]]; then
            echo "Input file not found: $input_file. Skipping dataset: $ds_name."
            continue
        fi

        # Execute the Python script for the current dataset and split type
        python -m src.data.sft_federated \
            --ds_name "$ds_name" \
            --num_client "$num_clients" \
            --input_file "$input_file" \
            --output_dir "$output_dir" \
            --split_type "$split_type" \
            --split_method "$split_method"

        echo "Federated $split_type dataset generated for: $ds_name with $num_clients clients."
    done

    echo "All $split_type datasets processed."
done

echo "All dataset splits (train, test, val) have been processed successfully."
