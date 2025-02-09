#!/bin/bash

# Define the round variable
round="syn1"

# List of datasets
# datasets=("diabetes" "adult" "german")
# datasets=("lendingclub" "travel" "german" "cleveland" "hungarian" "switzerland" "va")
# datasets=("cleveland" "hungarian" "switzerland" "va" "switzerland_healthy" "va_healthy")
datasets=("cleveland" "hungarian" "switzerland" "va" "switzerland_healthy" "va_healthy")

# Iterate over datasets and process each one
for dataset in "${datasets[@]}"; do
    # Define input and output paths dynamically
    input_file="syned_datasets/${round}_raw/${dataset}.raw.jsonl"
    output_file="syned_datasets/${round}_csv/${dataset}.train.csv"

    mkdir -p syned_datasets/${round}_csv

    # Check if the input file exists
    if [[ ! -f "$input_file" ]]; then
        echo "Input file not found: $input_file. Skipping dataset: $dataset."
        continue
    fi

    # Execute the Python script
    python convert_raw_to_csv.py \
        --input_file "$input_file" \
        --output_file "$output_file"

    echo "Processed dataset: $dataset (Input: $input_file, Output: $output_file)"
done

echo "All datasets processed."
