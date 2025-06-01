#!/bin/bash

# Define the round variable
round="syn1"

# Define input and output paths
input_path="syned_datasets/${round}_score"
output_path="syned_datasets/${round}_dpo"

# Define dataset names (modify as needed)
# ds_names=("cleveland" "hungarian" "switzerland" "va" "switzerland_healthy" "va_healthy")
ds_names=("cleveland" "hungarian" "switzerland")

# Convert ds_names array into a space-separated string
ds_names_str="${ds_names[@]}"

# Run the Python script
python convert_score_to_prefer.py --input_path "$input_path" \
                             --output_path "$output_path" \
                             --ds_names $ds_names_str
