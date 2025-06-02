#!/bin/bash

# Define the round variable
round="syn1"

# Define dataset names (modify this list as needed)
# ds_names=("cleveland" "hungarian" "switzerland" "va" "switzerland_healthy" "va_healthy")
ds_names=("cleveland" "hungarian" "switzerland")

# Define input and output paths
input_path="syned_datasets/${round}_raw_balanced"
output_path="syned_datasets/${round}_score"

# Convert ds_names array into a space-separated string
ds_names_str="${ds_names[@]}"

# Run the Python script
python syn_addscore.py \
    --ds_names $ds_names_str \
    --input_path "$input_path" \
    --output_path "$output_path" \
    --model "TheFinAI/fl-magnitude_prune-1-sft-merged-base-62" \
    --tensor_parallel_size "$N_CUDA"