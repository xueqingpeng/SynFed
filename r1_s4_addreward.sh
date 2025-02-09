#!/bin/bash

# Set environment variables
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export VLLM_WORKER_MULTIPROC_METHOD="spawn"
export NCCL_P2P_DISABLE="1"

# Define dataset names (modify this list as needed)
ds_names=("cleveland" "hungarian" "switzerland" "va" "switzerland_healthy" "va_healthy")

# Define input and output paths
input_path="syned_datasets/syn1_raw"
output_path="syned_datasets/syn1_score"

# Convert ds_names array into a space-separated string
ds_names_str="${ds_names[@]}"

# Run the Python script
python syn_addscore.py --ds_names $ds_names_str --input_path "$input_path" --output_path "$output_path"