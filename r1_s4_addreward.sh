#!/bin/bash

source .env
echo "HF_TOKEN: $HF_TOKEN" | cut -c1-20

# Set environment variables
export CUDA_VISIBLE_DEVICES="0,1"
export VLLM_WORKER_MULTIPROC_METHOD="spawn"
export NCCL_P2P_DISABLE="1"

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
    --model "ShawnXiaoyuWang/FedMerged-5-5-2025" \
    --tensor_parallel_size 2