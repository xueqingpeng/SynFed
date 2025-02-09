#!/bin/bash

# Define the working directory
working_directory="./"

# List of source datasets
# source_datasets=("lendingclub" "travel" "german" "cleveland" "hungarian" "switzerland" "va" "switzerland_healthy" "va_healthy")
source_datasets=("cleveland" "hungarian" "switzerland" "va" "switzerland_healthy" "va_healthy")
# source_datasets=("switzerland_healthy" "va_healthy")
# Base command
base_command="python -m src.data.sft_generator"

# Function to run a single job
run_job() {
    local target_dataset=$1
    local sft_dataset_dir="${working_directory}/syn_datasets/sft_syn_${target_dataset}"

    # Change to the working directory and run the command
    (
        cd "$working_directory"
        $base_command target_datasets="[$target_dataset]" sft_dataset_dir="$sft_dataset_dir"
    ) &
}

# Launch jobs for all source datasets in parallel
for target_dataset in "${source_datasets[@]}"; do
    run_job "$target_dataset"
done

# Wait for all background jobs to finish
wait

echo "All jobs have completed."
