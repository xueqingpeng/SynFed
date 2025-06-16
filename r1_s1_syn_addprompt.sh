#!/bin/bash

# List of datasets to process
# datasets=("german" "adult" "diabetes" )
# datasets=("lendingclub" "travel" "german" "cleveland" "hungarian" "switzerland" "va")
# datasets=("cleveland" "hungarian" "switzerland" "va" "switzerland_healthy" "va_healthy")
datasets=(
    "cleveland" 
    "hungarian" 
    "switzerland" 
    "switzerland_healthy"
)

# Iterate through datasets and run the script for each
for ds_name in "${datasets[@]}"; do
    python syn_addprompt.py ds_name=$ds_name debug=false
    echo "Processing completed for dataset: $ds_name"
done

echo "All datasets processed."
