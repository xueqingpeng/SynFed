#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input_folder> <output_folder>"
    exit 1
fi

INPUT_FOLDER=$1
OUTPUT_FOLDER=$2

# Create the output folder if it doesn't exist
mkdir -p "$OUTPUT_FOLDER"

# Iterate over all JSONLines files in the input folder
for input_file in "$INPUT_FOLDER"/*.jsonl; do
    # Skip if no JSONLines files are found
    if [ ! -f "$input_file" ]; then
        echo "No JSONLines files found in $INPUT_FOLDER."
        exit 1
    fi

    # Get the base name of the file (without the folder path) and change extension to .json
    base_name=$(basename "$input_file" .jsonl).json

    # Define the output file path in the output folder
    output_file="$OUTPUT_FOLDER/$base_name"

    # Run the Python conversion script
    python jsonline_to_alpaca.py "$input_file" --output "$output_file"

    echo "Converted: $input_file -> $output_file"
done

echo "All files converted successfully. Output saved to $OUTPUT_FOLDER."
