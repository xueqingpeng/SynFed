import os
import json
import argparse

def convert_to_alpaca_format(input_file, output_file):
    """
    Convert a JSONLines file with `input` and `output` fields to an Alpaca format JSON file.

    Args:
        input_file (str): Path to the input JSONLines file.
        output_file (str): Path to save the output Alpaca format JSON file.
    """
    alpaca_data = []

    # Read the input JSONLines file
    with open(input_file, 'r', encoding='utf-8') as infile:
        for line in infile:
            record = json.loads(line.strip())  # Parse each line as JSON

            # Construct the Alpaca format record
            alpaca_record = {
                "instruction": record.get("input", ""),
                "input": "",
                "output": record.get("output", "")
            }
            alpaca_data.append(alpaca_record)

    # Write the converted data to a single JSON file
    with open(output_file, 'w', encoding='utf-8') as outfile:
        json.dump(alpaca_data, outfile, ensure_ascii=False, indent=4)

    print(f"Conversion complete. Output saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Convert JSONLines to Alpaca format.")
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Path to the input directory containing .jsonl files."
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Path to the output directory where converted .json files will be saved."
    )

    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Process each .jsonl file in the input directory
    for filename in os.listdir(args.input_dir):
        if filename.endswith(".jsonl"):
            input_file = os.path.join(args.input_dir, filename)
            output_file = os.path.join(args.output_dir, filename.replace(".jsonl", ".json"))
            convert_to_alpaca_format(input_file, output_file)

if __name__ == "__main__":
    main()
