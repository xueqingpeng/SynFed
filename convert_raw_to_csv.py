import jsonlines
import pandas as pd
import argparse

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Convert JSONL file to CSV format.")
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to the input JSONL file.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to the output CSV file.",
    )
    
    # Parse arguments
    args = parser.parse_args()
    input_file = args.input_file
    output_file = args.output_file
    
    # Read JSONL file
    with jsonlines.open(input_file) as reader:
        data = list(reader)
    
    # Extract 'result' field
    results = []
    for d in data:
        r = d.get('result')  # Use .get() to avoid errors if 'result' key is missing
        if r:
            results.append(r)
    
    # Convert to DataFrame and save as CSV
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    
if __name__ == "__main__":
    main()
