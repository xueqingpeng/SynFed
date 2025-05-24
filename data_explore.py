import json
import pandas as pd

def print_first_valid_score(filename):
    """Prints details of the first successful entry with a valid gain score."""
    try:
        # Read and parse the JSONL file
        with open(filename, 'r') as f:
            data = [json.loads(line) for line in f]
        
        # Look for first valid entry
        for entry in data:
            # if entry["status"] == "SUCCESS" and not pd.isna(entry.get("gain_score")):
            if entry["status"] == "SUCCESS":
                print("Found valid entry:")
                for key, value in entry.items():
                    # print(f"{key}")
                    # print(f"{key}: {value}")
                    # if key == "reward_ds0" or key == "reward_ds1": print(f"{key}: {value}")
                    if key == "ds0" or key == "ds1": print(f"{key}: {value}")
                return
                
        print("No valid entries found.")
        
    except Exception as e:
        print(f"Error processing file: {e}")

# Run the analysis
if __name__ == "__main__":
    filename = "syned_datasets/syn1_score/cleveland.score.jsonl"
    filename = "syned_datasets/syn1_score/hungarian.score.jsonl"
    filename = "syned_datasets/syn1_score/switzerland.score.jsonl"
    print_first_valid_score(filename)