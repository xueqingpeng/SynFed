import os
import argparse
import re
import json
import pandas as pd
import jsonlines
from itertools import combinations
from datasets import Dataset

def load_jsonl_to_dataframe(input_file):
    """Load a JSONL file into a Pandas DataFrame."""
    # with jsonlines.open(input_file) as reader:
    #     data = list(reader)
    
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            cleaned_line = re.sub(r'\bNaN\b', 'null', line)
            try:
                obj = json.loads(cleaned_line)
                data.append(obj)
            except json.JSONDecodeError as e:
                print("Error line: ", cleaned_line)
                raise e
            
    df = pd.DataFrame(data)
    
    # Fill NaN values for necessary columns
    df['status_score'] = df['status_score'].fillna(0)
    df['gain_score'] = df['gain_score'].fillna(0)
    
    return df

def filter_successful_rows(df):
    """Filter rows where at least one row in the group has status_score == 5."""
    return df.groupby('syn_pmt_id').filter(lambda x: (x['status_score'] == 5).any())

def generate_index_combinations(group):
    """Generate index combinations within a grouped DataFrame."""
    indices = group.index.tolist()
    return list(combinations(indices, 2))

def prefer_r1_than_r2(row1, row2):
    """Determine preference between two rows based on status_score and gain_score."""
    if row1['status_score'] == 5 and row2['status_score'] == 5:
        return row1['gain_score'] > row2['gain_score']
    else:
        return row1['status_score'] > row2['status_score']

def reformat_data(row):
    """Reformat row data into a structured format."""
    return [
        {"content": row['prompt'], "role": "user"},
        {"content": row['answer'], "role": "assistant"}
    ]

def generate_preference_dataset(df):
    """Generate preference dataset based on row comparisons."""
    sucs_df = filter_successful_rows(df)

    # Generate index combinations for each group
    index_combinations = sucs_df.groupby('syn_pmt_id').apply(
        lambda group: generate_index_combinations(group)
    ).explode().dropna().tolist()

    prefer_dataset = []
    for indices in index_combinations:
        idx1, idx2 = indices
        row1, row2 = df.iloc[idx1], df.iloc[idx2]

        # Determine preferred and rejected row
        if prefer_r1_than_r2(row1, row2):
            chosen, reject = row1, row2
        else:
            chosen, reject = row2, row1

        chosen_score = chosen['gain_score'] + chosen['status_score']
        reject_score = reject['gain_score'] + reject['status_score']

        # Skip if scores are the same
        if chosen_score == reject_score:
            continue

        prefer_dataset.append({
            "chosen": reformat_data(chosen),
            "reject": reformat_data(reject),
            "chosen_score": chosen_score,
            "reject_score": reject_score
        })

    return pd.DataFrame(prefer_dataset)

def save_dataset(df, output_dir):
    """Save the preference dataset to a dataset-specific directory."""
    os.makedirs(output_dir, exist_ok=True)
    dataset = Dataset.from_pandas(df)
    dataset.save_to_disk(output_dir)
    print(f"Dataset saved at: {output_dir}")

def save_jsonl(df, output_file):
    """Save DataFrame to JSONL format."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with jsonlines.open(output_file, mode='w') as writer:
        writer.write_all(df.to_dict(orient='records'))
    print(f"JSONL file saved at: {output_file}")

def main(input_path, output_path, ds_names):
    """
    Process multiple datasets to generate preference datasets.
    
    Args:
        input_path (str): Path to input JSONL files.
        output_path (str): Path where output datasets will be saved.
        ds_names (list): List of dataset names to process.
    """
    for ds_name in ds_names:
        input_file = os.path.join(input_path, f"{ds_name}.score.jsonl")
        output_ds_dir = os.path.join(output_path, ds_name)  # Save as directory
        output_jsonl_file = os.path.join(output_ds_dir, f"{ds_name}.pref.jsonl")  # JSONL file in same directory

        if not os.path.exists(input_file):
            print(f"Input file not found: {input_file}. Skipping {ds_name}.")
            continue

        print(f"Processing dataset: {ds_name}")

        df = load_jsonl_to_dataframe(input_file)
        prefer_df = generate_preference_dataset(df)

        # sample 10000 rows
        prefer_df = prefer_df.sample(n=10000, random_state=42)

        save_dataset(prefer_df, output_ds_dir)
        
        # Save JSONL file in the same directory
        save_jsonl(prefer_df, output_jsonl_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process datasets to generate preference datasets in directories.")
    parser.add_argument("--input_path", type=str, required=True, help="Directory containing input JSONL files.")
    parser.add_argument("--output_path", type=str, required=True, help="Directory to save output datasets.")
    parser.add_argument("--ds_names", nargs="+", required=True, help="List of dataset names to process.")

    args = parser.parse_args()
    
    main(args.input_path, args.output_path, args.ds_names)
