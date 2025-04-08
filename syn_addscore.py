from src.utils import read_jsonlines, write_jsonlines
from src.client import VllmClient
from src.metrics import get_scorer
from src.rewarder import DataManager
from src.data import SCHEMA_MAP, BaseDataSchema
import re
import argparse

from omegaconf import OmegaConf
import os


class ExtractAnswer:
    def __init__(self, schema):
        self.schema = schema

    def reg_extract(self, text):
        first_line = text.split("\n")[0]
        # Regex to match a line of digits, commas, and dots
        pattern = r'^\d[0-9,\.]*$'

        # Search for a match
        match = re.match(pattern, first_line)

        if match:
            # Remove commas and convert to float
            matched_content = match.group(0)
            # Replace commas with empty string, then convert to float
            return float(matched_content.replace(',', ''))
        else:
            return None  # Return None if no match is found

    def class_extract(self, text):
        # Extract all keys (choices) from the inverse_target_map
        choices = self.schema.inverse_target_map.keys()
        # Get the first line of the text
        first_line = text.split("\n")[0].strip()

        # Create a regex pattern to match any of the choices in the first line (case-insensitive)
        pattern = r'\b(' + '|'.join(map(re.escape, choices)) + r')\b'
        # Find all matches in the first line, ignoring case
        matches = re.findall(pattern, first_line, re.IGNORECASE)
        if len(matches) == 1:
            matched_val = matches[0]
            for choice in choices:
                if choice.lower() == matched_val.lower():
                    return choice
        return None

    def __call__(self, text):
        if self.schema.target_type == "regression":
            return self.reg_extract(text)
        elif self.schema.target_type == "classification":
            return self.class_extract(text)
        else:
            raise NotImplementedError


def add_pseudo_anwer(row, schema):
    result = row.get("result")
    if result is None:
        return row
    new_answer = schema.dict_to_jsonstr(result)
    row["answer"] = new_answer
    return row


def add_score_answer(raw_data, client):

    success_data_map = {}
    for idx, d in enumerate(raw_data):
        if d.get("status") == "SUCCESS":
            success_data_map[idx] = d

    success_data = list(success_data_map.values())
    data_manager = DataManager(success_data)
    flat_success_data = data_manager.get_flat_data()

    prompts = [d['question'] for d in flat_success_data]

    answers = client.generate(prompts)

    for answer, row in zip(answers, flat_success_data):
        row['answer'] = answer

    data_manager.update_flat_data(flat_success_data)
    data_manager.apply_changes()

    changed_success_data = data_manager.raw_data

    for idx, row in zip(success_data_map, changed_success_data):
        raw_data[idx] = row

    return raw_data


def extract_score_pairs(ds, extractor):
    golds = [d['gold'] for d in ds]
    answers = [d['answer'] for d in ds]
    answers = [extractor(answer) for answer in answers]
    score_pairs = (golds, answers)
    return score_pairs


def calculate_score(pairs, schema):
    task_type = schema.target_type
    assert task_type in ["classification", "regression"]
    scorer = get_scorer(task_type)

    golds = []
    preds = []
    missing = 0
    for g, a in zip(*pairs):
        if a is not None:
            if task_type == "regression":
                golds.append(schema.inverse_target_map(g))
                preds.append(schema.inverse_target_map(a))
            elif task_type == "classification":
                golds.append(schema.inverse_target_map.get(g))
                preds.append(schema.inverse_target_map.get(a))
        else:
            missing += 1
    missing = missing/len(pairs)

    return {"pairs": list(zip(*pairs)), "missing": missing, "score": scorer(golds, preds)}


def add_gain_score(raw_data, schema):
    extractor = ExtractAnswer(schema)
    for d in raw_data:
        if d.get("status") == "SUCCESS":
            reward_ds0 = d.get("reward_ds0")
            reward_ds1 = d.get("reward_ds1")

            score_pairs_ds0 = extract_score_pairs(reward_ds0, extractor)
            score_pairs_ds1 = extract_score_pairs(reward_ds1, extractor)

            score0 = calculate_score(score_pairs_ds0, schema)
            score1 = calculate_score(score_pairs_ds1, schema)

            d['ds0'] = score0
            d['ds1'] = score1
            gain = score1["score"] - score0["score"]
            d['gain_score'] = gain
    return raw_data


def add_status_score(raw_data):
    for d in raw_data:
        if d.get("status") == "SUCCESS":
            d['status_score'] = 5
        elif d.get("status") == "INVALID_ANSWER":
            d['status_score'] = 0
        elif d.get("status") == "REDUNDANT_KEY":
            d['status_score'] = 3
        elif d.get("status") == "MISSING_KEY":
            d['status_score'] = 2
        elif d.get("status") == "CONVERSION_ERROR":
            d['status_score'] = 1
        else:
            raise NotImplementedError
    return raw_data


def main(ds_names, input_path, output_path, client):
    """
    Process datasets by adding pseudo answers, score answers, and gain scores.

    Args:
        ds_names (list): List of dataset names to process.
        input_path (str): Path to the input JSONL files.
        output_path (str): Path where processed JSONL files will be saved.
        client (VllmClient): The VLLM model client.
    """
    os.makedirs(output_path, exist_ok=True)

    for ds_name in ds_names:
        input_file = os.path.join(input_path, f"{ds_name}.raw.jsonl")
        output_file = os.path.join(output_path, f"{ds_name}.score.jsonl")

        if not os.path.exists(input_file):
            print(f"Input file not found: {input_file}. Skipping {ds_name}.")
            continue

        print(f"Processing dataset: {ds_name}")

        schema = SCHEMA_MAP[ds_name]
        raw_data = read_jsonlines(input_file)

        # 1. Add pseudo answer for MISSING_KEY target
        for row in raw_data:
            if row.get("status") == "MISSING_KEY":
                row = add_pseudo_anwer(row, schema)

        # 2. Add answer scores for SUCCESS target
        raw_data = add_score_answer(raw_data, client)

        # 3. Add gain scores
        raw_data = add_gain_score(raw_data, schema)

        # 4. Add status scores
        raw_data = add_status_score(raw_data)

        # Save the processed data
        write_jsonlines(raw_data, output_file)
        print(f"Saved processed dataset: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process datasets and generate score JSONL files.")
    parser.add_argument("--ds_names", nargs="+", required=True, help="List of dataset names to process")
    parser.add_argument("--input_path", type=str, default="syned_datasets/syn1_raw", help="Input directory path")
    parser.add_argument("--output_path", type=str, default="syned_datasets/syn1_score", help="Output directory path")

    args = parser.parse_args()

    # Define the VLLM client configuration
    client_config = OmegaConf.create({
        'vllm_params': {
            'model': "TheFinAI/fl-dare_linear-1-merged-base",
            'tensor_parallel_size': 4,
            'trust_remote_code': True,
            'dtype': "bfloat16",
            'max_model_len': 4096,
            'enable_chunked_prefill': False,
            'seed': 42,
            'max_num_batched_tokens': 4096,
            'enable_prefix_caching': True,
            'enforce_eager': False,
            'gpu_memory_utilization': 0.95
        },
        'sampling_params': {
            'max_tokens': 64,
            'top_p': 0.7,
            'temperature': 0.5,
            'repetition_penalty': 1.05,
            'skip_special_tokens': False,
            'truncate_prompt_tokens': 4096,
            'include_stop_str_in_output': False
        }
    })
    
    with VllmClient(client_config) as client:
        # Run the main function with parsed arguments
        main(args.ds_names, args.input_path, args.output_path, client)
