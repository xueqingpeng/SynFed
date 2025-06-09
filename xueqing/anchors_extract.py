import os
import json
import re


def extract_last_sample(text: str) -> str:
    """Extract the last 'Predict whether...Answer:' sample from text"""
    matches = re.findall(r"(Predict whether.*?Answer:)", text, re.DOTALL)
    return matches[-1].strip() if matches else None


def load_success_record(dir: str, fp: str) -> dict:
    with open(os.path.join(dir, fp), "r") as f:
        for line in f:
            if not line.strip():
                return None
            record = json.loads(line)
            if record["status"] == "SUCCESS":
                return record


def gen_test_data(record: dict) -> tuple[list[dict], list[dict]]:
    data = record.get("reward_ds0", [])
    test_few_shot = []
    test_zero_shot = []
    for item in data:
        # Process for few-shot
        test_few_shot.append(item)
            
        # Process for zero-shot
        processed_item = item.copy()
        processed_item["question"] = extract_last_sample(processed_item["question"])
        test_zero_shot.append(processed_item)
    return test_few_shot, test_zero_shot


def print_test_data(test_few_shot: list[dict], test_zero_shot: list[dict]):
    print("Few-shot:")
    for item in test_few_shot:
        print(item)
    print("Zero-shot:")
    for item in test_zero_shot:
        print(item)


# Prepare test data
data_names = ["cleveland", "hungarian", "switzerland"]
dir = "/gpfs/radev/home/xp83/Documents/project/scripts/SynFed/syned_datasets/syn1_raw_balanced"
test_few_shot_all = {}
test_zero_shot_all = {}
for data_name in data_names:
    print(f"Processing data: {data_name}")
    record = load_success_record(dir, f"{data_name}_balanced.jsonl")
    test_few_shot, test_zero_shot = gen_test_data(record)
    print_test_data(test_few_shot,test_zero_shot)
    test_few_shot_all[data_name] = test_few_shot
    test_zero_shot_all[data_name] = test_zero_shot
