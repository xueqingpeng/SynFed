# %%
import os
import json
import random
import re

def get_reanchors(results_fp: str) -> list[dict]:
    with open(results_fp, "r") as f:
        results = json.load(f)

    tp = []
    fp = []
    tn = []
    fn = []
    for result in results:
        if result["gold"] == "healthy" and result["pred"] == "healthy":
            tp.append(result)
        elif result["gold"] == "healthy" and result["pred"] == "unhealthy":
            fp.append(result)
        elif result["gold"] == "unhealthy" and result["pred"] == "healthy":
            fn.append(result)
        elif result["gold"] == "unhealthy" and result["pred"] == "unhealthy":
            tn.append(result)
    print(f"TP: {len(tp)}, FP: {len(fp)}, TN: {len(tn)}, FN: {len(fn)}")

    reanchors = []
    reanchors.extend(random.sample(tp, 2))
    reanchors.extend(random.sample(fp, 2))
    reanchors.extend(random.sample(fn, 2))
    reanchors.extend(random.sample(tn, 2))
    print(f"got {len(reanchors)} reanchors")

    return reanchors


def extract_first_sample(text: str) -> str:
    """Extract the first 'Predict whether...Answer: [answer]' sample from text, stopping at the next question"""
    matches = re.findall(r"(Predict whether.*?Answer:\s*[^\n]*(?=\s*Predict whether|$))", text, re.DOTALL)
    return matches[0].strip() if matches else None


def reanchor_data(data_name: str):
    reanchors = get_reanchors(results_fp=f"results_{data_name}.json")
    reward_ds0 = [{"question": reanchor["prompt"], "gold": reanchor["gold"]} for reanchor in reanchors]
    input_fp = f"/gpfs/radev/home/xp83/Documents/project/scripts/SynFed/syned_datasets/syn1_raw_balanced/{data_name}_balanced.jsonl"
    output_fp = f"/gpfs/radev/home/xp83/Documents/project/scripts/SynFed/syned_datasets/syn1_raw_balanced_reanchors/{data_name}_reanchors.jsonl"

    modified_records = []
    with open(input_fp, "r") as f:
        for line in f:
            record = json.loads(line)
            if "reward_ds0" in record:
                record["reward_ds0"] = reward_ds0
            if "reward_ds1" in record:
                one_shot = extract_first_sample(record["reward_ds1"][0]['question'])
                record["reward_ds1"] = [{"question": one_shot + " \n\n" + i["question"], "gold": i["gold"]} for i in reward_ds0]
            modified_records.append(record)

    with open(output_fp, "w") as f:
        for record in modified_records:
            f.write(json.dumps(record) + "\n")

    print(f"Modified JSONL saved to: {output_fp}")


os.makedirs("/gpfs/radev/home/xp83/Documents/project/scripts/SynFed/syned_datasets/syn1_raw_balanced_reanchors", exist_ok=True)
data_names = [
    # "cleveland", 
    # "hungarian", 
    "switzerland"
]
for data_name in data_names:
    reanchor_data(data_name)











# %%
import json

fp = "/gpfs/radev/home/xp83/Documents/project/scripts/SynFed/syned_datasets/syn1_raw_balanced/switzerland_balanced.jsonl"
fp = "/gpfs/radev/home/xp83/Documents/project/scripts/SynFed/syned_datasets/syn1_raw_balanced_reanchors/switzerland_reanchors.jsonl"
with open(fp, "r") as f:
    for line in f:
        record = json.loads(line)
        if record["status"] == "SUCCESS":
            print(record["reward_ds1"])
            print("byrow")
            for i in record["reward_ds1"]:
                print(i)
            break

# %%
