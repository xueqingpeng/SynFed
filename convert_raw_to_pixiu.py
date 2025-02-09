import os
from copy import deepcopy

from src.data import SCHEMA_MAP
from src.rewarder import DataManager
from src.utils import read_jsonlines, write_jsonlines


def modify_data(data, schema):
    for idx, d in enumerate(data):
        d["query"] = d.pop("question")
        d["id"] = idx
        if schema.target_type == "regression":
            d["answer"] = float(d.pop("gold"))
        elif schema.target_type == "classification":
            d["choices"] = list(schema.inverse_target_map.keys())
            d["answer"] = d["gold"]
            d['gold'] = schema.inverse_target_map[d['gold']]
    return data

def main():
    for ds_name in ("abalone", "adult", "buddy", "california", "diabetes", "insurance"):
        
        print(f"Processing {ds_name}...")
        
        raw_data = read_jsonlines(f"syned_datasets/raw/{ds_name}_syn.jsonl")
        
        success_data = [d for d in raw_data if d["status"] == "SUCCESS" ]
        
        if not success_data:
            print(f"No success data for {ds_name}")
            continue
        
        data = DataManager(success_data).get_flat_data()
        schema = SCHEMA_MAP[ds_name]
        data = modify_data(data, schema)
        save_path = f"syned_datasets/pixiu/{ds_name}"
        os.makedirs(save_path, exist_ok=True)

        #TODO:remove, only for dubug
        data = data[:50]
        write_jsonlines(data, os.path.join(save_path, "test.json"))


if __name__=="__main__":
    main()