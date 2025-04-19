import json
import random
from pathlib import Path
from collections import Counter


dataset_files = {
    'cleveland': ["cleveland.raw.jsonl"],
    'hungarian': ["hungarian.raw.jsonl"],
    'switzerland': ["switzerland.raw.jsonl", "switzerland_healthy.raw.jsonl"]
}

base_path = Path("./syned_datasets/syn1_raw")
output_path = Path("./syned_datasets/syn1_raw_balanced")
output_path.mkdir(parents=True, exist_ok=True)


def load_jsonl(file_path):
    with open(file_path, "r") as f:
        return [json.loads(line) for line in f]

def save_jsonl(data, file_path):
    with open(file_path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


for name, files in dataset_files.items():
    all_data = []
    for file in files:
        all_data.extend(load_jsonl(base_path / file))
    
    # 只保留标签为 healthy 或 heartdisease 的数据
    filtered_data = [item for item in all_data if item["result"].get("heartdiseaseStatus") in ["healthy", "heartdisease"]]

    # 按类别分组
    data_by_label = {
        "healthy": [item for item in filtered_data if item["result"]["heartdiseaseStatus"] == "healthy"],
        "heartdisease": [item for item in filtered_data if item["result"]["heartdiseaseStatus"] == "heartdisease"]
    }

    # 显示原始分布
    print(f"Before balancing - {name}: { {k: len(v) for k, v in data_by_label.items()} }")

    # 找到最小类的样本数
    min_count = min(len(data_by_label["healthy"]), len(data_by_label["heartdisease"]))

    # 下采样多数类
    balanced_data = random.sample(data_by_label["healthy"], min_count) + \
                    random.sample(data_by_label["heartdisease"], min_count)

    # 打乱顺序
    random.shuffle(balanced_data)

    # 显示新分布
    new_counter = Counter(item["result"]["heartdiseaseStatus"] for item in balanced_data)
    print(f"After balancing - {name}: {dict(new_counter)}")

    # 保存结果
    save_jsonl(balanced_data, output_path / f"{name}_balanced.jsonl")
    