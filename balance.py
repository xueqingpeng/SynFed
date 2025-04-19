import pandas as pd
import json
from sklearn.utils import resample
from pathlib import Path

# 原始文件路径结构
dataset_files = {
    'cleveland': ["cleveland_client_0.train.jsonl"],
    'hungarian': ["hungarian_client_0.train.jsonl"],
    'switzerland': ["switzerland_client_0.train.jsonl", "switzerland_healthy_client_0.train.jsonl"],
    'va': ["va_client_0.train.jsonl", "va_healthy_client_0.train.jsonl"]
}

base_path = Path("/home/xpeng3/SynFed/syned_datasets/syn2_sft_fed/")
output_path = Path("/home/xpeng3/SynFed/syned_datasets/syn2_sft_fed_balanced/")
output_path.mkdir(parents=True, exist_ok=True)

def load_jsonl(file_path):
    with open(file_path, "r") as f:
        return [json.loads(line) for line in f]

def save_jsonl(data, file_path):
    with open(file_path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

# 处理每个 dataset
for name, files in dataset_files.items():
    all_data = []
    for file in files:
        all_data.extend(load_jsonl(base_path / file))
    
    df = pd.DataFrame(all_data)

    # 仅保留 "healthy" 和 "unhealthy"
    df = df[df["output"].isin(["healthy", "unhealthy"])]

    # 查看原始分布
    print(f"Before balancing - {name}:\n{df['output'].value_counts()}\n")

    # 下采样多数类
    min_count = df["output"].value_counts().min()
    balanced = []
    for label in ["healthy", "unhealthy"]:
        sampled = resample(df[df["output"] == label], replace=False, n_samples=min_count, random_state=42)
        balanced.append(sampled)

    df_balanced = pd.concat(balanced, ignore_index=True)

    # 查看平衡后的分布
    print(f"After balancing - {name}:\n{df_balanced['output'].value_counts()}\n")

    # 保存为 JSONL
    save_jsonl(df_balanced.to_dict(orient="records"), output_path / f"{name}_balanced.jsonl")
