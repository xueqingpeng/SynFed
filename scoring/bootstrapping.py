#!/usr/bin/env python3
"""
Bootstrap sampling script for Hugging Face datasets (train split only)
"""

import argparse
import random
import numpy as np
from tqdm import tqdm
from datasets import load_dataset, Dataset
from huggingface_hub import create_repo


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)


def load_dataset_train(dataset_name: str, subset: str = None):
    """Load train split of a Hugging Face dataset"""
    dataset = load_dataset(dataset_name, subset, split="train") if subset else load_dataset(dataset_name, split="train")
    print(f"Loaded {dataset_name}: {len(dataset)} samples")
    return dataset


def bootstrap_sampling(dataset, n_samples: int, sample_size: int = None, seed: int = 42):
    """Perform n rounds of bootstrap sampling with replacement"""
    sample_size = sample_size or len(dataset)
    samples = []
    
    for i in tqdm(range(n_samples), desc=f"Generating {n_samples} samples"):
        np.random.seed(seed + i)
        indices = np.random.choice(len(dataset), size=sample_size, replace=True)
        samples.append([dataset[int(idx)] for idx in indices])
    
    return samples


def create_hf_dataset(sample_idx, sample):
    """Convert bootstrap sample to HF Dataset"""
    data = []
    for item in sample:
        item_copy = item.copy()
        item_copy['bootstrap_id'] = sample_idx
        data.append(item_copy)
    return Dataset.from_list(data)


def upload_dataset(dataset, repo_name: str, private: bool = True, token: str = None):
    """Upload dataset to HF Hub"""
    create_repo(repo_id=repo_name, repo_type="dataset", private=private, token=token, exist_ok=True)
    dataset.push_to_hub(repo_id=repo_name, private=private, token=token)
    print(f"✓ {repo_name}")


def main():
    parser = argparse.ArgumentParser(description="Bootstrap sampling for HF datasets.")
    parser.add_argument("--dataset", required=True, help="HF Dataset name (user/repo)")
    parser.add_argument("--subset", help="Dataset subset")
    parser.add_argument("--n_samples", type=int, default=10, help="Number of samples")
    parser.add_argument("--sample_size", type=int, help="Size per sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--token", help="HF token")
    parser.add_argument("--private", action="store_true", help="Private repo")
    
    args = parser.parse_args()
    set_seed(args.seed)
    
    # Load dataset and bootstrap sample
    dataset = load_dataset_train(args.dataset, args.subset)
    samples = bootstrap_sampling(dataset, args.n_samples, args.sample_size, args.seed)
    
    # Upload each sample to separate repos
    for sample_idx, sample in enumerate(tqdm(samples, desc="Uploading samples")):
        repo_name = f"{args.dataset}_{sample_idx}"
        sample_dataset = create_hf_dataset(sample_idx, sample)
        upload_dataset(sample_dataset, repo_name, args.private, args.token)
    
    print("Done!")
    return samples


if __name__ == "__main__":
    main()
