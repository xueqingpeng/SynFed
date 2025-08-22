import os
import pandas as pd
from bootstrapping import load_dataset_train, upload_dataset

def get_percentile_samples(scores, dataset, percentile=0.2, top=True):
    """Get top/bottom percentile samples from dataset based on scores."""
    sorted_scores = scores.sort_values("score", ascending=not top)
    n = int(len(scores) * percentile)
    indices = sorted_scores.head(n).index
    
    direction = "top" if top else "bottom"
    score_min, score_max = sorted_scores.iloc[0]['score'], sorted_scores.iloc[n-1]['score']
    print(f"* {direction.title()} {percentile*100:.0f}%: {n} samples ({score_min:.4f} - {score_max:.4f})")
    print(f"* indices: {indices[:10]}...")
    
    return dataset.select(indices)

def main():
    # Load data
    scores_file = "/gpfs/radev/home/xp83/Documents/project/scripts/SynFed/scoring/cleveland_perplexity_scores.jsonl"
    dataset_name = "TheFinAI/MED_SYN1_CLEVELAND_train"
    
    scores = pd.read_json(scores_file, lines=True)
    dataset = load_dataset_train(dataset_name)
    print(f"Loaded {len(scores)} scores, {len(dataset)} samples")
    
    # Get top/bottom 20% samples
    top_samples = get_percentile_samples(scores, dataset, 0.2, top=True)
    bottom_samples = get_percentile_samples(scores, dataset, 0.2, top=False)
    
    # Upload to HuggingFace
    token = os.getenv("HF_TOKEN")
    
    for samples, suffix in [(top_samples, "top20"), (bottom_samples, "bottom20")]:
        repo_name = f"{dataset_name}_{suffix}"
        upload_dataset(samples, repo_name, private=True, token=token)
        print(f"✓ Uploaded {len(samples)} samples to {repo_name}")
    
    return top_samples, bottom_samples

if __name__ == "__main__":
    main()
    