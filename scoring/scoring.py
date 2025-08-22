import os
import pandas as pd
import requests
from io import StringIO
from scipy.stats import spearmanr


def get_df_from_url(url):
    headers = {"Authorization": f"Bearer {os.getenv('HF_TOKEN')}"}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    df = pd.read_json(StringIO(response.text), lines=True)
    return df

def calculate_scores(mccs, dfs):
    docs, scores = [], []
    for i in range(500):
        doc = dfs[0].iloc[i]['doc']
        docs.append(doc)

        perplexities = [df.iloc[i]['perplexity'] for df in dfs]
        score = spearmanr(perplexities, mccs).correlation
        scores.append(score)

    return pd.DataFrame({"doc": docs, "score": scores})

def save_scores(df_output, fp_output):
    df_output.to_json(fp_output, orient='records', lines=True)
    print(f"* {len(df_output)} scores saved to {fp_output}!")


def main():
    # MCC values and URLs - easy to modify
    mccs = [0.3261, 0.6667, 0.6722, 0.5424, 0.5446]
    urls = [
        "https://huggingface.co/datasets/TheFinAI/lm-eval-results-fl-0shot-private/resolve/main/TheFinAI__fl-cleveland-sft-1-0-adapter/samples_Cleveland_perplexity_2025-08-20T23-52-01.586431.jsonl",
        "https://huggingface.co/datasets/TheFinAI/lm-eval-results-fl-0shot-private/resolve/main/TheFinAI__fl-cleveland-sft-1-1-adapter/samples_Cleveland_perplexity_2025-08-21T00-51-46.391925.jsonl",
        "https://huggingface.co/datasets/TheFinAI/lm-eval-results-fl-0shot-private/resolve/main/TheFinAI__fl-cleveland-sft-1-2-adapter/samples_Cleveland_perplexity_2025-08-21T00-07-59.149868.jsonl",
        "https://huggingface.co/datasets/TheFinAI/lm-eval-results-fl-0shot-private/resolve/main/TheFinAI__fl-cleveland-sft-1-3-adapter/samples_Cleveland_perplexity_2025-08-21T00-20-49.121285.jsonl",
        "https://huggingface.co/datasets/TheFinAI/lm-eval-results-fl-0shot-private/resolve/main/TheFinAI__fl-cleveland-sft-1-4-adapter/samples_Cleveland_perplexity_2025-08-21T00-33-37.840771.jsonl"
    ]
    
    # Load data
    dfs = [get_df_from_url(url) for url in urls]
    print(f"* loaded {[len(df) for df in dfs]} rows!")
    
    # Calculate scores
    df_scores = calculate_scores(mccs, dfs)
    print(f"* calculated {len(df_scores)} scores!")

    # Save scores
    fp_scores = "cleveland_perplexity_scores.jsonl"
    save_scores(df_scores, fp_scores)

if __name__ == "__main__":
    main()
