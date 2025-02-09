import os
from copy import deepcopy
from functools import partial

import json 

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.compose import ColumnTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from datasets import Dataset, DatasetDict
from src.data import SCHEMA_MAP
from src.logger import get_logger

logger = get_logger(__name__, f"{__name__}.log")


def __int_to_str(n: int) -> str:
    num_dict = {
        1: "one",
        2: "two",
        3: "three",
        4: "four",
        5: "five",
        6: "six",
        7: "seven",
        8: "eight",
        9: "nine",
        10: "ten",
        11: "eleven",
        12: "twelve",
        13: "thirteen",
        14: "fourteen",
        15: "fifteen",
        16: "sixteen",
        17: "seventeen",
        18: "eighteen",
        19: "nineteen",
        20: "twenty",
        21: "twenty-one",
        22: "twenty-two",
        23: "twenty-three",
        24: "twenty-four",
        25: "twenty-five",
        26: "twenty-six",
        27: "twenty-seven",
        28: "twenty-eight",
        29: "twenty-nine",
        30: "thirty",
    }
    return num_dict.get(n, str(n))  # Default to string if not in dict


def __shuffle_row(row):
    """Shuffle keys in a dictionary."""
    keys = list(row.keys())
    np.random.shuffle(keys)
    return {key: row[key] for key in keys}


def __sort_row(row):
    keys = list(row.keys())
    keys.sort()
    return {key: row[key] for key in keys}


def __similarity_score(data):
    """
    Calculate similarity scores between rows, handling both numerical and categorical variables.
    Handles missing values and ensures proper preprocessing.
    """
    # Identify numerical and categorical columns
    numerical_cols = data.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = data.select_dtypes(exclude=np.number).columns.tolist()

    # Handle missing values for numerical columns
    if numerical_cols:
        # Remove numerical columns with all NaN values
        valid_numerical_cols = [col for col in numerical_cols if not data[col].isna().all()]
        if valid_numerical_cols:
            imputer = SimpleImputer(strategy="mean")
            data[valid_numerical_cols] = imputer.fit_transform(data[valid_numerical_cols])
        else:
            print("No valid numerical columns to process.")

    # Handle missing values for categorical columns
    if categorical_cols:
        imputer = SimpleImputer(strategy="constant", fill_value="missing")
        data[categorical_cols] = imputer.fit_transform(data[categorical_cols])

    # Define transformers for numerical and categorical columns
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols if numerical_cols else []),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols if categorical_cols else []),
        ],
        remainder="drop"
    )

    # Fit and transform the data
    processed_data = preprocessor.fit_transform(data)

    # Handle potential NaN in processed data
    processed_data_filled = np.nan_to_num(processed_data, nan=0.0)

    # Compute similarity matrix
    similarity_matrix = cosine_similarity(processed_data_filled)

    # Convert similarity matrix to DataFrame for easier interpretation
    similarity_df = pd.DataFrame(similarity_matrix, index=data.index, columns=data.index)

    return similarity_df



def _search_sim_pairs(data, policy_fun):
    """For each row, generate a pair using policy_fun."""
    pairs = []
    sim_mat = __similarity_score(data)

    logger.debug(f"Similarity matrix:\n{sim_mat}")

    for idx, row in data.iterrows():
        similar_indices = policy_fun(idx, sim_mat)
        # Convert the similar indices to a list of dictionaries
        if similar_indices:
            similar_rows = [data.iloc[i].to_dict() for i in similar_indices]

            pairs.append(
                {
                    "ans": [row.to_dict()],
                    "qas": similar_rows,
                }
            )
    return pairs


def _pair_to_sample(pair, schema, is_shuffle=False):

    qas = pair["qas"]
    q_prompt = deepcopy(schema.syn_q_prompt)

    q_content = ""
    for i, q in enumerate(qas):
        if is_shuffle:
            q = __shuffle_row(q)
        q_content += f"sample {__int_to_str(i + 1)}:\n{schema.dict_to_jsonstr(q)}\n\n"

    q_prompt = q_prompt.format(q_content)

    ans = pair["ans"]
    a_prompt = deepcopy(schema.syn_a_prompt)

    a_content = ""
    for i, a in enumerate(ans):
        a = __sort_row(a)
        a_content += f"\n{schema.dict_to_jsonstr(a)}"

    a_prompt = a_prompt.format(a_content)

    sample = {}
    sample["question"] = q_prompt
    sample["response"] = a_prompt
    return sample


def preprocess_data(data):
    """Drop rows with missing values and reset the index."""
    # data = data.dropna(axis=0, how="any")  # Drop rows with any missing values
    data = data.reset_index(drop=True)  # Reset the index
    return data


def mixsim_policy(idx, sim_mat, n_sim, n_unsim):
    """Return an index list of most similar and most different rows."""

    sim_scores = sim_mat.iloc[idx]
    most_similar_indices = sim_scores.nlargest(n_sim + 1).index.tolist()[1:]
    if n_unsim != 0:
        most_dissimilar_indices = sim_scores.nsmallest(n_unsim).index.tolist()
    else:
        most_dissimilar_indices = []

    use_indices = most_similar_indices + most_dissimilar_indices
    np.random.shuffle(use_indices)

    return use_indices


def generate_data(dataset_name, dataset, policy, is_shuffle=False):
    """Generate samples from a dataset using a similarity policy."""
    pairs = _search_sim_pairs(dataset, policy)
    res = []
    schema = SCHEMA_MAP.get(dataset_name)
    for pair in pairs:
        sample = _pair_to_sample(pair, schema, is_shuffle)
        res.append(sample)
    return res


def show_sample(obj):
    for k, v in obj.items():
        print(f"{k}:\n\n{v}\n\n")
    print("*************")
    return obj


def convert_to_alpaca_format(dataset):
    """
    Converts a dataset with 'question' and 'response' keys into Alpaca SFT format.

    Args:
        dataset (list of dict): The input dataset, where each entry has 'question' and 'response' keys.

    Returns:
        list of dict: The dataset in Alpaca SFT format.
    """
    alpaca_dataset = []
    for entry in dataset:
        if 'question' in entry and 'response' in entry:
            alpaca_entry = {
                "instruction": entry['question'],  # Map 'question' to 'instruction'
                "input": "",                       # Leave 'input' empty
                "output": entry['response']        # Map 'response' to 'output'
            }
            alpaca_dataset.append(alpaca_entry)
        else:
            raise ValueError("Each entry in the dataset must have 'question' and 'response' keys.")
    return alpaca_dataset


@hydra.main(config_path="./", config_name="sft_generator", version_base="1.3")
def main(cfg: DictConfig):

    raw_dataset_dir = cfg.raw_dataset_dir
    sft_dataset_dir = cfg.sft_dataset_dir

    os.makedirs(sft_dataset_dir, exist_ok=True)

    source_datasets = cfg.source_datasets
    source_dict = {}

    logger.info(f"Loading datasets: {source_datasets}")
    for ds_name in source_datasets:
        data_path = os.path.join(
            raw_dataset_dir, ds_name, f"raw/{ds_name}_train.csv")
        logger.info(f"Reading data from {data_path}...")
        data = pd.read_csv(data_path)
        ds_schema = SCHEMA_MAP.get(ds_name)
        data = data[ds_schema.dtype_dict.keys()]
        logger.info(f"Original data shape: {data.shape}")
        data = preprocess_data(data)
        
        logger.info(f"Processed data shape: {data.shape}")
        source_dict[ds_name] = data

    # 1. Train data generation
    n_sim = cfg.n_sim
    n_unsim = cfg.n_unsim

    # Define the similar data and unsimilar data numbers
    train_policy = partial(mixsim_policy, n_sim=n_sim, n_unsim=n_unsim)

    train_set = {}
    logger.info(f"Generating Training Sets")
    for ds_name, ds in source_dict.items():
        if ds_name not in cfg.target_datasets:
            train_set[ds_name] = generate_data(
                ds_name, ds, train_policy, is_shuffle=True)

    # 2. Test data generation
    test_set = {}
    # Default to n_sim + n_unsim for test data, all data should similar
    test_policy = partial(mixsim_policy, n_sim=n_sim + n_unsim, n_unsim=0)
    logger.info(f"Generating Validation Sets")
    for ds_name, ds in source_dict.items():
        if ds_name in cfg.target_datasets:
            test_set[ds_name] = generate_data(
                ds_name, ds, test_policy, is_shuffle=False)

    sft_dataset = {"train": [], "test": []}
    for ds_name, ds in train_set.items():
        sft_dataset["train"].extend(ds)
    for ds_name, ds in test_set.items():
        sft_dataset["test"].extend(ds)

    # np.random.shuffle(sft_dataset["train"])
    # np.random.shuffle(sft_dataset["test"])

    # logger.info(show_sample(sft_dataset["train"][0]))

    sft_dataset["train"] = Dataset.from_list(sft_dataset["train"])
    sft_dataset["test"] = Dataset.from_list(sft_dataset["test"])
    sft_dataset = DatasetDict(sft_dataset)

    logger.info(f"Saving dataset to {sft_dataset_dir}")
    sft_dataset.save_to_disk(sft_dataset_dir)

    alpaca_train = convert_to_alpaca_format(sft_dataset["train"])
    alpaca_test = convert_to_alpaca_format(sft_dataset["test"])
    with open(os.path.join(sft_dataset_dir, "alpaca_train.json"), 'w') as f:
        json.dump(alpaca_train, f, indent=2)
    with open(os.path.join(sft_dataset_dir, "alpaca_test.json"), 'w') as f:
        json.dump(alpaca_test, f, indent=2) 
    
    

if __name__ == "__main__":
    main()
