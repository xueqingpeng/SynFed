import os
from abc import ABC
from collections import defaultdict
from typing import Any, Dict, List, Union

import argparse
import jsonlines
import numpy as np
import pandas as pd


from src.data import SCHEMA_MAP

# from sklearn.cluster import KMeans
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import OneHotEncoder, StandardScaler


class FedResampler(ABC):
    def __init__(
        self,
        data: pd.DataFrame,
        schema: Dict[str, Any],
        num_clients: int,
        label_key: str,
    ) -> None:
        self.num_clients: int = num_clients
        self.data_schema: Dict[str, Any] = schema
        self.label_key: str = label_key
        self.data: pd.DataFrame = data
        self._preprocess_data()
        self.labels: pd.Series = self._get_labels()

    def get_client_datas(
        self, method: str = "dirichlet", is_pair_string=True, alpha: float = 0.5
    ) -> Dict[int, List[Dict[str, Any]]]:
        if method == "dirichlet":
            datas = self._dirichlet_non_iid_data(
                self.data, self.labels, self.num_clients, alpha=alpha)
        elif method == "none":
            datas = {i: self.data.to_dict(orient="records")
                     for i in range(self.num_clients)}
        else:
            raise ValueError(f"Invalid method: {method}")

        if is_pair_string:
            for k, v in datas.items():
                datas[k] = self._map_dicts_to_prompt_pair(v)

        return datas

    def _get_labels(self) -> pd.Series:
        if self.data_schema.target_type == "regression":
            labels = self._regression_as_categorical(self.data, self.label_key)
        else:
            labels = self.data[self.label_key]
        return labels

    def _preprocess_data(self) -> None:
        # Drop rows with missing data
        # self.data = self.data.dropna(axis=0, how="any").reset_index(drop=True)

        # Convert data types using the schema directly
        # self.data = self.data.fillna(0)
        self.data = self.data.astype(self.data_schema.dtype_dict)

    def _map_dicts_to_prompt_pair(self, ds) -> None:

        def _to_pair(d):
            label = self.label_key
            label_val = d[label]
            del d[label]

            schema = self.data_schema
            qas = schema.llm_q_prompt.format(schema.dict_to_str(d))
            ans = schema.llm_a_prompt.format(str(schema.target_map(label_val)))
            return {"input": qas, "output": ans}

        return list(map(_to_pair, ds))

    @staticmethod
    def _regression_as_categorical(data: pd.DataFrame, label_key: str, bins: int = 4) -> pd.Series:
        return pd.cut(data[label_key], bins=bins, labels=range(bins))

    @staticmethod
    def _dirichlet_non_iid_data(
        data: pd.DataFrame, labels: pd.Series, num_clients: int, alpha: float = 0.5
    ) -> Dict[int, List[Dict[str, Any]]]:
        # Get the unique classes and their indices
        num_classes = len(set(labels))
        class_data_indices = {i: np.where(
            labels == i)[0] for i in range(num_classes)}

        # Initialize the distribution of data for each client
        client_data_indices = defaultdict(list)

        # Distribute data according to the Dirichlet distribution
        for c, indices in class_data_indices.items():
            # Generate proportions using Dirichlet distribution
            proportions = np.random.dirichlet([alpha] * num_clients)

            # Ensure at least one sample per client
            min_samples_per_client = max(1, len(indices) // num_clients)
            indices = np.random.permutation(indices)
            split_indices = np.array_split(
                indices, (proportions * len(indices)).astype(int))

            # Adjust if some clients have no data
            for i in range(len(split_indices)):
                if len(split_indices[i]) < min_samples_per_client:
                    deficit = min_samples_per_client - len(split_indices[i])
                    # Borrow data from other clients if possible
                    for j in range(len(split_indices)):
                        if i != j and len(split_indices[j]) > min_samples_per_client:
                            transfer_indices = split_indices[j][:deficit]
                            split_indices[j] = split_indices[j][deficit:]
                            split_indices[i] = np.concatenate(
                                [split_indices[i], transfer_indices])
                            break

            for client_id, client_indices in enumerate(split_indices):
                client_data_indices[client_id].extend(client_indices)

        # Construct the dataset for each client
        client_datasets = {client_id: data.iloc[indices]
                           for client_id, indices in client_data_indices.items()}

        for k, v in client_datasets.items():
            if len(v) > 0:
                client_datasets[k] = v.to_dict(orient="records")
        return client_datasets


import os
import pandas as pd
import jsonlines
import argparse

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Generate federated datasets.")
    parser.add_argument("--ds_name", type=str, required=True, help="Name of the dataset.")
    parser.add_argument("--num_client", type=int, required=True, help="Number of clients.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input file.")
    parser.add_argument("--output_dir", type=str, required=True, help="File path to save federated dataset files.")
    parser.add_argument("--split_type", type=str, choices=["train", "test", "val"], required=True, help="Specify 'train' or 'test' for the dataset split.")
    parser.add_argument("--split_method", type=str, default="dirichlet", help="Method to split data.")

    # Parse the arguments
    args = parser.parse_args()

    # Assign arguments to variables
    ds_name = args.ds_name
    num_clients = args.num_client
    input_file = args.input_file
    output_dir = args.output_dir
    split_type = args.split_type
    split_method = args.split_method

    # Load the dataset
    print(f"Loading dataset: {ds_name} ({split_type})")
    data = pd.read_csv(input_file)

    # Get the schema for the dataset
    schema = SCHEMA_MAP.get(ds_name)
    if not schema:
        raise ValueError(f"Schema not found for dataset: {ds_name}")

    # Initialize the resampler
    print(f"Initializing resampler with {num_clients} clients using '{split_method}' method.")
    resampler = FedResampler(data, schema, num_clients, schema.target)

    # Get the client data
    client_datas = resampler.get_client_datas(method=split_method)

    # Save all client data into JSONL files, with split type in file name
    print(f"Saving federated datasets to dir: {output_dir}")
    for client_id, client_data in client_datas.items():
        file_name = os.path.join(output_dir, f"{ds_name}_client_{client_id}.{split_type}.jsonl")
        with jsonlines.open(file_name, mode="w") as writer:
            writer.write_all(client_data)

    print(f"Federated dataset generation completed for {split_type} split. Output saved to dir: {output_dir}.")

if __name__ == "__main__":
    main()
