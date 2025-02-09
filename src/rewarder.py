
import copy
import multiprocessing
import os
import warnings
from typing import Any, Callable, Dict, List

import re
import pandas as pd
from omegaconf import DictConfig
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from src.client import VllmClient
from src.data import SCHEMA_MAP, BaseDataSchema
from src.logger import get_logger

logger = get_logger(__name__)

PREFIX = {
    # "system":"Few-shot Examples:\n\n",
    # "example": "INPUT: {} \nOUTPUT: {} \n\n",
    # "request": "INPUT: {} \nOUTPUT: {}",
    # "instruction": "Please provide the OUTPUT for the following INPUT.\n",

    "system": "",
    "example": "{}{} \n\n",
    "request": "{}{}",
    "instruction": "",

    # "system": "Few-shot Examples:\n\n",
    # "example": "Example No.{}: \n<INPUT>: {}\n<OUTPUT>: {} \n\n",
    # "request": "<INPUT>: {}{}",
    # "instruction": "Directly generate the final <OUTPUT> from the <INPUT> below: \n\n",
}


class UnsimilarDataFinder:
    def __init__(self, schema, data, n_cluster):
        self.schema = schema
        self.data = pd.DataFrame(data)
        self.n_clusters = n_cluster

        self.target = schema.target

        self.pipeline = None

        self._clustering_init()

    def _clustering_init(self):
        df = self.data
        n_clusters = self.n_clusters

        # Identify numerical and categorical columns
        numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        # Handle missing values
        if numerical_cols:
            df[numerical_cols] = df[numerical_cols].fillna(0)  # Fill NaN in numerical columns with 0
        if categorical_cols:
            df[categorical_cols] = df[categorical_cols].fillna("missing")  # Fill NaN in categorical columns with "missing"

        # Define the preprocessing steps
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", Pipeline([
                    ("imputer", SimpleImputer(strategy="constant", fill_value=0)),  # Ensure numerical NaNs are filled
                    ("scaler", StandardScaler())
                ]), numerical_cols),
                ("cat", Pipeline([
                    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),  # Ensure categorical NaNs are filled
                    ("onehot", OneHotEncoder(handle_unknown="ignore"))
                ]), categorical_cols),
            ],
            remainder="drop"  # Drop any columns not explicitly listed in transformers
        )

        # Create a pipeline for preprocessing and clustering
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("kmeans", KMeans(n_clusters=n_clusters, random_state=42))
            ]
        )

        # Fit the pipeline to the data
        pipeline.fit(df)

        # Add cluster labels to the DataFrame
        df["cluster"] = pipeline.named_steps["kmeans"].labels_

        # Store the pipeline for future use
        self.pipeline = pipeline
        self.data = df  # Save the updated DataFrame with cluster labels

    def get_not_sames(self, row):
        # Ensure row is a valid dict
        if not isinstance(row, dict):
            raise ValueError("Input must be a dictionary.")

        # Predict the cluster for the given row
        predicted_cluster = self.pipeline.predict(pd.DataFrame([row]))[0]

        # Initialize a list to store results
        results = []

        # Iterate over groups in the data grouped by "cluster"
        for cluster_value, group in self.data.groupby("cluster"):
            if cluster_value != predicted_cluster:
                # Filter the group where the target column differs
                diff_group = group[group[self.target] != row[self.target]]
                if not diff_group.empty:
                    # Sample a row from the filtered group, drop 'cluster' column, and store it as a dict
                    sampled_record = diff_group.sample(1).drop(columns="cluster").to_dict(orient="records")[0]
                    results.append(sampled_record)

        return results


class AddonePromptAdder:
    def __init__(self,
                 schema: BaseDataSchema,
                 anchor_data: List[Dict[str, Any]],
                 unsim_anchor_data: List[List[Dict[str, Any]]],
                 *args, **kwargs):
        self.schema = schema
        self.anchor_data = anchor_data
        self.unsim_anchor_data = unsim_anchor_data

        # Extract schema-related values to avoid long dot usage
        self.llm_q_prompt = self.schema.llm_q_prompt
        self.target = schema.target
        self.target_map = schema.target_map
        self.dict_to_str = schema.dict_to_str

        self.qas_zeros = []
        self.qas_ones = []
        self.ans_grounds = []

        self._build_zero_prompt()

    def _gen_prefix_prompt(self, datapoints):
        if isinstance(datapoints, dict):
            datapoints = [datapoints]
        prefix = PREFIX["system"]
        for i, d in enumerate(datapoints):
            input_str = self.llm_q_prompt.format(self.dict_to_str({k: v for k, v in d.items() if k != self.target}))
            output_str = str(self.target_map(d[self.target]))
            # prefix += PREFIX["example"].format(i, input_str, output_str)
            prefix += PREFIX["example"].format(input_str, output_str)
        prefix += PREFIX["instruction"]

        return prefix

    def _build_zero_prompt(self,):
        for d, unsim in zip(self.anchor_data, self.unsim_anchor_data):
            d_without_label = {k: v for k, v in d.items() if k != self.target}
            input_str = self.llm_q_prompt.format(self.dict_to_str(d_without_label))
            output_str = str(self.target_map(d[self.target]))
            qas = self._gen_prefix_prompt(unsim) + PREFIX["request"].format(input_str, "")
            self.qas_zeros.append(qas)
            self.ans_grounds.append(output_str)

    def _gen_one_prompt(self, datapoint):
        if not isinstance(datapoint, dict):
            raise ValueError("Input must be a records of row.")
        qas_ones = []
        for d, unsim in zip(self.anchor_data, self.unsim_anchor_data):
            d_without_label = {k: v for k, v in d.items() if k != self.target}
            input_str = self.llm_q_prompt.format(self.dict_to_str(d_without_label))
            qas = self._gen_prefix_prompt([datapoint] + unsim) + PREFIX["request"].format(input_str, "")
            qas_ones.append(qas)
        return qas_ones

    def __call__(self, datapoint):
        qas_ones = self._gen_one_prompt(datapoint)
        ds0 = []
        ds1 = []
        for q0, q1, a in zip(self.qas_zeros, qas_ones, self.ans_grounds):
            ds0.append({"question": q0, "gold": a})
            ds1.append({"question": q1, "gold": a})

        return {"reward_ds0": ds0, "reward_ds1": ds1}


class MutiProcessInference:
    def __init__(self, config):
        self.gpu_groups = config.gpu_gropus
        self.config = config
        self.logger = logger

    @staticmethod
    def process_shard(gpu_ids, shard, config):
        """Process a single shard of data on specified GPU."""
        try:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids)

            prompts = [d['question'] for d in shard]

            with VllmClient(config) as client:
                results = client.generate(prompts=prompts)

            for d, r in zip(shard, results):
                d['response'] = r

            return shard

        except Exception as e:
            logger.error(f"Error processing shard on GPU {gpu_ids}: {str(e)}")
            raise

    @staticmethod
    def data_split(data, num_shards):
        shard_size = len(data) // num_shards
        shards = []
        for i in range(num_shards):
            st = i * shard_size
            ed = min(len(data), (i + 1) * shard_size)
            shards.append(data[st:ed])
        return shards

    def run_inference(self, data):
        """Run parallel inference across multiple GPUs."""
        if not data:
            self.logger.warning("Empty input data")
            return []

        num_shards = len(self.gpu_groups)
        shards = self.data_split(data, num_shards)

        try:
            with multiprocessing.Pool(num_shards) as pool:
                result_shards = pool.starmap(
                    self.process_shard,
                    [(gpu_ids, shard, self.config)
                     for gpu_ids, shard in zip(self.gpu_groups, shards)]
                )

            # Combine results maintaining order
            all_results = []
            for shard_results in result_shards:
                all_results.extend(shard_results)

            self.logger.info(f"Successfully processed {len(all_results)} items")
            return all_results

        except Exception as e:
            self.logger.error(f"Inference failed: {str(e)}")
            raise


class DataManager:
    def __init__(self, raw_data=None):
        self.shared_reward_ds0: List[Any] = []
        self.nshared_reward_ds1s: List[List[Any]] = []
        self.flat_data: List[Any] = []
        self.block_size: int = 0
        self.raw_data: List[Dict[str, Any]] = []
        self.total_length: int = 0

        if raw_data is not None:
            self.load_raw_data(raw_data)

    def load_raw_data(self, raw_data: List[Dict[str, Any]]) -> None:
        """
        Load and initialize data from raw_data.

        Args:
            raw_data (List[Dict[str, Any]]): List of dictionaries containing 'reward_ds0' and 'reward_ds1'.
        """
        # Deep copy to prevent modifications to the original raw_data
        self.raw_data = copy.deepcopy(raw_data)
        if not self.raw_data:
            raise ValueError("raw_data is empty.")

        # Assume 'reward_ds0' is shared across all items and taken from the first item
        self.shared_reward_ds0 = self.raw_data[0].get('reward_ds0', [])
        self.nshared_reward_ds1s = [item.get('reward_ds1', []) for item in self.raw_data]

        # Validate that all 'reward_ds1' have the same length as 'reward_ds0'
        if not all(len(ds1) == len(self.shared_reward_ds0) for ds1 in self.nshared_reward_ds1s):
            raise ValueError("All 'reward_ds1' must have the same length as 'reward_ds0'.")

        self.block_size = len(self.shared_reward_ds0)

        # Concatenate all 'reward_ds1' lists followed by 'reward_ds0'
        self.flat_data = []
        for ds1 in self.nshared_reward_ds1s:
            self.flat_data.extend(ds1)
        self.flat_data.extend(self.shared_reward_ds0)

        self.total_length = len(self.flat_data)

    def get_flat_data(self) -> List[Any]:
        """
        Retrieve the concatenated flat data.

        Returns:
            List[Any]: The concatenated list of all 'reward_ds1' and 'reward_ds0'.
        """
        return self.flat_data

    def update_flat_data(self, new_data: List[Any]) -> None:
        """
        Update the flat data with new_data.

        Args:
            new_data (List[Any]): The new flat data to set.
        """
        if len(new_data) != self.total_length:
            raise ValueError(
                f"New data length mismatch: expected {self.total_length} elements, got {len(new_data)}."
            )
        self.flat_data = new_data

    def apply_changes(self) -> List[Dict[str, Any]]:
        """
        Write the modified flat_data back to raw_data.

        This method splits self.flat_data into segments corresponding to each 'reward_ds1' and 'reward_ds0'
        and updates the raw_data accordingly.

        Returns:
            List[Dict[str, Any]]: The updated raw_data with modified 'reward_ds1' and 'reward_ds0'.
        """
        if not self.raw_data or not self.flat_data:
            raise ValueError("DataManager has not been initialized with raw_data.")

        expected_length = len(self.raw_data) * self.block_size + self.block_size

        if len(self.flat_data) != expected_length:
            raise ValueError(
                f"Data length mismatch: expected {expected_length} elements, got {len(self.flat_data)}."
            )

        # Update 'reward_ds1' for each item
        for i, item in enumerate(self.raw_data):
            start_idx = i * self.block_size
            end_idx = (i + 1) * self.block_size
            item['reward_ds1'] = self.flat_data[start_idx:end_idx]

        # Extract the 'reward_ds0' from the end of self.flat_data
        updated_reward_ds0 = self.flat_data[-self.block_size:]

        # Update 'reward_ds0' for each item (assuming 'reward_ds0' is shared)
        for item in self.raw_data:
            # Use a copy to prevent unintended references if mutable
            item['reward_ds0'] = updated_reward_ds0.copy()

        return self.raw_data


class AnswerExtractor:

    def __init__(self, config: DictConfig, schema):
        super().__init__(config)
        self.schema = schema

    def text_to_real_value(self, input: str) -> Any:

        # extract answer with pattern
        pat1 = r"output:(.*?)\s"
        compiled_pat1 = re.compile(pat1, re.IGNORECASE, re.DOTALL)
        match1 = compiled_pat1.search(input)

        text = input.strip().split("\n")[0].strip(" ':\"")
        target_type = self.schema.target_type
        mapping = self.schema.inverse_target_map

        if target_type == "classification":
            if isinstance(mapping, dict):
                for key in mapping.keys():
                    if re.search(re.escape(key), text, re.IGNORECASE):
                        return mapping.get(key)
                warnings.warn(f"Unrecognized classification result in: {text}")
            return None

        elif target_type == "regression":
            if isinstance(mapping, Callable):
                match = re.search(r'-?\d+\.?\d*', text)
                if match:
                    return float(match.group()) if '.' in match.group() else int(match.group())
                warnings.warn(f"No numerical value found in regression result: {text}")
            return None

        else:
            raise ValueError(f"Unknown target type: {target_type}")


def test_addprompt():

    class MockSchema:
        def __init__(self):
            self.target = "target"
            self.target_map = lambda x: x.upper()

    real_schema = SCHEMA_MAP['buddy']

    schema = MockSchema()
    schema.llm_q_prompt = real_schema.llm_q_prompt
    schema.dict_to_str = real_schema.dict_to_str

    anchor_data = [
        {"field1": "value1", "field2": "value2", "target": "A"},
        {"field1": "value3", "field2": "value4", "target": "A"}
    ]

    unsim_anchor_data = [
        [
            {"field1": "unsim_value1", "field2": "unsim_value2", "target": "B"},
            {"field1": "unsim_value3", "field2": "unsim_value4", "target": "C"}
        ],
        [
            {"field1": "unsim_value5", "field2": "unsim_value6", "target": "B"},
            {"field1": "unsim_value7", "field2": "unsim_value8", "target": "A"}
        ]
    ]
    datapoint = {"field1": "test_value1", "field2": "test_value2", "target": "C"}
    
    adder = AddonePromptAdder(schema=schema,
                              anchor_data=anchor_data,
                              unsim_anchor_data=unsim_anchor_data)

    result = adder(datapoint)
   

    
    print("_____[ds0]_____\n")
    for k, v in result['reward_ds0'][0].items():
        print(k)
        print("-"*10)
        print(v)
        print("-"*5 +"end"+ "-"*5 + "\n\n")
        
    print("_____[ds1]_____\n")
    for k, v in result['reward_ds1'][0].items():
        print(k)
        print("-"*10)
        print(v)
        print("-"*5 +"end"+ "-"*5 + "\n\n")
        
if __name__ == "__main__":
    test_addprompt()
    
    