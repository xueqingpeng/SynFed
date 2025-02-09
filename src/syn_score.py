import os

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from datasets import load_from_disk
from src.client import SynClient, TextToDictStatus, ExtractValueClient
from src.data import SCHEMA_MAP, BaseDataSchema
from src.logger import get_logger
from src.rewarder import DownstreamImproveReward
from src.utils import (generate_uuid, mask_safe_json_serializable,
                       write_jsonlines)
from src.metrics import get_scorer

logger = get_logger(__name__)


class UnsimilarDataFinder:
    def __init__(self, schema, data, n_cluster):
        self.schema = schema
        self.data = pd.DataFrame(data)
        self.n_clusters = n_cluster

        self.target = schema.target.upper()

        self.pipeline = None

        self._clustering_init()

    def _clustering_init(self):
        df = self.data
        n_cluster = self.n_clusters

        # Identify numerical and categorical columns
        numerical_cols = df.select_dtypes(
            include=["int64", "float64"]).columns.tolist()
        categorical_cols = df.select_dtypes(
            include=["object", "category"]).columns.tolist()

        # Define the preprocessing steps
        preprocessor = ColumnTransformer(
            transformers=[("num", StandardScaler(), numerical_cols),
                          ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),]
        )

        # Create a pipeline for preprocessing and clustering
        pipeline = Pipeline(
            steps=[("preprocessor", preprocessor),
                   ("kmeans", KMeans(n_clusters=self.n_clusters, random_state=42)),]
        )

        # Fit the pipeline to the data
        pipeline.fit(df)

        df["cluster"] = pipeline.named_steps["kmeans"].labels_

        self.pipeline = pipeline

    def sample_not_sames(self, row):
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


class SyntheticDataProcessor:
    def __init__(self, syn_config, downstream_config, schema, prompt_data, train_data, anchor_data):
        self.syn_config = syn_config
        self.downstream_config = downstream_config

        self.schema = schema

        self.prompt_data = prompt_data
        self.train_data = train_data
        self.anchor_data = anchor_data

        self.data_finder = UnsimilarDataFinder(self.schema, self.train_data, 3)

    def generate_synthetic_data(self, client):
        """Generates synthetic data for the prompt dataset."""
        results = []
        for i, data in enumerate(self.prompt_data):
            prompt = data['question']
            prompt_id = generate_uuid()

            syn_datas = client.generate([prompt] * 10)
            for syn_data in syn_datas:
                syn_data['prompt_id'] = prompt_id
                syn_data['syn_id'] = generate_uuid()

            results.extend(syn_datas)
        return results

    def add_status_scores(self, results):
        """Adds status scores to the results."""
        for idx, data in enumerate(results):
            data['score'] = data['status'].score()
        return results

    def add_addon_scores(self, results, addon_scorer):
        """Adds addon scores to the results using a downstream scoring system."""
        for idx, data in enumerate(results):
            if data['status'] is TextToDictStatus.SUCCESS:
                dpoint = data['result']
                not_same_dpoints = self.data_finder.sample_not_sames(dpoint)
                prefix_dpoints = [dpoint] + not_same_dpoints
                score = addon_scorer.get_score(prefix_dpoints, is_show_on_step=True)
                data['score'] += score
        return results

    def process(self):
        """Main method to process the dataset, generate synthetic data, and apply scoring."""

        # step1 Generate synthetic data
        logger.info("+"*30 + "  Generating synthetic data...")
        with SynClient(self.syn_config, self.schema) as client:
            results = self.generate_synthetic_data(client)

        # step2. Add scores
        # Add status scores
        logger.info("+"*30 + "  Adding status scores...")
        results = self.add_status_scores(results)
        with ExtractValueClient(self.downstream_config, self.schema) as client:

            addon_scorer = DownstreamImproveReward(schema=self.schema,
                                                   client=client,
                                                   scorer=get_scorer(self.schema.target_type),
                                                   anchor_data=self.anchor_data)
            results = self.add_addon_scores(results, addon_scorer)

        return results


@hydra.main(config_path="./config", config_name="sample_config", version_base="1.3")
def main(cfg: DictConfig):

    for ds_name, ds_config in cfg.datasets.items():

        # if ds_name != "diabetes":
        #     continue

        logger.info("*"*50 + f"  Processing dataset: {ds_name}\n\n")
        schema = SCHEMA_MAP.get(ds_name)

        prompt_data = load_from_disk(ds_config.prompt_data)['test']
        anchor_data = schema.format_load_csv(ds_config.anchor_data)
        train_data = schema.format_load_csv(ds_config.train_data)

        syn_config = cfg.syn_config
        downstream_config = cfg.downstream_config
        downstream_config.vllm_params.model = ds_config.vllm_model

        processor = SyntheticDataProcessor(syn_config,
                                           downstream_config,
                                           schema,
                                           prompt_data,
                                           train_data,
                                           anchor_data)
        results = processor.process()

        for result in results:
            result['status'] = str(result['status'])
            result['result'] = mask_safe_json_serializable(result['result'])

        print(results[0])
        write_jsonlines(results, f"{ds_name}_results.jsonl")

        break


if __name__ == "__main__":
    main()
