import os

import hydra
import pandas as pd
from omegaconf import DictConfig
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

def clustering(data, n_clusters, random_state=42):
    """
    Clusters tabular data into a specified number of clusters and selects real data points as cluster centers.

    Parameters:
    - data (list of dict): The input data to cluster.
    - n_clusters (int): The number of clusters to form.
    - random_state (int): Random state for reproducibility. Default is 42.

    Returns:
    - tuple:
        - list of list of dict: A list where each element is a list of data points (dicts) belonging to a cluster.
        - list of dict: A list of dictionaries representing the closest real data points to the computed centroids.
    """
    # Convert list of dictionaries to Pandas DataFrame
    df = pd.DataFrame(data)

    # Identify numerical and categorical columns
    numerical_cols = df.select_dtypes(
        include=["int64", "float64"]).columns.tolist()
    categorical_cols = df.select_dtypes(
        include=["object", "category"]).columns.tolist()

    # Define the preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value=0)),  # Fill missing with 0
                ("scaler", StandardScaler()),
            ]), numerical_cols),
            ("cat", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),  # Fill missing with "missing"
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), categorical_cols),
        ]
    )

    # Create a pipeline for preprocessing and clustering
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("kmeans", KMeans(n_clusters=n_clusters, random_state=random_state)),
        ]
    )

    # Fit the pipeline to the data
    pipeline.fit(df)

    # Retrieve cluster labels and add to DataFrame
    df["cluster"] = pipeline.named_steps["kmeans"].labels_

    # Split data into clusters
    clustered_data = [df[df["cluster"] == cluster].drop(columns="cluster").to_dict(orient="records")
                      for cluster in range(n_clusters)]

    # Compute closest real data points to cluster centroids
    centers_preprocessed = pipeline.named_steps["kmeans"].cluster_centers_

    # Find the closest real data points to the centroids
    transformed_data = pipeline.named_steps["preprocessor"].transform(
        df.drop(columns="cluster"))
    closest_indices, _ = pairwise_distances_argmin_min(
        centers_preprocessed, transformed_data)

    # Retrieve the closest original data points as cluster centers
    cluster_centers = df.iloc[closest_indices].drop(
        columns="cluster").to_dict(orient="records")

    return clustered_data, cluster_centers



@hydra.main(config_path="./", config_name="synscore_anchors", version_base="1.3")
def main(cfg: DictConfig):
    """
    Main function to process datasets, perform clustering, and save results using Hydra for configuration.

    Parameters:
    - cfg (DictConfig): Hydra configuration object.
    """
    os.makedirs(cfg.anchor_dataset_dir, exist_ok=True)

    for dataset in cfg.datasets_config:
        dataset_name = next(iter(dataset))
        dataset_config = dataset[dataset_name]
        num_clusters = dataset_config.num_clusters

        # TODO: modify dataset as needed
        dataset_path = os.path.join(
            cfg.raw_dataset_dir, f"{dataset_name}/raw/{dataset_name}_train.csv")
        data = pd.read_csv(dataset_path)

        # Perform clustering
        _, cluster_centers = clustering(data, num_clusters)

        # Save cluster centers to a CSV file
        centers_file = os.path.join(
            cfg.anchor_dataset_dir, f"{dataset_name}_anchors.csv")
        pd.DataFrame(cluster_centers).to_csv(centers_file, index=False)

        print(f"Processed and saved results for {dataset_name}")


if __name__ == "__main__":

    main()
