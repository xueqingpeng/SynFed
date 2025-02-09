import itertools
import os
import random
import uuid

import jsonlines
import matplotlib.pyplot as plt
import numpy as np
import torch


def rolling_visit(data, window_size=5):
    for i in range(len(data) - window_size + 1):
        yield data[i:i+window_size]


def mask_safe_json_serializable(d):
    """
    Convert dictionary to a safe JSON serializable format by handling non-serializable types.
    Excludes handling for datetime and timestamp types.

    Args:
    - d (dict): Input dictionary to be serialized.

    Returns:
    - dict: Dictionary with non-serializable types replaced or converted.
    """
    def safe_serialize(value):
        """
        Attempt to convert a value to a serializable form.
        Handles specific non-serializable types (e.g., numpy, set).
        """
        if isinstance(value, np.ndarray):
            # Convert numpy arrays to lists
            return value.tolist()
        elif isinstance(value, np.generic):
            # Convert numpy scalar types to native Python types (e.g., np.int64 -> int)
            return value.item()
        elif isinstance(value, (set, frozenset)):
            # Convert sets to lists (since JSON doesn't support sets)
            return list(value)
        elif isinstance(value, dict):
            # Recursively serialize the dictionary
            return {key: safe_serialize(val) for key, val in value.items()}
        else:
            # Return the value as-is if it's serializable
            return value

    # Apply the safe serialization function to all elements in the dictionary
    return {key: safe_serialize(value) for key, value in d.items()}


def write_jsonlines(data, file_path):
    with jsonlines.open(file_path, mode='w') as writer:
        writer.write_all(data)


def read_jsonlines(file_path):
    with jsonlines.open(file_path, mode='r') as reader:
        return list(reader)


def generate_uuid():
    """
    Generates a random UUID (Universal Unique Identifier).

    Returns:
    - str: A random UUID.
    """
    return str(uuid.uuid4())


def show_dict(metrics):
    from prettytable import PrettyTable
    
    # Create a PrettyTable object
    table = PrettyTable()

    # Define the table columns
    table.field_names = ["Metric", "Value"]

    # Add rows from the metrics dictionary
    for metric, value in metrics.items():
        # Format the value to have 3 digits
        formatted_value = f"{value:.3f}" if isinstance(value, (int, float)) else value
        table.add_row([metric, formatted_value])

    # Print the table with aligned columns
    print(table)


def set_seed_all(seed: int):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        from transformers import set_seed
        set_seed(seed)
    except ImportError:
        print("Transformers library not installed. Skipping Transformers seed setting.")
    print(f"All seeds set to {seed}.")


def plot_histograms_comparison(source_df, synthetic_df, columns=None, bins=60):
    import seaborn as sns
    
    """
    Plots overlapping histograms for numerical columns comparing source and synthetic datasets.

    Parameters:
    - source_df (pd.DataFrame): The original source dataset.
    - synthetic_df (pd.DataFrame): The synthetic dataset to compare against the source.
    - columns (list, optional): Specific columns to plot. Defaults to all numerical columns.
    - bins (int, optional): Number of bins for the histograms. Defaults to 60.
    """
    if columns is None:
        columns = source_df.select_dtypes(include=[np.number]).columns.tolist()

    num_columns = len(columns)
    cols_per_row = 2
    rows = num_columns // cols_per_row + int(num_columns % cols_per_row > 0)

    plt.figure(figsize=(cols_per_row * 6, rows * 4))

    for idx, column in enumerate(columns, 1):
        plt.subplot(rows, cols_per_row, idx)
        sns.histplot(source_df[column].dropna(
        ), color='blue', label='Source', kde=True, bins=bins, alpha=0.6, stat='density')
        sns.histplot(synthetic_df[column].dropna(
        ), color='red', label='Synthetic', kde=True, bins=bins, alpha=0.6, stat='density')
        plt.title(f'Histogram of {column}')
        plt.xlabel(column)
        plt.ylabel('Probability')
        plt.legend()

    plt.tight_layout()
    plt.show()


def plot_merged_correlation_scatter(source_df, synthetic_df, title='Merged Correlation Scatter Plots', columns=None, cols_per_row=3):
    """
    Plots scatter plots for all pairwise numerical column combinations comparing source and synthetic datasets.

    Parameters:
    - source_df (pd.DataFrame): The original source dataset.
    - synthetic_df (pd.DataFrame): The synthetic dataset to compare against the source.
    - title (str, optional): The title of the entire plot. Defaults to 'Merged Correlation Scatter Plots'.
    - columns (list, optional): Specific columns to plot. Defaults to all numerical columns.
    - cols_per_row (int, optional): Number of scatter plots per row. Defaults to 3.
    """
    import seaborn as sns
    
    if columns is None:
        columns = source_df.select_dtypes(include=[np.number]).columns.tolist()

    pairs = list(itertools.combinations(columns, 2))
    num_pairs = len(pairs)

    rows = num_pairs // cols_per_row + int(num_pairs % cols_per_row > 0)

    plt.figure(figsize=(cols_per_row * 5, rows * 4))
    plt.suptitle(title, fontsize=16)

    for idx, (x, y) in enumerate(pairs, 1):
        plt.subplot(rows, cols_per_row, idx)

        sns.scatterplot(
            data=synthetic_df, x=x, y=y,
            color='red', label='Synthetic',
            alpha=0.5, edgecolor=None
        )

        sns.scatterplot(
            data=source_df, x=x, y=y,
            color='blue', label='Source',
            alpha=0.5, edgecolor=None
        )

        plt.title(f'{x} vs {y}')
        plt.xlabel(x)
        plt.ylabel(y)
        plt.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
