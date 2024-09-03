# dataprep/augmentation.py

"""
augmentation.py

This module provides functions for data augmentation and synthetic data generation,
primarily for tabular datasets. It includes techniques to handle imbalanced datasets
and create new samples that improve the robustness of machine learning models.

Functions:
    augment_data: Augments data using oversampling or undersampling techniques.
    generate_synthetic_data: Generates synthetic data using methods like SMOTE.
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import resample


def augment_data(
    data: pd.DataFrame,
    target_column: str,
    method: str = 'smote',
    imbalance_threshold: float = 0.2,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Augments the dataset to handle class imbalance using the specified method.

    Parameters:
        data (pd.DataFrame): The input dataframe containing the features and target column.
        target_column (str): The name of the target column in the dataframe.
        method (str): The augmentation method to use. Options are 'smote', 'oversample', or 'undersample'.
        imbalance_threshold (float): The threshold for class imbalance (0 to 1). Default is 0.2.
        random_state (int): Random state for reproducibility. Default is 42.

    Returns:
        pd.DataFrame: The augmented dataframe with balanced classes.
    """
    if method not in ['smote', 'oversample', 'undersample']:
        raise ValueError("Method must be one of 'smote', 'oversample', or 'undersample'.")

    # Separate majority and minority classes
    class_counts = data[target_column].value_counts()
    majority_class = class_counts.idxmax()
    minority_class = class_counts.idxmin()
    majority_data = data[data[target_column] == majority_class]
    minority_data = data[data[target_column] == minority_class]

    # Calculate class imbalance ratio
    imbalance_ratio = len(minority_data) / len(majority_data)

    if imbalance_ratio >= 1 - imbalance_threshold:
        print("Data is already balanced or within acceptable threshold.")
        return data

    if method == 'oversample':
        minority_upsampled = resample(
            minority_data,
            replace=True,
            n_samples=len(majority_data),
            random_state=random_state
        )
        return pd.concat([majority_data, minority_upsampled])

    elif method == 'undersample':
        majority_downsampled = resample(
            majority_data,
            replace=False,
            n_samples=len(minority_data),
            random_state=random_state
        )
        return pd.concat([majority_downsampled, minority_data])

    elif method == 'smote':
        return smote_augmentation(data, target_column, random_state)


def smote_augmentation(data: pd.DataFrame, target_column: str, random_state: int = 42) -> pd.DataFrame:
    """
    Applies Synthetic Minority Over-sampling Technique (SMOTE) to generate synthetic samples
    for the minority class.

    Parameters:
        data (pd.DataFrame): The input dataframe containing the features and target column.
        target_column (str): The name of the target column in the dataframe.
        random_state (int): Random state for reproducibility. Default is 42.

    Returns:
        pd.DataFrame: The augmented dataframe with SMOTE-applied synthetic samples.
    """
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Separate majority and minority classes
    class_counts = y.value_counts()
    majority_class = class_counts.idxmax()
    minority_class = class_counts.idxmin()

    X_minority = X[y == minority_class]

    # Fit the nearest neighbors model
    nn = NearestNeighbors(n_neighbors=5)
    nn.fit(X_minority.values)  # Use values to avoid feature names issue

    synthetic_samples = []
    for i, sample in X_minority.iterrows():
        neighbors = nn.kneighbors([sample.values], return_distance=False).flatten()
        synthetic_sample = create_synthetic_sample(X_minority.iloc[neighbors], sample, random_state)
        synthetic_samples.append(synthetic_sample)

    synthetic_data = pd.DataFrame(synthetic_samples, columns=X.columns)
    synthetic_data[target_column] = minority_class

    return pd.concat([data, synthetic_data])


def create_synthetic_sample(neighbors: pd.DataFrame, sample: pd.Series, random_state: int = 42) -> pd.Series:
    """
    Creates a synthetic sample using the k-nearest neighbors.

    Parameters:
        neighbors (pd.DataFrame): Dataframe of nearest neighbors.
        sample (pd.Series): The original minority sample.
        random_state (int): Random state for reproducibility. Default is 42.

    Returns:
        pd.Series: A synthetic sample created by interpolating between the sample and its neighbors.
    """
    np.random.seed(random_state)
    neighbor = neighbors.sample(n=1).iloc[0]
    diff = neighbor - sample
    gap = np.random.rand()
    synthetic_sample = sample + gap * diff

    return synthetic_sample


def generate_synthetic_data(
    data: pd.DataFrame,
    num_samples: int,
    strategy: str = 'random',
    random_state: int = 42
) -> pd.DataFrame:
    """
    Generates synthetic data samples based on the chosen strategy.

    Parameters:
        data (pd.DataFrame): The input dataframe to use as a base for generating synthetic data.
        num_samples (int): The number of synthetic samples to generate.
        strategy (str): The strategy to use for generating synthetic data ('random' or 'smote').
        random_state (int): Random state for reproducibility. Default is 42.

    Returns:
        pd.DataFrame: A dataframe containing the synthetic samples.
    """
    if strategy == 'random':
        return random_synthetic_generation(data, num_samples, random_state)
    elif strategy == 'smote':
        return smote_based_generation(data, num_samples, random_state)
    else:
        raise ValueError("Strategy must be either 'random' or 'smote'.")


def random_synthetic_generation(
    data: pd.DataFrame,
    num_samples: int,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Generates synthetic data by random sampling within the feature space.

    Parameters:
        data (pd.DataFrame): The input dataframe to use as a base for generating synthetic data.
        num_samples (int): The number of synthetic samples to generate.
        random_state (int): Random state for reproducibility. Default is 42.

    Returns:
        pd.DataFrame: A dataframe containing the synthetic samples.
    """
    np.random.seed(random_state)
    synthetic_samples = []

    for _ in range(num_samples):
        sample = {col: np.random.uniform(data[col].min(), data[col].max()) for col in data.columns}
        synthetic_samples.append(sample)

    synthetic_data = pd.DataFrame(synthetic_samples, columns=data.columns)
    return synthetic_data


def smote_based_generation(
    data: pd.DataFrame,
    num_samples: int,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Generates synthetic data using SMOTE-based interpolation.

    Parameters:
        data (pd.DataFrame): The input dataframe to use as a base for generating synthetic data.
        num_samples (int): The number of synthetic samples to generate.
        random_state (int): Random state for reproducibility. Default is 42.

    Returns:
        pd.DataFrame: A dataframe containing the synthetic samples.
    """
    np.random.seed(random_state)
    synthetic_data = []
    nn = NearestNeighbors(n_neighbors=5)
    nn.fit(data)

    for _ in range(num_samples):
        idx = np.random.randint(0, len(data))
        sample = data.iloc[idx]
        neighbors = nn.kneighbors([sample], return_distance=False).flatten()
        synthetic_sample = create_synthetic_sample(data.iloc[neighbors], sample, random_state)
        synthetic_data.append(synthetic_sample)

    return pd.DataFrame(synthetic_data, columns=data.columns)
