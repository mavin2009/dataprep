# dataprep/feature_engineering.py

"""
feature_engineering.py

This module provides functions for automated feature engineering, including transformations,
encoding, and generation of new features. The goal is to streamline the process of preparing
data for machine learning models, enhancing model performance by providing meaningful features.

Functions:
    log_transform: Applies logarithmic transformation to numeric features.
    sqrt_transform: Applies square root transformation to numeric features.
    poly_transform: Generates polynomial features from existing numeric features.
    interaction_features: Creates interaction terms between numeric features.
    one_hot_encode: Applies one-hot encoding to categorical features.
    label_encode: Applies label encoding to categorical features.
    target_encode: Applies target encoding to categorical features.
    extract_datetime_features: Extracts useful features from datetime columns.
    text_to_features: Converts text data into numerical features using TF-IDF or other methods.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from collections import defaultdict

def log_transform(data: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Applies logarithmic transformation to specified numeric features.

    Parameters:
        data (pd.DataFrame): The input dataframe containing features to be transformed.
        columns (list): List of column names to apply the logarithmic transformation to.

    Returns:
        pd.DataFrame: A dataframe with the specified columns transformed.
    """
    for col in columns:
        if np.any(data[col] <= 0):
            raise ValueError(f"Column {col} contains non-positive values, cannot apply log transform.")
        data[col] = np.log1p(data[col])
    return data


def sqrt_transform(data: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Applies square root transformation to specified numeric features.

    Parameters:
        data (pd.DataFrame): The input dataframe containing features to be transformed.
        columns (list): List of column names to apply the square root transformation to.

    Returns:
        pd.DataFrame: A dataframe with the specified columns transformed.
    """
    for col in columns:
        if np.any(data[col] < 0):
            raise ValueError(f"Column {col} contains negative values, cannot apply square root transform.")
        data[col] = np.sqrt(data[col])
    return data


def poly_transform(data: pd.DataFrame, columns: list, degree: int = 2, include_bias: bool = False) -> pd.DataFrame:
    """
    Generates polynomial features from specified numeric features.

    Parameters:
        data (pd.DataFrame): The input dataframe containing features to be transformed.
        columns (list): List of column names to generate polynomial features from.
        degree (int): The degree of the polynomial features. Default is 2.
        include_bias (bool): Whether to include a bias (intercept) column. Default is False.

    Returns:
        pd.DataFrame: A dataframe with the original and polynomial features.
    """
    poly = PolynomialFeatures(degree=degree, include_bias=include_bias)
    poly_features = poly.fit_transform(data[columns])
    poly_feature_names = poly.get_feature_names_out(columns)
    poly_df = pd.DataFrame(poly_features, columns=poly_feature_names, index=data.index)
    data = pd.concat([data, poly_df.drop(columns, axis=1)], axis=1)
    return data


def interaction_features(data: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Creates interaction terms between specified numeric features.

    Parameters:
        data (pd.DataFrame): The input dataframe containing features to be transformed.
        columns (list): List of column names to create interaction features between.

    Returns:
        pd.DataFrame: A dataframe with the interaction terms added.
    """
    interaction_df = pd.DataFrame(index=data.index)
    for i, col1 in enumerate(columns):
        for col2 in columns[i+1:]:
            interaction_df[f"{col1}_x_{col2}"] = data[col1] * data[col2]
    data = pd.concat([data, interaction_df], axis=1)
    return data


def one_hot_encode(data: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Applies one-hot encoding to specified categorical features.

    Parameters:
        data (pd.DataFrame): The input dataframe containing features to be transformed.
        columns (list): List of column names to apply one-hot encoding to.

    Returns:
        pd.DataFrame: A dataframe with the specified columns one-hot encoded.
    """
    encoder = OneHotEncoder(sparse_output=False, drop='first')  # Update argument to 'sparse_output'
    encoded_data = encoder.fit_transform(data[columns])
    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(columns), index=data.index)
    data = pd.concat([data.drop(columns, axis=1), encoded_df], axis=1)
    return data


def label_encode(data: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Applies label encoding to specified categorical features.

    Parameters:
        data (pd.DataFrame): The input dataframe containing features to be transformed.
        columns (list): List of column names to apply label encoding to.

    Returns:
        pd.DataFrame: A dataframe with the specified columns label encoded.
    """
    le_dict = {}
    for col in columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        le_dict[col] = le
    return data


def target_encode(data: pd.DataFrame, columns: list, target: str, n_splits: int = 5) -> pd.DataFrame:
    """
    Applies target encoding to specified categorical features using K-Fold strategy.

    Parameters:
        data (pd.DataFrame): The input dataframe containing features to be transformed.
        columns (list): List of column names to apply target encoding to.
        target (str): The target column for encoding.
        n_splits (int): Number of splits for K-Fold encoding. Default is 5.

    Returns:
        pd.DataFrame: A dataframe with the specified columns target encoded.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    for col in columns:
        encoded_col = pd.Series(index=data.index, dtype='float64')
        for train_idx, val_idx in kf.split(data):
            train, val = data.iloc[train_idx], data.iloc[val_idx]
            means = train.groupby(col)[target].mean()
            encoded_col.iloc[val_idx] = val[col].map(means)
        data[col + "_encoded"] = encoded_col.fillna(data[target].mean())
    return data


def extract_datetime_features(data: pd.DataFrame, datetime_columns: list) -> pd.DataFrame:
    """
    Extracts useful features from datetime columns such as year, month, day, etc.

    Parameters:
        data (pd.DataFrame): The input dataframe containing datetime features.
        datetime_columns (list): List of datetime column names to extract features from.

    Returns:
        pd.DataFrame: A dataframe with additional datetime-related features.
    """
    for col in datetime_columns:
        data[f"{col}_year"] = data[col].dt.year
        data[f"{col}_month"] = data[col].dt.month
        data[f"{col}_day"] = data[col].dt.day
        data[f"{col}_dayofweek"] = data[col].dt.dayofweek
        data[f"{col}_is_weekend"] = data[col].dt.dayofweek >= 5
    return data


def text_to_features(data: pd.DataFrame, text_columns: list, method: str = 'tfidf', max_features: int = 100) -> pd.DataFrame:
    """
    Converts text data into numerical features using specified methods like TF-IDF.

    Parameters:
        data (pd.DataFrame): The input dataframe containing text features.
        text_columns (list): List of text column names to be transformed into features.
        method (str): The method to convert text to features ('tfidf'). Default is 'tfidf'.
        max_features (int): Maximum number of features to extract from text. Default is 100.

    Returns:
        pd.DataFrame: A dataframe with text features transformed into numerical features.
    """
    if method == 'tfidf':
        vectorizer = TfidfVectorizer(max_features=max_features)
        for col in text_columns:
            tfidf_matrix = vectorizer.fit_transform(data[col])
            tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=[f"{col}_tfidf_{i}" for i in range(tfidf_matrix.shape[1])], index=data.index)
            data = pd.concat([data, tfidf_df], axis=1).drop(col, axis=1)
    else:
        raise ValueError("Currently, only 'tfidf' method is supported for text feature extraction.")
    return data
