# dataprep/data_cleaning.py

"""
data_cleaning.py

This module provides functions for cleaning and preprocessing raw data, ensuring data quality
and consistency before analysis or model building. It includes functions for handling missing
values, detecting and removing outliers, and validating data types.

Functions:
    fill_na: Fills missing values in specified columns using different strategies.
    drop_na: Drops rows or columns with missing values based on a threshold.
    predict_na: Imputes missing values using predictive modeling techniques.
    detect_outliers: Identifies outliers in numeric data using various methods.
    remove_outliers: Removes outliers from the dataset based on specified criteria.
    transform_outliers: Applies transformations to mitigate the impact of outliers.
    validate_dtypes: Validates and converts data types of specified columns.
    convert_dtypes: Converts data types of specified columns to optimize memory usage.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.stats import mstats


def fill_na(
    data: pd.DataFrame,
    columns: list,
    strategy: str = 'mean',
    fill_value: any = None
) -> pd.DataFrame:
    """
    Fills missing values in specified columns using the specified strategy.

    Parameters:
        data (pd.DataFrame): The input dataframe containing features with missing values.
        columns (list): List of column names to fill missing values for.
        strategy (str): The strategy to use for filling missing values. Options are 'mean', 'median',
                        'mode', 'constant', or 'ffill'/'bfill'. Default is 'mean'.
        fill_value (any): The value to fill missing data with when using 'constant' strategy. Default is None.

    Returns:
        pd.DataFrame: A dataframe with missing values filled.
    """
    strategies = ['mean', 'median', 'mode', 'constant', 'ffill', 'bfill']
    if strategy not in strategies:
        raise ValueError(f"Invalid strategy '{strategy}'. Supported strategies are: {strategies}.")

    for col in columns:
        if strategy == 'mean':
            data[col] = data[col].fillna(data[col].mean())
        elif strategy == 'median':
            data[col] = data[col].fillna(data[col].median())
        elif strategy == 'mode':
            data[col] = data[col].fillna(data[col].mode().iloc[0])
        elif strategy == 'constant':
            if fill_value is None:
                raise ValueError("fill_value must be specified when using 'constant' strategy.")
            data[col] = data[col].fillna(fill_value)
        elif strategy == 'ffill':
            data[col] = data[col].fillna(method='ffill')
        elif strategy == 'bfill':
            data[col] = data[col].fillna(method='bfill')
    return data


def drop_na(
    data: pd.DataFrame,
    axis: int = 0,
    threshold: float = 0.5
) -> pd.DataFrame:
    """
    Drops rows or columns with missing values based on a threshold.

    Parameters:
        data (pd.DataFrame): The input dataframe containing missing values.
        axis (int): Whether to drop rows (0) or columns (1). Default is 0 (rows).
        threshold (float): The proportion of non-missing values required to retain the row/column. Default is 0.5.

    Returns:
        pd.DataFrame: A dataframe with rows or columns dropped based on missing values.
    """
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1.")

    if axis == 0:
        data = data.dropna(thresh=int(threshold * data.shape[1]), axis=axis)
    elif axis == 1:
        data = data.dropna(thresh=int(threshold * data.shape[0]), axis=axis)
    else:
        raise ValueError("Axis must be 0 (rows) or 1 (columns).")
    
    return data


def predict_na(
    data: pd.DataFrame,
    target_column: str,
    predictor_columns: list,
    model=None,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Imputes missing values in a column using a predictive model.

    Parameters:
        data (pd.DataFrame): The input dataframe containing features with missing values.
        target_column (str): The name of the column to impute.
        predictor_columns (list): List of column names to use as predictors for the imputation model.
        model: The machine learning model to use for imputation. Default is RandomForestRegressor.
        random_state (int): Random state for reproducibility. Default is 42.

    Returns:
        pd.DataFrame: A dataframe with missing values in the target column imputed.
    """
    if model is None:
        model = RandomForestRegressor(n_estimators=100, random_state=random_state)
    
    missing_idx = data[data[target_column].isna()].index
    not_missing_idx = data[~data[target_column].isna()].index

    X_train = data.loc[not_missing_idx, predictor_columns]
    y_train = data.loc[not_missing_idx, target_column]
    X_missing = data.loc[missing_idx, predictor_columns]

    model.fit(X_train, y_train)
    predictions = model.predict(X_missing)
    
    data.loc[missing_idx, target_column] = predictions
    return data


def detect_outliers(
    data: pd.DataFrame,
    columns: list,
    method: str = 'iqr',
    threshold: float = 1.5
) -> dict:
    """
    Identifies outliers in numeric data using specified methods.

    Parameters:
        data (pd.DataFrame): The input dataframe containing numeric features.
        columns (list): List of column names to detect outliers in.
        method (str): The method to use for outlier detection ('iqr' or 'zscore'). Default is 'iqr'.
        threshold (float): The threshold for identifying outliers. Default is 1.5 for 'iqr' and 3.0 for 'zscore'.

    Returns:
        dict: A dictionary with column names as keys and indices of outliers as values.
    """
    outliers = {}

    if method not in ['iqr', 'zscore']:
        raise ValueError("Method must be 'iqr' or 'zscore'.")

    for col in columns:
        if method == 'iqr':
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outliers[col] = data[(data[col] < lower_bound) | (data[col] > upper_bound)].index.tolist()
        elif method == 'zscore':
            mean = data[col].mean()
            std = data[col].std()
            outliers[col] = data[(np.abs((data[col] - mean) / std) > threshold)].index.tolist()

    return outliers


def remove_outliers(
    data: pd.DataFrame,
    columns: list,
    method: str = 'iqr',
    threshold: float = 1.5
) -> pd.DataFrame:
    """
    Removes outliers from the dataset based on specified criteria.

    Parameters:
        data (pd.DataFrame): The input dataframe containing numeric features.
        columns (list): List of column names to remove outliers from.
        method (str): The method to use for outlier detection ('iqr' or 'zscore'). Default is 'iqr'.
        threshold (float): The threshold for identifying outliers. Default is 1.5 for 'iqr' and 3.0 for 'zscore'.

    Returns:
        pd.DataFrame: A dataframe with outliers removed from the specified columns.
    """
    outliers = detect_outliers(data, columns, method, threshold)
    for col, outlier_indices in outliers.items():
        data = data.drop(index=outlier_indices)
    return data


def transform_outliers(
    data: pd.DataFrame,
    columns: list,
    method: str = 'winsorize',
    limits: float = 0.05
) -> pd.DataFrame:
    """
    Applies transformations to mitigate the impact of outliers.

    Parameters:
        data (pd.DataFrame): The input dataframe containing numeric features.
        columns (list): List of column names to transform outliers for.
        method (str): The method to use for transforming outliers ('winsorize' or 'clip'). Default is 'winsorize'.
        limits (float or tuple): The limits for winsorizing or clipping. Default is 0.05 (5th and 95th percentile).

    Returns:
        pd.DataFrame: A dataframe with outliers transformed in the specified columns.
    """
    if method not in ['winsorize', 'clip']:
        raise ValueError("Method must be 'winsorize' or 'clip'.")

    for col in columns:
        if method == 'winsorize':
            # Apply clipping instead if winsorize does not work as expected
            lower_limit = data[col].quantile(limits)
            upper_limit = data[col].quantile(1 - limits)
            data[col] = data[col].clip(lower=lower_limit, upper=upper_limit)
            

        elif method == 'clip':
            lower_limit = data[col].quantile(limits)
            upper_limit = data[col].quantile(1 - limits)
            data[col] = data[col].clip(lower=lower_limit, upper=upper_limit)
    
    return data

def validate_dtypes(
    data: pd.DataFrame,
    columns: dict
) -> pd.DataFrame:
    """
    Validates and converts data types of specified columns.

    Parameters:
        data (pd.DataFrame): The input dataframe to validate data types for.
        columns (dict): Dictionary with column names as keys and desired data types as values.

    Returns:
        pd.DataFrame: A dataframe with validated and converted data types.
    """
    for col, dtype in columns.items():
        try:
            data[col] = data[col].astype(dtype)
        except ValueError as e:
            raise ValueError(f"Cannot convert column {col} to {dtype}: {e}")
    return data


def convert_dtypes(data: pd.DataFrame) -> pd.DataFrame:
    """
    Converts data types of columns to optimize memory usage.

    Parameters:
        data (pd.DataFrame): The input dataframe to optimize data types for.

    Returns:
        pd.DataFrame: A dataframe with optimized data types.
    """
    float_cols = data.select_dtypes(include='float').columns
    int_cols = data.select_dtypes(include='int').columns
    object_cols = data.select_dtypes(include='object').columns

    data[float_cols] = data[float_cols].astype('float32')
    data[int_cols] = data[int_cols].apply(pd.to_numeric, downcast='unsigned')
    data[object_cols] = data[object_cols].astype('category')

    return data
