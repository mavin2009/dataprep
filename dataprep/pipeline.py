# dataprep/pipeline.py

"""
pipeline.py

This module provides utilities for creating and managing preprocessing pipelines.
A pipeline allows for sequential execution of data preprocessing steps, ensuring
consistency and reproducibility in data transformations.

Functions:
    create_pipeline: Creates a preprocessing pipeline with specified steps.
    run_pipeline: Executes a preprocessing pipeline on a given dataset.
    save_pipeline: Saves a pipeline configuration to a file.
    load_pipeline: Loads a pipeline configuration from a file.

Classes:
    DataPrepPipeline: A class representing a preprocessing pipeline for data preparation.
"""

import pandas as pd
import pickle
from typing import List, Callable, Tuple, Any


class DataPrepPipeline:
    """
    A class representing a preprocessing pipeline for data preparation.

    Attributes:
        steps (List[Tuple[str, Callable]]): A list of tuples containing step names and functions.
    """

    def __init__(self):
        """
        Initializes an empty DataPrepPipeline.
        """
        self.steps = []

    def add_step(self, name: str, func: Callable) -> None:
        """
        Adds a preprocessing step to the pipeline.

        Parameters:
            name (str): The name of the preprocessing step.
            func (Callable): The function to execute for this step.
        """
        if not callable(func):
            raise ValueError(f"The provided function '{func}' is not callable.")
        self.steps.append((name, func))

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Executes the pipeline on the provided dataset.

        Parameters:
            data (pd.DataFrame): The input dataframe to process.

        Returns:
            pd.DataFrame: The processed dataframe after applying all pipeline steps.
        """
        for name, func in self.steps:
            print(f"Running step: {name}")
            data = func(data)
        return data

    def remove_step(self, name: str) -> None:
        """
        Removes a step from the pipeline by name.

        Parameters:
            name (str): The name of the step to remove.
        """
        self.steps = [(n, f) for n, f in self.steps if n != name]

    def list_steps(self) -> List[str]:
        """
        Lists all the steps currently in the pipeline.

        Returns:
            List[str]: A list of step names in the pipeline.
        """
        return [name for name, _ in self.steps]


def create_pipeline() -> DataPrepPipeline:
    """
    Creates an empty preprocessing pipeline.

    Returns:
        DataPrepPipeline: An instance of DataPrepPipeline.
    """
    return DataPrepPipeline()


def run_pipeline(pipeline: DataPrepPipeline, data: pd.DataFrame) -> pd.DataFrame:
    """
    Runs the given preprocessing pipeline on the dataset.

    Parameters:
        pipeline (DataPrepPipeline): The pipeline to run.
        data (pd.DataFrame): The input dataframe to process.

    Returns:
        pd.DataFrame: The processed dataframe.
    """
    return pipeline.run(data)


def save_pipeline(pipeline: DataPrepPipeline, file_path: str) -> None:
    """
    Saves the pipeline configuration to a file using pickle.

    Parameters:
        pipeline (DataPrepPipeline): The pipeline to save.
        file_path (str): The path to the file where the pipeline will be saved.
    """
    with open(file_path, 'wb') as file:
        pickle.dump(pipeline, file)
    print(f"Pipeline saved to {file_path}")


def load_pipeline(file_path: str) -> DataPrepPipeline:
    """
    Loads a pipeline configuration from a file.

    Parameters:
        file_path (str): The path to the file containing the saved pipeline.

    Returns:
        DataPrepPipeline: The loaded pipeline instance.
    """
    with open(file_path, 'rb') as file:
        pipeline = pickle.load(file)
    print(f"Pipeline loaded from {file_path}")
    return pipeline
