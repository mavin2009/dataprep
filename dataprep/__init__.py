# dataprep/__init__.py

"""
Dataprep: A Python library for automated data preprocessing and feature engineering.

Modules:
    data_cleaning - Functions for cleaning and preparing raw data.
    feature_engineering - Tools for generating and transforming features for machine learning models.
    visualization - Quick visualization utilities for exploratory data analysis.
    pipeline - Utilities for creating and managing preprocessing pipelines.
    augmentation - Functions for data augmentation and synthetic data generation.

Author: Michael Avina
License: MIT License
"""

from .data_cleaning import (
    fill_na,
    drop_na,
    predict_na,
    detect_outliers,
    remove_outliers,
    transform_outliers,
    validate_dtypes,
    convert_dtypes
)

from .feature_engineering import (
    log_transform,
    sqrt_transform,
    poly_transform,
    interaction_features,
    one_hot_encode,
    label_encode,
    target_encode,
    extract_datetime_features,
    text_to_features
)

from .visualization import (
    plot_histogram,
    plot_scatter,
    plot_pairplot,
    generate_eda_report
)

from .pipeline import (
    create_pipeline,
    run_pipeline,
    save_pipeline,
    load_pipeline
)

from .augmentation import (
    augment_data,
    generate_synthetic_data
)

__all__ = [
    'fill_na', 'drop_na', 'predict_na', 'detect_outliers', 'remove_outliers', 'transform_outliers',
    'validate_dtypes', 'convert_dtypes', 'log_transform', 'sqrt_transform', 'poly_transform',
    'interaction_features', 'one_hot_encode', 'label_encode', 'target_encode',
    'extract_datetime_features', 'text_to_features', 'plot_histogram', 'plot_scatter', 'plot_pairplot',
    'generate_eda_report', 'create_pipeline', 'run_pipeline', 'save_pipeline', 'load_pipeline',
    'augment_data', 'generate_synthetic_data'
]
