# dataprep/visualization.py

"""
visualization.py

This module provides quick visualization utilities for exploratory data analysis (EDA).
It includes functions for common data visualizations such as histograms, scatter plots,
pair plots, and heatmaps. These visualizations help in understanding data distributions,
relationships, and identifying potential issues like outliers or missing data.

Functions:
    plot_histogram: Plots histograms for specified numeric features.
    plot_scatter: Plots scatter plots between two specified features.
    plot_pairplot: Plots a pairplot for selected features to visualize pairwise relationships.
    plot_heatmap: Plots a correlation heatmap for numeric features.
    generate_eda_report: Generates a comprehensive EDA report with multiple visualizations and statistics.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_histogram(data: pd.DataFrame, columns: list, bins: int = 10) -> None:
    """
    Plots histograms for the specified numeric features.

    Parameters:
        data (pd.DataFrame): The input dataframe containing numeric features to plot.
        columns (list): List of column names to plot histograms for.
        bins (int): Number of bins for the histograms. Default is 10.

    Returns:
        None
    """
    data[columns].hist(bins=bins, figsize=(12, 8), edgecolor='black')
    plt.suptitle('Histogram of Numeric Features')
    plt.show()


def plot_scatter(data: pd.DataFrame, x: str, y: str, hue: str = None) -> None:
    """
    Plots a scatter plot between two specified features.

    Parameters:
        data (pd.DataFrame): The input dataframe containing features to plot.
        x (str): The column name for the x-axis.
        y (str): The column name for the y-axis.
        hue (str): The column name for color encoding. Default is None.

    Returns:
        None
    """
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=x, y=y, hue=hue, data=data)
    plt.title(f'Scatter Plot of {x} vs {y}')
    plt.xlabel(x)
    plt.ylabel(y)
    plt.show()


def plot_pairplot(data: pd.DataFrame, columns: list, hue: str = None) -> None:
    """
    Plots a pairplot for selected features to visualize pairwise relationships.

    Parameters:
        data (pd.DataFrame): The input dataframe containing features to plot.
        columns (list): List of column names to include in the pairplot.
        hue (str): The column name for color encoding. Default is None.

    Returns:
        None
    """
    sns.pairplot(data[columns], hue=hue)
    plt.suptitle('Pairplot of Selected Features', y=1.02)
    plt.show()


def plot_heatmap(data: pd.DataFrame, annot: bool = True, cmap: str = 'coolwarm') -> None:
    """
    Plots a correlation heatmap for numeric features in the dataframe.

    Parameters:
        data (pd.DataFrame): The input dataframe containing numeric features.
        annot (bool): Whether to annotate the heatmap with correlation coefficients. Default is True.
        cmap (str): The color map to use for the heatmap. Default is 'coolwarm'.

    Returns:
        None
    """
    plt.figure(figsize=(10, 8))
    correlation_matrix = data.corr()
    sns.heatmap(correlation_matrix, annot=annot, cmap=cmap, fmt=".2f")
    plt.title('Correlation Heatmap')
    plt.show()


def generate_eda_report(data: pd.DataFrame, target: str = None) -> None:
    """
    Generates a comprehensive EDA report with multiple visualizations and statistics.

    Parameters:
        data (pd.DataFrame): The input dataframe for which to generate the EDA report.
        target (str): The target column for additional analysis (optional). Default is None.

    Returns:
        None
    """
    print("Generating EDA Report...\n")
    
    # Display basic statistics
    print("Basic Statistics:\n")
    print(data.describe(include='all').transpose())
    print("\n")

    # Plot histograms for numeric features
    numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    print("Plotting Histograms for Numeric Features...\n")
    plot_histogram(data, numeric_columns)

    # Plot correlation heatmap
    print("Plotting Correlation Heatmap...\n")
    plot_heatmap(data)

    # Pairplot for numeric features if target is provided
    if target:
        print(f"Plotting Pairplot with '{target}' as hue...\n")
        plot_pairplot(data, numeric_columns, hue=target)

    # Scatter plots between target and numeric features if target is provided
    if target and target in numeric_columns:
        for col in numeric_columns:
            if col != target:
                print(f"Plotting Scatter Plot between '{target}' and '{col}'...\n")
                plot_scatter(data, x=col, y=target)

    print("EDA Report Generation Complete.")
