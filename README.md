# Dataprep

**Dataprep** is a Python library for automated data preprocessing and feature engineering, designed to simplify and speed up the process of preparing data for machine learning models. The library provides a comprehensive set of tools for data cleaning, feature engineering, visualization, augmentation, and creating preprocessing pipelines.

## Why Dataprep?

Data preprocessing is a crucial step in any data science or machine learning workflow. However, it can often be time-consuming and error-prone, requiring repetitive coding for tasks like handling missing values, encoding categorical variables, generating new features, and visualizing data. **Dataprep** was created to streamline these tasks, providing an intuitive and easy-to-use interface that automates many of the common preprocessing steps. This allows data scientists to focus more on model building and less on the tedious preprocessing work.

### Key Features

- **Data Cleaning**: Automated handling of missing values, outlier detection and treatment, and data type validation.
- **Feature Engineering**: Tools for generating new features, such as polynomial features, interaction terms, and target encoding.
- **Data Augmentation**: Techniques for augmenting data to handle class imbalance or generate synthetic data.
- **Visualization**: Quick visualization utilities for exploratory data analysis (EDA), including histograms, scatter plots, pair plots, and correlation heatmaps.
- **Pipeline Management**: Utilities for creating and managing preprocessing pipelines, enabling reproducible and consistent data transformations.

## Installation

You can install **Dataprep** via pip

## Usage

### 1. Data Cleaning

```python
import pandas as pd
from dataprep import fill_na, detect_outliers, remove_outliers

# Load your dataset
data = pd.read_csv('your_dataset.csv')

# Fill missing values
data = fill_na(data, columns=['Age', 'Income'], strategy='mean')

# Detect outliers
outliers = detect_outliers(data, columns=['Income'], method='iqr', threshold=1.5)
print(f"Outliers detected in 'Income': {outliers}")

# Remove outliers
data = remove_outliers(data, columns=['Income'], method='iqr', threshold=1.5)

### 2. Feature Engineering
```python
from dataprep import poly_transform, one_hot_encode, extract_datetime_features

# Generate polynomial features
data = poly_transform(data, columns=['Age', 'Income'], degree=2)

# One-hot encode categorical variables
data = one_hot_encode(data, columns=['Gender', 'Occupation'])

# Extract features from datetime column
data = extract_datetime_features(data, datetime_columns=['Registration_Date'])

### 3. Data Augmentation
```python
from dataprep import augment_data

# Augment data using SMOTE for class imbalance
data_augmented = augment_data(data, target_column='Target', method='smote')

### 4. Visualization
```python
from dataprep import plot_histogram, plot_scatter, plot_heatmap, generate_eda_report

# Plot histogram of numeric features
plot_histogram(data, columns=['Age', 'Income'])

# Plot scatter plot
plot_scatter(data, x='Age', y='Income', hue='Gender')

# Generate a comprehensive EDA report
generate_eda_report(data, target='Target')

### 5. Create and Run a Preprocessing Pipeline
```python
from dataprep import create_pipeline, run_pipeline, save_pipeline, load_pipeline

# Create a preprocessing pipeline
pipeline = create_pipeline()
pipeline.add_step('Fill Missing Values', lambda df: fill_na(df, columns=['Age', 'Income'], strategy='mean'))
pipeline.add_step('Generate Polynomial Features', lambda df: poly_transform(df, columns=['Age', 'Income'], degree=2))
pipeline.add_step('One-Hot Encode Categorical Variables', lambda df: one_hot_encode(df, columns=['Gender', 'Occupation']))

# Run the pipeline
data_processed = run_pipeline(pipeline, data)

# Save the pipeline
save_pipeline(pipeline, 'preprocessing_pipeline.pkl')

# Load the pipeline
loaded_pipeline = load_pipeline('preprocessing_pipeline.pkl')

# Run the loaded pipeline
data_processed_loaded = run_pipeline(loaded_pipeline, data)

#### `LICENSE`

```plaintext
MIT License

MIT License

Copyright (c) 2024 Michael Avina

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.