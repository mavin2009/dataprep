# tests/test_feature_engineering.py

import unittest
import pandas as pd
import numpy as np
from dataprep.feature_engineering import (
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


class TestFeatureEngineering(unittest.TestCase):
    
    def setUp(self):
        """
        Set up test data for the feature engineering tests.
        """
        # Create a simple dataframe for testing
        data = {
            'A': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'B': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
            'C': ['cat', 'dog', 'cat', 'dog', 'cat', 'dog', 'cat', 'dog', 'cat', 'dog'],
            'D': pd.to_datetime(['2021-01-01', '2021-01-02', '2021-01-03', '2021-01-04', '2021-01-05', 
                                 '2021-01-06', '2021-01-07', '2021-01-08', '2021-01-09', '2021-01-10']),
            'E': ['text data one', 'text data two', 'text data three', 'text data four', 
                  'text data five', 'text data six', 'text data seven', 'text data eight', 
                  'text data nine', 'text data ten']
        }
        self.df = pd.DataFrame(data)

    def test_log_transform(self):
        """
        Test log_transform function to apply logarithmic transformation.
        """
        transformed_df = log_transform(self.df.copy(), columns=['A', 'B'])
        self.assertTrue((transformed_df['A'] > 0).all())
        self.assertTrue((transformed_df['B'] > 0).all())

    def test_sqrt_transform(self):
        """
        Test sqrt_transform function to apply square root transformation.
        """
        transformed_df = sqrt_transform(self.df.copy(), columns=['A', 'B'])
        self.assertTrue((transformed_df['A'] >= 0).all())
        self.assertTrue((transformed_df['B'] >= 0).all())

    def test_poly_transform(self):
        """
        Test poly_transform function to generate polynomial features.
        """
        transformed_df = poly_transform(self.df.copy(), columns=['A', 'B'], degree=2)
        self.assertIn('A^2', transformed_df.columns)
        self.assertIn('B^2', transformed_df.columns)
        self.assertIn('A B', transformed_df.columns)

    def test_interaction_features(self):
        """
        Test interaction_features function to create interaction terms.
        """
        transformed_df = interaction_features(self.df.copy(), columns=['A', 'B'])
        self.assertIn('A_x_B', transformed_df.columns)

    def test_one_hot_encode(self):
        """
        Test one_hot_encode function to apply one-hot encoding.
        """
        transformed_df = one_hot_encode(self.df.copy(), columns=['C'])
        self.assertIn('C_dog', transformed_df.columns)
        self.assertNotIn('C', transformed_df.columns)

    def test_label_encode(self):
        """
        Test label_encode function to apply label encoding.
        """
        transformed_df = label_encode(self.df.copy(), columns=['C'])
        self.assertTrue(transformed_df['C'].dtype == 'int32' or transformed_df['C'].dtype == 'int64')

    def test_target_encode(self):
        """
        Test target_encode function to apply target encoding.
        """
        self.df['target'] = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]  # Simple binary target
        transformed_df = target_encode(self.df.copy(), columns=['C'], target='target')
        self.assertIn('C_encoded', transformed_df.columns)
        self.assertTrue(transformed_df['C_encoded'].isna().sum() == 0)  # Ensure no NaNs after encoding

    def test_extract_datetime_features(self):
        """
        Test extract_datetime_features function to extract features from datetime columns.
        """
        transformed_df = extract_datetime_features(self.df.copy(), datetime_columns=['D'])
        self.assertIn('D_year', transformed_df.columns)
        self.assertIn('D_month', transformed_df.columns)
        self.assertIn('D_day', transformed_df.columns)
        self.assertIn('D_dayofweek', transformed_df.columns)
        self.assertIn('D_is_weekend', transformed_df.columns)

    def test_text_to_features(self):
        """
        Test text_to_features function to convert text data into numerical features using TF-IDF.
        """
        transformed_df = text_to_features(self.df.copy(), text_columns=['E'], method='tfidf', max_features=5)
        self.assertGreaterEqual(len([col for col in transformed_df.columns if 'E_tfidf_' in col]), 1)
        self.assertNotIn('E', transformed_df.columns)  # Original text column should be dropped

if __name__ == "__main__":
    unittest.main()
