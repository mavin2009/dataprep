# dataprep/tests/test_augmentation.py

"""
test_augmentation.py

Unit tests for the augmentation module of the dataprep library.
These tests validate the functionality of data augmentation methods such as SMOTE,
random oversampling, undersampling, and synthetic data generation.
"""

import unittest
import pandas as pd
import numpy as np
from dataprep.augmentation import augment_data, generate_synthetic_data


class TestAugmentation(unittest.TestCase):
    def setUp(self):
        """
        Set up test data for the augmentation tests.
        """
        # Create a simple dataframe for testing
        data = {
            'Feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'Feature2': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
            'Target': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        }
        self.df = pd.DataFrame(data)

    def test_augment_data_smote(self):
        """
        Test augment_data function using SMOTE method.
        """
        # Perform SMOTE augmentation
        augmented_df = augment_data(self.df, target_column='Target', method='smote', random_state=42)
        
        # Check the number of samples for each class
        target_counts = augmented_df['Target'].value_counts()
        self.assertEqual(target_counts[0], target_counts[1], "The classes are not balanced after SMOTE augmentation.")

    def test_augment_data_oversample(self):
        """
        Test augment_data function using oversampling method.
        """
        # Perform oversampling
        oversampled_df = augment_data(self.df, target_column='Target', method='oversample', random_state=42)
        
        # Check the number of samples for each class
        target_counts = oversampled_df['Target'].value_counts()
        self.assertEqual(target_counts[0], target_counts[1], "The classes are not balanced after oversampling.")

    def test_augment_data_undersample(self):
        """
        Test augment_data function using undersampling method.
        """
        # Perform undersampling
        undersampled_df = augment_data(self.df, target_column='Target', method='undersample', random_state=42)
        
        # Check the number of samples for each class
        target_counts = undersampled_df['Target'].value_counts()
        self.assertEqual(target_counts[0], target_counts[1], "The classes are not balanced after undersampling.")
    
    def test_generate_synthetic_data_random(self):
        """
        Test generate_synthetic_data function using random sampling method.
        """
        # Generate synthetic data
        synthetic_df = generate_synthetic_data(self.df.drop(columns=['Target']), num_samples=5, strategy='random', random_state=42)
        
        # Check the shape of the generated data
        self.assertEqual(synthetic_df.shape[0], 5, "The number of synthetic samples generated is incorrect.")
        self.assertEqual(synthetic_df.shape[1], 2, "The number of columns in the synthetic data is incorrect.")
    
    def test_generate_synthetic_data_smote(self):
        """
        Test generate_synthetic_data function using SMOTE-based method.
        """
        # Generate synthetic data using SMOTE-based method
        synthetic_df = generate_synthetic_data(self.df.drop(columns=['Target']), num_samples=5, strategy='smote', random_state=42)
        
        # Check the shape of the generated data
        self.assertEqual(synthetic_df.shape[0], 5, "The number of synthetic samples generated is incorrect.")
        self.assertEqual(synthetic_df.shape[1], 2, "The number of columns in the synthetic data is incorrect.")

    def test_invalid_augmentation_method(self):
        """
        Test augment_data function with an invalid method.
        """
        with self.assertRaises(ValueError):
            augment_data(self.df, target_column='Target', method='invalid_method')

    def test_invalid_synthetic_generation_strategy(self):
        """
        Test generate_synthetic_data function with an invalid strategy.
        """
        with self.assertRaises(ValueError):
            generate_synthetic_data(self.df.drop(columns=['Target']), num_samples=5, strategy='invalid_strategy')

if __name__ == "__main__":
    unittest.main()
