# dataprep/tests/test_data_cleaning.py

import unittest
import pandas as pd
import numpy as np
from dataprep.data_cleaning import (
    fill_na,
    drop_na,
    predict_na,
    detect_outliers,
    remove_outliers,
    transform_outliers,
    validate_dtypes,
    convert_dtypes
)
from sklearn.ensemble import RandomForestRegressor


class TestDataCleaning(unittest.TestCase):
    def setUp(self):
        """
        Set up test data for the data cleaning tests.
        """
        # Create a simple dataframe for testing
        data = {
            'A': [1, 2, np.nan, 4, 5, np.nan, 7, 8, 9, 10],
            'B': [10, np.nan, 8, 7, np.nan, 5, 4, 3, 2, 1],
            'C': ['cat', 'dog', 'cat', 'dog', 'cat', 'dog', 'cat', np.nan, 'dog', 'cat'],
            'D': pd.to_datetime(['2021-01-01', '2021-01-02', '2021-01-03', np.nan, '2021-01-05', 
                                 '2021-01-06', '2021-01-07', '2021-01-08', '2021-01-09', '2021-01-10'])
        }
        self.df = pd.DataFrame(data)

    def test_fill_na_mean(self):
        """
        Test fill_na function using mean strategy.
        """
        filled_df = fill_na(self.df.copy(), columns=['A', 'B'], strategy='mean')
        expected_mean_A = self.df['A'].mean()
        expected_mean_B = self.df['B'].mean()
        self.assertAlmostEqual(filled_df['A'].iloc[2], expected_mean_A, places=3)
        self.assertAlmostEqual(filled_df['B'].iloc[1], expected_mean_B, places=1)

    def test_fill_na_constant(self):
        """
        Test fill_na function using constant strategy.
        """
        filled_df = fill_na(self.df.copy(), columns=['A', 'B'], strategy='constant', fill_value=0)
        self.assertEqual(filled_df['A'].iloc[2], 0)
        self.assertEqual(filled_df['B'].iloc[1], 0)

    def test_drop_na(self):
        """
        Test drop_na function to drop rows with missing values.
        """
        dropped_df = drop_na(self.df.copy(), axis=0, threshold=0.75)
        # Rows with at least 75% non-NA values should remain
        expected_rows = self.df.dropna(thresh=int(0.75 * self.df.shape[1])).shape[0]
        self.assertEqual(dropped_df.shape[0], expected_rows)

    def test_predict_na(self):
        """
        Test predict_na function to impute missing values using a predictive model.
        """
        # Train a model to predict missing values in column 'A' using column 'B'
        predictors = ['B']
        imputed_df = predict_na(self.df.copy(), target_column='A', predictor_columns=predictors)
        self.assertFalse(imputed_df['A'].isna().any(), "There are still missing values in column 'A'.")

    def test_detect_outliers_iqr(self):
        """
        Test detect_outliers function using IQR method.
        """
        outliers = detect_outliers(self.df.copy(), columns=['A'], method='iqr', threshold=1.5)
        self.assertEqual(len(outliers['A']), 0, "Unexpected outliers detected in column 'A'.")

    def test_detect_outliers_zscore(self):
        """
        Test detect_outliers function using z-score method.
        """
        outliers = detect_outliers(self.df.copy(), columns=['A'], method='zscore', threshold=2)
        self.assertEqual(len(outliers['A']), 0, "Unexpected outliers detected in column 'A'.")

    def test_remove_outliers(self):
        """
        Test remove_outliers function to remove outliers from data.
        """
        self.df.loc[0, 'A'] = 100  # Introduce an outlier
        cleaned_df = remove_outliers(self.df.copy(), columns=['A'], method='iqr', threshold=1.5)
        self.assertNotIn(100, cleaned_df['A'], "Outlier not removed from column 'A'.")

    def test_transform_outliers_winsorize(self):
        """
        Test transform_outliers function using winsorize method.
        """
        self.df.loc[0, 'A'] = 100  # Introduce an outlier
        
        
        transformed_df = transform_outliers(self.df.copy(), columns=['A'], method='winsorize', limits=0.05)
        
        
        # Calculate the 95th percentile cap value after winsorization
        original_values = self.df['A'].dropna().copy()
        cap_value = np.percentile(original_values, 95)
        

        # Check if the maximum value in the column is less than or equal to the 95th percentile value
        self.assertLessEqual(transformed_df['A'].max(), cap_value, "Outlier not winsorized in column 'A'.")
        
        # Verify that the specific outlier (100) is not in the transformed data
        self.assertNotEqual(transformed_df['A'].iloc[0], 100, "Outlier was not transformed correctly.")
    def test_validate_dtypes(self):
        """
        Test validate_dtypes function to validate and convert data types.
        """
        converted_df = validate_dtypes(self.df.copy(), columns={'A': 'float64', 'C': 'category'})
        self.assertEqual(converted_df['A'].dtype, 'float64')
        self.assertTrue(isinstance(converted_df['C'].dtype, pd.CategoricalDtype))

    def test_convert_dtypes(self):
        """
        Test convert_dtypes function to optimize data types for memory usage.
        """
        optimized_df = convert_dtypes(self.df.copy())
        self.assertTrue(optimized_df['A'].dtype == 'float32' or optimized_df['A'].dtype == 'float16')
        self.assertTrue(isinstance(optimized_df['C'].dtype, pd.CategoricalDtype))

    def test_invalid_fill_na_strategy(self):
        """
        Test fill_na function with an invalid strategy.
        """
        with self.assertRaises(ValueError):
            fill_na(self.df.copy(), columns=['A'], strategy='invalid_strategy')

    def test_invalid_predict_na_model(self):
        """
        Test predict_na function with an invalid model.
        """
        with self.assertRaises(AttributeError):
            predict_na(self.df.copy(), target_column='A', predictor_columns=['B'], model='invalid_model')

if __name__ == "__main__":
    unittest.main()
