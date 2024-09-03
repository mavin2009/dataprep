# tests/test_pipeline.py

import unittest
import pandas as pd
import os
from dataprep.pipeline import (
    create_pipeline,
    run_pipeline,
    save_pipeline,
    load_pipeline,
    DataPrepPipeline
)

# Define the dummy step outside the test class for pickling
def dummy_step(data):
    return data * 2

class TestPipeline(unittest.TestCase):

    def setUp(self):
        """
        Set up test data and initialize pipeline for the tests.
        """
        # Create a simple dataframe for testing
        self.data = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [5, 4, 3, 2, 1]
        })

        # Initialize a DataPrepPipeline instance
        self.pipeline = create_pipeline()

    def test_create_pipeline(self):
        """
        Test create_pipeline function to ensure it initializes an empty pipeline.
        """
        pipeline = create_pipeline()
        self.assertIsInstance(pipeline, DataPrepPipeline)
        self.assertEqual(len(pipeline.steps), 0, "Pipeline should be initialized with no steps.")

    def test_add_step(self):
        """
        Test add_step function to add a step to the pipeline.
        """
        self.pipeline.add_step('Double Data', dummy_step)
        self.assertEqual(len(self.pipeline.steps), 1, "Pipeline should have 1 step after adding.")

    def test_run_pipeline(self):
        """
        Test run_pipeline function to run the pipeline on a dataset.
        """
        self.pipeline.add_step('Double A', dummy_step)
        result = run_pipeline(self.pipeline, self.data.copy())
        self.assertTrue((result['A'] == self.data['A'] * 2).all(), "Column 'A' should be doubled.")

    def test_remove_step(self):
        """
        Test remove_step function to remove a step from the pipeline.
        """
        self.pipeline.add_step('Double Data', dummy_step)
        self.pipeline.remove_step('Double Data')
        self.assertEqual(len(self.pipeline.steps), 0, "Pipeline should have 0 steps after removal.")

    def test_list_steps(self):
        """
        Test list_steps function to list all steps in the pipeline.
        """
        self.pipeline.add_step('Double Data', dummy_step)
        steps = self.pipeline.list_steps()
        self.assertIn('Double Data', steps, "Step name should be in the list of steps.")

    def test_save_and_load_pipeline(self):
        """
        Test save_pipeline and load_pipeline functions for saving and loading a pipeline.
        """
        self.pipeline.add_step('Double Data', dummy_step)
        file_path = 'test_pipeline.pkl'
        
        # Save the pipeline
        save_pipeline(self.pipeline, file_path)
        self.assertTrue(os.path.exists(file_path), "Pipeline file should exist after saving.")
        
        # Load the pipeline
        loaded_pipeline = load_pipeline(file_path)
        self.assertIsInstance(loaded_pipeline, DataPrepPipeline)
        self.assertEqual(len(loaded_pipeline.steps), 1, "Loaded pipeline should have 1 step.")
        
        # Clean up the test file
        os.remove(file_path)

    def test_invalid_add_step(self):
        """
        Test add_step function with invalid function to ensure proper error handling.
        """
        with self.assertRaises(ValueError):
            self.pipeline.add_step('Invalid Step', 'not_a_function')

if __name__ == "__main__":
    unittest.main()
