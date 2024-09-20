import unittest
import pandas as pd
import os
os.chdir(r'c:\Users\habteyes.asfaw\10Accadamy\Rossmann-Sales-Prediction'
)
from src.eda_plots import sales_customers_correlation, assortment_type_sales, competitor_distance_effect, customer_behavior_opening_closing

class TestEDAFunctions(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Set up the sample dataframe for testing
        cls.train_df_clean = pd.DataFrame({
            'Sales': [5000, 7000, 6500, 8000, 7200],
            'Customers': [300, 450, 400, 500, 460],
            'Assortment': ['a', 'b', 'c', 'a', 'b'],
            'CompetitionDistance': [150, 300, 500, 800, 1000],
            'Hour': [9, 10, 11, 12, 13],
            'Date': pd.to_datetime(['2023-09-01 09:00:00', '2023-09-01 10:00:00', '2023-09-01 11:00:00', '2023-09-01 12:00:00', '2023-09-01 13:00:00'])
        })


   

    def test_assortment_type_sales(self):
        """Test the effect of assortment type on sales."""
        # Check if the 'Assortment' and 'Sales' columns are in the dataframe
        self.assertIn('Assortment', self.train_df_clean.columns, "'Assortment' column missing in the dataframe")
        self.assertIn('Sales', self.train_df_clean.columns, "'Sales' column missing in the dataframe")

        try:
            assortment_type_sales(self.train_df_clean)
        except Exception as e:
            self.fail(f"assortment_type_sales raised an exception: {e}")

    def test_competitor_distance_effect(self):
        """Test the effect of competitor distance on sales."""
        # Check if the 'CompetitionDistance' and 'Sales' columns are in the dataframe
        self.assertIn('CompetitionDistance', self.train_df_clean.columns, "'CompetitionDistance' column missing in the dataframe")
        self.assertIn('Sales', self.train_df_clean.columns, "'Sales' column missing in the dataframe")

        try:
            competitor_distance_effect(self.train_df_clean)
        except Exception as e:
            self.fail(f"competitor_distance_effect raised an exception: {e}")

    def test_customer_behavior_opening_closing(self):
        """Test customer behavior during store opening and closing."""
        # Check if the 'Hour' and 'Sales' columns are in the dataframe
        self.assertIn('Hour', self.train_df_clean.columns, "'Hour' column missing in the dataframe")
        self.assertIn('Sales', self.train_df_clean.columns, "'Sales' column missing in the dataframe")

        try:
            customer_behavior_opening_closing(self.train_df_clean)
        except Exception as e:
            self.fail(f"customer_behavior_opening_closing raised an exception: {e}")

if __name__ == '__main__':
    unittest.main()
