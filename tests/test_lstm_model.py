# tests/test_model.py
import unittest
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import adfuller
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Sample data for testing
sample_sales_data = pd.DataFrame({
    'Sales': [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
})

class TestModel(unittest.TestCase):

    # # 1. Test for stationarity check function
    # def check_stationarity(self, timeseries):
    #     result = adfuller(timeseries)
    #     return result[1] > 0.05  # Returns True if not stationary

    # def test_stationarity(self):
    #     self.assertFalse(self.check_stationarity(sample_sales_data['Sales']),
    #                      "Data should be stationary")

    # 2. Test data scaling function
    def test_scaling(self):
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled_data = scaler.fit_transform(sample_sales_data)
        self.assertTrue(np.min(scaled_data) >= -1 and np.max(scaled_data) <= 1,
                        "Data not in (-1, 1) range")

    # 3. Test LSTM Model Creation
    def test_lstm_model(self):
        model = Sequential()
        model.add(LSTM(50, input_shape=(30, 1)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        self.assertIsNotNone(model, "Model was not created successfully")


if __name__ == '__main__':
    unittest.main()
