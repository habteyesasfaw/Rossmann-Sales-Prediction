import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from tensorflow.keras.utils import Sequence


class DataGenerator(Sequence):
    def __init__(self, data, target, time_steps, batch_size):
        self.data = data
        self.target = target
        self.time_steps = time_steps
        self.batch_size = batch_size

    def __len__(self):
        return int(np.floor(len(self.data) / self.batch_size))

    def __getitem__(self, idx):
        return self.__data_generation(idx)

    def __data_generation(self, idx):
        X, y = [], []
        for i in range(self.batch_size):
            start = idx * self.batch_size + i
            if start + self.time_steps < len(self.data):
                X.append(self.data[start:start + self.time_steps])
                y.append(self.target[start + self.time_steps])
        return np.array(X), np.array(y)
    
def check_stationarity(time_series):
    result = adfuller(time_series)
    return result[1] < 0.05

def difference_data(data):
    return data.diff().dropna()

def plot_acf_pacf(time_series, lags=40):
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plot_acf(time_series, lags=lags, ax=plt.gca())
    plt.title('Autocorrelation')
    plt.subplot(122)
    plot_pacf(time_series, lags=lags, ax=plt.gca())
    plt.title('Partial Autocorrelation')
    plt.tight_layout()
    plt.show()

def prepare_data_for_lstm(df, target_column, time_steps=30):
    X, y = [], []
    for i in range(time_steps, len(df)):
        X.append(df[i-time_steps:i])
        y.append(df[i, target_column])
    return np.array(X), np.array(y)

def scale_data(df):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    return scaler.fit_transform(df)

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def save_model(model, folder_path="models/"):
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    model_filename = f"{folder_path}/lstm_model_{timestamp}.h5"
    model.save(model_filename)



