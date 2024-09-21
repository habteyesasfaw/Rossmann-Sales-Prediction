import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import joblib

def check_stationarity(time_series):
    result = adfuller(time_series)
    return result[1]  # Return the p-value

def difference_data(df, column, lag=1):
    return df[column].diff(lag).dropna()

def create_supervised_data(df, target_column, time_steps=30):
    X, y = [], []
    for i in range(time_steps, len(df)):
        X.append(df[i - time_steps:i])
        y.append(df[i, target_column])
    return np.array(X), np.array(y)

def scale_data(df, target_column):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_data = scaler.fit_transform(df)
    joblib.dump(scaler, 'models/scaler.pkl')  # Save the scaler
    return scaled_data

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def save_lstm_model(model, folder_path="models/"):
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    model_filename = f"{folder_path}/lstm_model_{timestamp}.h5"
    model.save(model_filename)

def fit_model(model, X_train, y_train, epochs=10, batch_size=32):
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

def plot_predictions(y_true, y_pred):
    plt.figure(figsize=(10, 6))
    plt.plot(y_true, label="True")
    plt.plot(y_pred, label="Predicted")
    plt.legend()
    plt.show()
