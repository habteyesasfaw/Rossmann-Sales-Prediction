import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def prepare_data_for_lstm(df, target_column, time_steps=30):
    # Create supervised data
    X, y = [], []
    for i in range(time_steps, len(df)):
        X.append(df[i-time_steps:i])
        y.append(df[i, target_column])
    return np.array(X), np.array(y)

# Scale the data between -1 and 1
scaler = MinMaxScaler(feature_range=(-1, 1))

def scale_data(df, target_column):
    df_scaled = scaler.fit_transform(df)
    return df_scaled
