import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def handle_missing_values(df):
    # Fill missing numeric values with the median and categorical with mode
    df.fillna(df.median(numeric_only=True), inplace=True)
    df.fillna(df.mode().iloc[0], inplace=True)
    return df

def encode_categorical(df):
    # Convert categorical columns using one-hot encoding or label encoding
    df = pd.get_dummies(df, drop_first=True)
    return df

def extract_date_features(df, date_column):
    df[date_column] = pd.to_datetime(df[date_column])
    df['Year'] = df[date_column].dt.year
    df['Month'] = df[date_column].dt.month
    df['Week'] = df[date_column].dt.isocalendar().week
    df['DayOfWeek'] = df[date_column].dt.dayofweek
    df['Weekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
    df['MonthStart'] = df[date_column].dt.is_month_start
    df['MonthEnd'] = df[date_column].dt.is_month_end
    return df

def feature_scaling(df):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    return pd.DataFrame(scaled_data, columns=df.columns)

# Combine all preprocessing steps
def preprocess_data(df, date_column):
    df = handle_missing_values(df)
    df = encode_categorical(df)
    df = extract_date_features(df, date_column)
    df = feature_scaling(df)
    return df
