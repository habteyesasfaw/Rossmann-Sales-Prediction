import pandas as pd
from sklearn.preprocessing import StandardScaler

def handle_missing_values(df):
    df.fillna(df.median(numeric_only=True), inplace=True)
    df.fillna(df.mode().iloc[0], inplace=True)
    return df

def encode_categorical(df):
    df = pd.get_dummies(df, drop_first=True)
    return df

def extract_date_features(df, date_column):
    """
    Extract year, month, and day from a date column.
    """
    df[date_column] = pd.to_datetime(df[date_column])
    df['Year'] = df[date_column].dt.year
    df['Month'] = df[date_column].dt.month
    df['Day'] = df[date_column].dt.day
    return df

def feature_scaling(df):
    """
    Scale numeric features using StandardScaler.
    """
    # Select only numeric columns
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    scaler = StandardScaler()
    
    # Scale only numeric data
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    return df

def preprocess_data(df, date_column):
    """
    Preprocess the DataFrame: extract date features and scale numeric columns.
    """
    # Extract date features
    df = extract_date_features(df, date_column)
    
    # Perform feature scaling only on numeric columns
    df = feature_scaling(df)
    
    return df

def merge_store_data(train_df, test_df, store_df):
    train_merged = train_df.merge(store_df, on='Store', how='left')
    test_merged = test_df.merge(store_df, on='Store', how='left')
    train_merged.fillna(0, inplace=True)
    test_merged.fillna(0, inplace=True)
    return train_merged, test_merged

def preprocess_data(df, date_column):
    df = handle_missing_values(df)
    df = encode_categorical(df)
    df = extract_date_features(df, date_column)
    # df = feature_scaling(df)
    return df
