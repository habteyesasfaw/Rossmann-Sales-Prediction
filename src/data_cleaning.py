# src/data_cleaning.py

import pandas as pd
import numpy as np
import logging

# Configure logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

def clean_data(train_df, test_df, store_df):
    logger.info("Starting data cleaning process...")

    # Merge store data with train and test sets
    train_df = train_df.merge(store_df, on='Store', how='left')
    test_df = test_df.merge(store_df, on='Store', how='left')

    # Handle missing values
    logger.info("Handling missing values...")
    train_df.fillna({
        'CompetitionDistance': train_df['CompetitionDistance'].median(),
        'CompetitionOpenSinceMonth': train_df['CompetitionOpenSinceMonth'].mode()[0],
        'CompetitionOpenSinceYear': train_df['CompetitionOpenSinceYear'].mode()[0],
        'Promo2SinceWeek': train_df['Promo2SinceWeek'].mode()[0],
        'Promo2SinceYear': train_df['Promo2SinceYear'].mode()[0],
        'PromoInterval': train_df['PromoInterval'].mode()[0]
    }, inplace=True)

    test_df.fillna({
        'CompetitionDistance': test_df['CompetitionDistance'].median(),
        'CompetitionOpenSinceMonth': test_df['CompetitionOpenSinceMonth'].mode()[0],
        'CompetitionOpenSinceYear': test_df['CompetitionOpenSinceYear'].mode()[0],
        'Promo2SinceWeek': test_df['Promo2SinceWeek'].mode()[0],
        'Promo2SinceYear': test_df['Promo2SinceYear'].mode()[0],
        'PromoInterval': test_df['PromoInterval'].mode()[0]
    }, inplace=True)

    # Convert date columns to datetime format
    logger.info("Converting date columns...")
    train_df['Date'] = pd.to_datetime(train_df['Date'])
    test_df['Date'] = pd.to_datetime(test_df['Date'])

    # Remove duplicates
    logger.info("Removing duplicates...")
    train_df.drop_duplicates(inplace=True)
    test_df.drop_duplicates(inplace=True)

    # Handle outliers
    logger.info("Handling outliers...")
    sales_cap = train_df['Sales'].quantile(0.99)
    customer_cap = train_df['Customers'].quantile(0.99)
    train_df['Sales'] = np.where(train_df['Sales'] > sales_cap, sales_cap, train_df['Sales'])
    train_df['Customers'] = np.where(train_df['Customers'] > customer_cap, customer_cap, train_df['Customers'])

    logger.info("Data cleaning completed successfully.")
    
    return train_df, test_df
