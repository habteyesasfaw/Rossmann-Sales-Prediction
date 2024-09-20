import pandas as pd
import numpy as np
from src.logger import get_logger

logger = get_logger(__name__)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Starting data cleaning process.")
    
    # Handle missing values
    df['CompetitionDistance'].fillna(df['CompetitionDistance'].mean(), inplace=True)
    df['Promo2SinceWeek'].fillna(0, inplace=True)
    
    # Handle outliers
    df = df[df['Sales'] > 0]  # Removing zero sales
    df = df[df['Customers'] > 0]  # Removing zero customers
    
    # Convert date to datetime format
    df['Date'] = pd.to_datetime(df['Date'])
    
    logger.info("Data cleaning completed successfully.")
    return df
