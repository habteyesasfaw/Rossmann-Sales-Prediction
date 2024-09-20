# eda_plots.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import logging

# Configure logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Function to compare promo distributions in training and test sets
def compare_promo_distribution(train_df, test_df):
    logger.info("Comparing promo distribution between train and test sets.")
    plt.figure(figsize=(10, 6))
    sns.histplot(train_df['Promo'], label='Train', color='blue', kde=False)
    sns.histplot(test_df['Promo'], label='Test', color='green', kde=False)
    plt.title('Promo Distribution in Train vs Test Sets')
    plt.xlabel('Promo')
    plt.legend()
    plt.show()

# Function to compare sales behavior during holidays
def sales_behavior_during_holidays(df):
    logger.info("Analyzing sales behavior during holidays.")
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='StateHoliday', y='Sales', data=df)
    plt.title('Sales Before, During, and After Holidays')
    plt.show()

# Function to plot seasonal sales behavior
def plot_seasonal_behavior(df):
    logger.info("Plotting seasonal purchase behavior.")
    df['Month'] = df['Date'].dt.month
    plt.figure(figsize=(14, 8))
    sns.lineplot(x='Month', y='Sales', data=df, estimator='mean')
    plt.title('Sales Trends Across Different Months')
    plt.show()

# Function to calculate correlation between sales and customers
def sales_customers_correlation(df):
    logger.info("Calculating correlation between sales and number of customers.")
    correlation = df['Sales'].corr(df['Customers'])
    print(f"Correlation between Sales and Customers: {correlation}")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Customers', y='Sales', data=df)
    plt.title('Sales vs. Number of Customers')
    plt.show()

# Function to check effect of promotions on sales and customers
def promo_effect_on_sales(df):
    logger.info("Analyzing the effect of promos on sales and customers.")
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Promo', y='Sales', data=df)
    plt.title('Sales with vs. without Promotions')
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Promo', y='Customers', data=df)
    plt.title('Number of Customers with vs. without Promotions')
    plt.show()

# Function to analyze promo effectiveness by store
def promo_effectiveness_by_store(df):
    logger.info("Analyzing promo effectiveness by store.")
    promo_sales = df.groupby('Store')['Sales'].mean().unstack()
    sns.heatmap(promo_sales, cmap="YlGnBu", annot=True)
    plt.title("Sales Effectiveness by Store During Promotions")
    plt.show()

# Function to analyze customer behavior around store opening/closing times
def customer_behavior_opening_closing(df):
    logger.info("Analyzing customer behavior around store opening and closing times.")
    df['Hour'] = df['Date'].dt.hour
    plt.figure(figsize=(12, 8))
    sns.lineplot(x='Hour', y='Sales', data=df, estimator='mean')
    plt.title('Customer Behavior Around Store Opening and Closing Times')
    plt.show()

# Function to analyze sales on weekends vs weekdays
def sales_weekend_vs_weekday(df):
    logger.info("Comparing sales on weekdays vs weekends.")
    weekend_sales = df[df['DayOfWeek'].isin([6, 7])]['Sales'].mean()
    weekday_sales = df[df['DayOfWeek'].isin([1, 2, 3, 4, 5])]['Sales'].mean()
    print(f"Weekend Sales: {weekend_sales}, Weekday Sales: {weekday_sales}")

# Function to analyze the effect of assortment types on sales
def assortment_type_sales(df):
    logger.info("Analyzing the effect of assortment types on sales.")
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Assortment', y='Sales', data=df)
    plt.title('Effect of Assortment Type on Sales')
    plt.show()

# Function to analyze the effect of competitor distance on sales
def competitor_distance_effect(df):
    logger.info("Analyzing the effect of competitor distance on sales.")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='CompetitionDistance', y='Sales', data=df)
    plt.title('Effect of Competitor Distance on Sales')
    plt.show()

# Function to analyze sales before and after new competitor opens
def new_competitor_effect(df):
    logger.info("Analyzing the effect of new competitors on store sales.")
    df['CompetitionOpen'] = df['CompetitionDistance'].notnull().astype(int)
    sns.lineplot(x='Date', y='Sales', hue='CompetitionOpen', data=df)
    plt.title('Sales Before and After Competitor Openings')
    plt.show()
