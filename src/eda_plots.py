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
def sales_behavior_during_holidays(train_data):
    # Convert 'Date' column to datetime
    train_data['Date'] = pd.to_datetime(train_data['Date'])
    
    # Create 'HolidayFlag' where StateHoliday is not 0
    train_data['HolidayFlag'] = train_data['StateHoliday'].apply(lambda x: 1 if x != '0' else 0)
    
    # Shift dates to create 'BeforeHoliday' and 'AfterHoliday' flags
    train_data['BeforeHoliday'] = train_data['HolidayFlag'].shift(1, fill_value=0)
    train_data['AfterHoliday'] = train_data['HolidayFlag'].shift(-1, fill_value=0)
    
    # Define a function to categorize the period
    def categorize_period(row):
        if row['HolidayFlag'] == 1:
            return 'During Holiday'
        elif row['BeforeHoliday'] == 1:
            return 'Before Holiday'
        elif row['AfterHoliday'] == 1:
            return 'After Holiday'
        else:
            return 'Non-Holiday'
    
    # Apply categorization to create a new column
    train_data['HolidayPeriod'] = train_data.apply(categorize_period, axis=1)
    
    # Group by 'HolidayPeriod' and calculate average sales
    holiday_sales = train_data.groupby('HolidayPeriod')['Sales'].mean()
    
    # Plot the sales trends
    plt.figure(figsize=(10, 6))
    holiday_sales.plot(kind='bar', color=['gray', 'blue', 'green', 'orange'])
    plt.title('Sales Before, During, and After Holidays')
    plt.xlabel('Holiday Period')
    plt.ylabel('Average Sales')
    plt.show()

    # Log the completion of the plot
    logger.info("Generating line plot for sales behavior before, during, and after holidays.")



# Function to plot seasonal sales behavior

def plot_seasonal_behavior(df):
    logger.info("Plotting seasonal purchase behavior.")
    
    # Ensure 'Date' is in datetime format
    if not pd.api.types.is_datetime64_any_dtype(df['Date']):
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')  # Convert to datetime

    # Check for NaT values and drop if any
    if df['Date'].isnull().any():
        logger.warning("Some dates could not be converted and will be dropped.")
        df = df.dropna(subset=['Date'])

    # Extract month from the Date column
    df['Month'] = df['Date'].dt.month
    
    plt.figure(figsize=(14, 8))
    sns.lineplot(x='Month', y='Sales', data=df, estimator='mean')
    plt.title('Sales Trends Across Different Months', fontsize=16)
    plt.xlabel('Month', fontsize=14)
    plt.ylabel('Average Sales', fontsize=14)
    plt.xticks(range(1, 13))  # Set x-ticks for months
    plt.grid(True)
    plt.show()

# Function to calculate correlation between sales and customers
def sales_customers_correlation(df):
    logger.info("Calculating correlation between sales and number of customers.")
    
    # Calculate the correlation between Sales and Customers
    correlation = df['Sales'].corr(df['Customers'])
    print(f"Correlation between Sales and Customers: {correlation}")
    
    # Scatterplot for Sales vs. Customers
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Customers', y='Sales', data=df)
    plt.title('Sales vs. Number of Customers')
    plt.grid(True)
    plt.show()
    
    # Generate a correlation matrix for relevant columns (Sales and Customers)
    corr_matrix = df[['Sales', 'Customers']].corr()
    
    # Display the correlation matrix as a heatmap
    plt.figure(figsize=(6, 4))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5, fmt='.2f')
    plt.title('Correlation Heatmap: Sales and Customers')
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
    
    promo_sales = df[df['Promo'] == 1].groupby('Store')['Sales'].mean().reset_index()
    non_promo_sales = df[df['Promo'] == 0].groupby('Store')['Sales'].mean().reset_index()
    
    promo_sales.rename(columns={'Sales': 'Promo_Sales'}, inplace=True)
    non_promo_sales.rename(columns={'Sales': 'Non_Promo_Sales'}, inplace=True)
    
    sales_comparison = pd.merge(promo_sales, non_promo_sales, on='Store')
    sales_comparison['Promo_Effectiveness'] = sales_comparison['Promo_Sales'] - sales_comparison['Non_Promo_Sales']
    
    plt.figure(figsize=(14, 8))
    sns.barplot(x='Store', y='Promo_Effectiveness', data=sales_comparison, palette='coolwarm', edgecolor='black')
    plt.title('Promo Effectiveness by Store', fontsize=18)
    plt.xlabel('Store', fontsize=14)
    plt.ylabel('Sales Difference (Promo vs Non-Promo)', fontsize=14)
    plt.xticks(rotation=90)

    plt.tight_layout()
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
