import pandas as pd

# Load sentiment data
df = pd.read_csv('data\\msft_stock_news_sentiment.csv')
df['publishedDate'] = pd.to_datetime(df['publishedDate']).dt.date
df['publishedDate'] = pd.to_datetime(df['publishedDate'])

# Load stock data
msft_data = pd.read_csv('data\\msft_data.csv')
msft_data['Date'] = pd.to_datetime(msft_data['Date'])

# Step 1: Aggregate sentiment per day (average, can also use sum, etc.)
# Replace 'sentiment_score' with your actual sentiment column
daily_sentiment = df.groupby('publishedDate', as_index=False).mean(numeric_only=True)

# Step 2: Merge sentiment into stock data
merged_data = pd.merge(msft_data, daily_sentiment, left_on='Date', right_on='publishedDate', how='left')

# Step 3: Add weekday and month info
merged_data['day_of_week'] = merged_data['Date'].dt.day_name()
merged_data['month_of_year'] = merged_data['Date'].dt.month_name()

# Step 4 (Optional): Fill missing sentiment with 0 if you prefer
# merged_data['sentiment_score'] = merged_data['sentiment_score'].fillna(0)

# Save final cleaned dataset
merged_data.to_csv('data\\merged_data.csv', index=False)
