import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv('/content/merged_data.csv')


# Convert date column to datetime
df['publishedDate'] = pd.to_datetime(df['publishedDate'])

# Group by 'publishedDate' and aggregate relevant columns
daily_df = df.groupby('publishedDate').agg({
    'sentiment_score': 'mean',  # Average sentiment score for the day
    'open_price': 'first',  # Opening stock price of the day
    'high_price': 'max',  # Highest stock price of the day
    'low_price': 'min',  # Lowest stock price of the day
    'close_price': 'last',  # Closing stock price of the day
    'volume': 'sum',  # Total trading volume for the day
    'price_change': 'mean',  # Average price change for the day
    'moving_avg_50': 'mean',  # 50-day moving average
    'moving_avg_200': 'mean',  # 200-day moving average
    'rsi_14': 'mean',  # Relative Strength Index (14-day)
    'ema_12': 'mean',  # 12-day Exponential Moving Average
    'ema_26': 'mean',  # 26-day Exponential Moving Average
    'macd': 'mean',  # Moving Average Convergence Divergence
    'macd_signal': 'mean',  # MACD Signal Line
    'bollinger_middle': 'mean',  # Bollinger Bands Middle Line
    'bollinger_std': 'mean',  # Bollinger Bands Standard Deviation
    'bollinger_upper': 'mean',  # Bollinger Bands Upper Band
    'bollinger_lower': 'mean',  # Bollinger Bands Lower Band
    'high_low': 'mean',  # High-Low price difference
    'high_close': 'mean',  # High-Close price difference
    'low_close': 'mean',  # Low-Close price difference
    'tr': 'mean',  # True Range
    'atr': 'mean',  # Average True Range
    'day_of_week': 'first',  # Day of the week (as is, no encoding required)
    'month_of_year': 'first'  # Month of the year (as is, no encoding required)
}).reset_index()

# Select only numeric columns (exclude 'publishedDate', 'day_of_week', and 'month_of_year')
numeric_columns = daily_df.select_dtypes(include=[np.number]).columns

# Normalize only the numeric columns
scaler = MinMaxScaler()
daily_df[numeric_columns] = scaler.fit_transform(daily_df[numeric_columns])

# Create sequences for LSTM
sequence_length = 30  # Use past 30 days to predict next day
X, y = [], []
for i in range(len(daily_df) - sequence_length):
    X.append(daily_df.iloc[i:i+sequence_length, 1:].values)  # Features
    y.append(daily_df.iloc[i+sequence_length]['close_price'])  # Target: next day's closing price

X, y = np.array(X), np.array(y)

# Split data
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, shuffle=False)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=False)

# Save processed data
np.save("X_train.npy", X_train)
np.save("y_train.npy", y_train)
np.save("X_val.npy", X_val)
np.save("y_val.npy", y_val)
np.save("X_test.npy", X_test)
np.save("y_test.npy", y_test)

print("Data prepared for LSTM: Sequences created and saved!")
