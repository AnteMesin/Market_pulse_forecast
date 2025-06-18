import yfinance as yf
import pandas as pd

# Fetch historical stock price data using Yahoo Finance
def get_stock_data(symbol, start_date='2019-01-01'):
    df = yf.download(symbol, start=start_date)

    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df = df.rename(columns={
        'Open': 'open_price',
        'High': 'high_price',
        'Low': 'low_price',
        'Close': 'close_price',
        'Volume': 'volume'
    })
    df['price_change'] = ((df['close_price'] - df['open_price']) / df['open_price']) * 100
    df = df.sort_index(ascending=True)
    return df

# Compute technical indicators using pandas
def add_technical_indicators(df):
    # Moving Averages
    df['moving_avg_50'] = df['close_price'].rolling(window=50).mean()
    df['moving_avg_200'] = df['close_price'].rolling(window=200).mean()

    # Relative Strength Index (RSI)
    delta = df['close_price'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))

    # MACD (Moving Average Convergence Divergence)
    df['ema_12'] = df['close_price'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['close_price'].ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

    # Bollinger Bands
    df['bollinger_middle'] = df['close_price'].rolling(window=20).mean()
    df['bollinger_std'] = df['close_price'].rolling(window=20).std()
    df['bollinger_upper'] = df['bollinger_middle'] + (df['bollinger_std'] * 2)
    df['bollinger_lower'] = df['bollinger_middle'] - (df['bollinger_std'] * 2)

    # ATR (Average True Range)
    df['high_low'] = df['high_price'] - df['low_price']
    df['high_close'] = abs(df['high_price'] - df['close_price'].shift())
    df['low_close'] = abs(df['low_price'] - df['close_price'].shift())
    df['tr'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
    df['atr'] = df['tr'].rolling(window=14).mean()

    return df

# Fetch and process data

symbol = "MSFT"
msft_data = get_stock_data(symbol)
msft_data = add_technical_indicators(msft_data)

msft_data.columns = msft_data.columns.get_level_values(0)

# Reset index to ensure 'Date' is a column
msft_data.reset_index(inplace=True)

# Drop unnecessary index columns if present
msft_data = msft_data.drop(columns=['level_0', 'index'], errors='ignore')

# Convert 'Date' to datetime format
msft_data['Date'] = pd.to_datetime(msft_data['Date'])






# Further processing

# Flatten the MultiIndex columns by taking only the first level

# Save msft_data as csv file

