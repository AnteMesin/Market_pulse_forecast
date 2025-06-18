import requests
import json
import datetime
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
import re
import pandas as pd
import yfinance as yf
import yaml

def collect_news_sentiment_with_rolling(ticker="MSFT", start_date="2015-01-01", end_date=None):
    # === PART 1: Download News ===

    # API key and base URL
    api_key = "xxxxxxH7G6x2nowAxxxxxxxxkr0T"
    base_url = "https://financialmodelingprep.com/api/v3/stock_news"

    # Set the tickers
    tickers = ticker

    if end_date is None:
        end_date = datetime.datetime.today().strftime("%Y-%m-%d")

    # Function to fetch stock news for a given month and year
    def fetch_stock_news(year, month):
        start_date_local = f"{year}-{month:02d}-01"
        end_date_local = f"{year + 1}-01-01" if month == 12 else f"{year}-{month + 1:02d}-01"
        url = f"{base_url}?tickers={tickers}&page=1&from={start_date_local}&to={end_date_local}&apikey={api_key}"
        response = requests.get(url)
        return response.json() if response.status_code == 200 else None

    # List to store all news data
    news_collection = []

    # Use provided date range
    start_year = pd.to_datetime(start_date).year
    end_year = pd.to_datetime(end_date).year if end_date else datetime.datetime.now().year

    # Loop through each year/month
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            current = datetime.date(year, month, 1)
            if current < pd.to_datetime(start_date).date() or (end_date and current > pd.to_datetime(end_date).date()):
                continue
            print(f"Fetching news for {year}-{month:02d}")
            news_data = fetch_stock_news(year, month)
            if news_data:
                news_collection.extend(news_data)

    # Save raw JSON
    raw_file = f"data/{ticker.lower()}_stock_news.json"
    with open(raw_file, "w") as file:
        json.dump(news_collection, file, indent=4)
    print(f"News collection complete. Data saved to '{raw_file}'.")

    # === PART 2: Sentiment Analysis ===

    # Download necessary NLTK data (only needed once)
    nltk.download('vader_lexicon')
    nltk.download('stopwords')

    # Initialize VADER sentiment analyzer
    analyzer = SentimentIntensityAnalyzer()

    # Function to clean text and remove stop words
    def clean_text(text):
        if isinstance(text, str):
            text = re.sub(r'[^a-zA-Z\s]', '', text, re.I | re.A)
            text = text.lower()
            stop_words = set(stopwords.words('english'))
            words = text.split()
            words = [w for w in words if not w in stop_words]
            return " ".join(words)
        else:
            return ""

    # Open and load JSON file
    with open(raw_file, "r", encoding="utf-8") as file:
        try:
            data = json.load(file)
        except json.JSONDecodeError as e:
            print(f"Error loading JSON: {e}")
            data = []

    df = pd.DataFrame(data)

    # Apply the cleaning function to the 'text' column
    df['cleaned_text'] = df['text'].apply(clean_text)

    # Assign sentiment labels based on sentiment score
    def assign_sentiment_label(score):
        if score >= 0.05:
            return "positive"
        elif score <= -0.05:
            return "negative"
        else:
            return "neutral"

    # Calculate sentiment score
    def get_sentiment_score(text):
        scores = analyzer.polarity_scores(text)
        return scores['compound']

    df['sentiment_score'] = df['cleaned_text'].apply(get_sentiment_score)
    df['sentiment_label'] = df['sentiment_score'].apply(assign_sentiment_label)

    # === PART 3: Rolling Statistics ===
    df['publishedDate'] = pd.to_datetime(df['publishedDate'])
    df = df.sort_values(['symbol', 'publishedDate'])
    df.set_index('publishedDate', inplace=True)

    df['news_count_last_24h'] = (
        df.groupby('symbol')['title']
        .rolling('24h').count()
        .reset_index(level=0, drop=True)
    )

    df['average_sentiment_last_24h'] = (
        df.groupby('symbol')['sentiment_score']
        .rolling('24h').mean()
        .reset_index(level=0, drop=True)
    )

    df['news_volatility_impact'] = (
        df.groupby('symbol')['sentiment_score']
        .rolling('24h').std()
        .reset_index(level=0, drop=True)
    )

    df.reset_index(inplace=True)

    output_path = f"{ticker.lower()}_stock_news_sentiment.csv"
    #df.to_csv(output_path, index=False)
    print(f"✅ Sentiment-enriched data saved to: {output_path}")
    return df

# Example usage:

# collect_news_sentiment_with_rolling("MSFT", "2019-01-01", "2020-12-31")

def collect_stock_data(ticker="MSFT", start_date="2019-01-01", end_date=None):
    # === Fetch historical stock price data using Yahoo Finance ===
    df = yf.download(ticker, start=start_date, end=end_date)

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

    # === Compute technical indicators using pandas ===
    df['moving_avg_50'] = df['close_price'].rolling(window=50).mean()
    df['moving_avg_200'] = df['close_price'].rolling(window=200).mean()

    delta = df['close_price'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))

    df['ema_12'] = df['close_price'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['close_price'].ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

    df['bollinger_middle'] = df['close_price'].rolling(window=20).mean()
    df['bollinger_std'] = df['close_price'].rolling(window=20).std()
    df['bollinger_upper'] = df['bollinger_middle'] + (df['bollinger_std'] * 2)
    df['bollinger_lower'] = df['bollinger_middle'] - (df['bollinger_std'] * 2)

    df['high_low'] = df['high_price'] - df['low_price']
    df['high_close'] = abs(df['high_price'] - df['close_price'].shift())
    df['low_close'] = abs(df['low_price'] - df['close_price'].shift())
    df['tr'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
    df['atr'] = df['tr'].rolling(window=14).mean()

    # === Format output ===
    df.columns = df.columns.get_level_values(0) if isinstance(df.columns, pd.MultiIndex) else df.columns
    df.reset_index(inplace=True)
    df = df.drop(columns=['level_0', 'index'], errors='ignore')
    df['Date'] = pd.to_datetime(df['Date'])

    return df

# Example usage:
# df = collect_stock_data("MSFT", "2020-01-01", "2023-12-31")
# Load sentiment data




def run_data_preparation ():
    # Load YAML config
    with open("config/settings.yml", "r") as f:
        config = yaml.safe_load(f)

    # Access values
    ticker = config.get("ticker")
    start_date = config.get("start_date")
    end_date = config.get("end_date")

    df = collect_news_sentiment_with_rolling(ticker= ticker ,  start_date=start_date, end_date=end_date)

    df['publishedDate'] = pd.to_datetime(df['publishedDate']).dt.date
    df['publishedDate'] = pd.to_datetime(df['publishedDate'])

    # Load stock data
    stock_data = collect_stock_data(ticker= ticker ,  start_date=start_date, end_date=end_date)
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])

    # Step 1: Aggregate sentiment per day (average, can also use sum, etc.)
    # Replace 'sentiment_score' with your actual sentiment column
    daily_sentiment = df.groupby('publishedDate', as_index=False).mean(numeric_only=True)

    # Step 2: Merge sentiment into stock data
    merged_data = pd.merge(stock_data, daily_sentiment, left_on='Date', right_on='publishedDate', how='left')


    # Step 4 (Optional): Fill missing sentiment with 0 if you prefer
    # merged_data['sentiment_score'] = merged_data['sentiment_score'].fillna(0)

    # Save final cleaned dataset


    with open("data/merged_schema.json", "r") as f:
        schema = json.load(f)

    expected_columns = schema["columns"]
    expected_dtypes = schema["dtypes"]

    # Add missing columns
    for col in expected_columns:
        if col not in merged_data.columns:
            merged_data[col] = pd.NA

    # Drop extra columns
    merged_data = merged_data[[col for col in expected_columns if col in merged_data.columns]]

    # Reorder
    merged_data = merged_data[expected_columns]

    # Match data types
    for col, dtype in expected_dtypes.items():
        try:
            merged_data[col] = merged_data[col].astype(dtype)
        except:
            pass  # Skip type casting error

    # Save cleaned version
    merged_data.to_csv("data/merged_data.csv", index=False)
