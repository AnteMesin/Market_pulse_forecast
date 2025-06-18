# Key Features to Collect

## 1. News Features
- **sentiment_score**: Sentiment polarity score of the news article (TextBlob/VADER).  
  Helps assess the emotional tone of news articles (positive, negative, or neutral).
  
- **sentiment_label**: Categorical sentiment (Positive, Neutral, Negative).  
  Converts sentiment score into categorical labels to classify sentiment.

- **news_count_last_24h**: Number of news articles in the last 24 hours for a stock.  
  Provides insight into how much attention the stock is receiving, which may correlate with price movements.

- **average_sentiment_last_24h**: Average sentiment score of news in the last 24 hours.  
  Captures the overall market mood over a short time period.

- **news_volatility_impact**: Sentiment standard deviation (volatility in news).  
  Measures fluctuations in sentiment, which may correlate with stock volatility.


## 2. Stock Price Data
- **open_price**: Opening stock price.  
  Shows the initial price at the beginning of the trading day.

- **close_price**: Closing stock price.  
  Represents the final price at the end of the trading day.

- **high_price**: Highest stock price of the day.  
  Reflects the peak price during the trading day.

- **low_price**: Lowest stock price of the day.  
  Reflects the lowest price during the trading day.

- **volume**: Trading volume (important for liquidity and market movement).  
  Indicates how much of the stock is being traded and helps assess market sentiment.

- **price_change**: Percentage change in stock price ((close - open) / open * 100).  
  Measures the daily price movement, useful for identifying trends.


## 3. Technical Indicators
- **moving_avg_50**: 50-day moving average (trend indicator).  
  A medium-term trend indicator that helps smooth out price data.

- **moving_avg_200**: 200-day moving average (long-term trend).  
  A long-term trend indicator widely used in stock analysis.

- **rsi_14**: Relative Strength Index (momentum indicator).  
  Helps identify overbought or oversold conditions, signaling potential reversals.

- **macd_signal**: MACD signal line (trend strength).  
  Measures the difference between a short-term and long-term moving average, used to identify momentum.

- **bollinger_upper**: Upper Bollinger Band (volatility measure).  
  Helps assess whether a stock is overbought and indicates price volatility.

- **bollinger_lower**: Lower Bollinger Band.  
  Indicates whether a stock is oversold and provides insights into market volatility.

- **atr**: Average True Range (volatility measure).  
  Measures market volatility and potential price movement.


## 4. **Time-based Features**
- **day_of_week**: Day of the week (Monday, Tuesday, etc.).  
  Stocks often exhibit different behavior on weekdays, with Mondays typically having lower volume.

- **month_of_year**: Month of the year (January, February, etc.).  
  Seasonality can influence stock performance (e.g., end-of-year rallies, tax season effects).


## Conclusion - feature selections

By combining **sentiment analysis**, **stock price data**, **technical indicators**, we can create a robust model that captures complex relationships and patterns in stock price movements.


## 🧼 Handling Missing Values

In real-world financial and news datasets, missing values are common due to:
- Rolling technical indicators (e.g., moving averages, RSI)
- Gaps in sentiment coverage (no news on certain days)
- Lagging computations (e.g., MACD, Bollinger bands)

To avoid losing valuable time series data, we do **not drop rows**. Instead, we apply a feature-aware missing value strategy:

### 🛠 Strategy

| Feature Type            | Imputation Strategy        | Rationale                                  |
|-------------------------|----------------------------|--------------------------------------------|
| Price/Volume            | Forward + Backward Fill    | Price trends are continuous in time        |
| Technical Indicators    | Forward + Backward Fill    | Lagging features can be interpolated       |
| Sentiment Scores        | Fill with `0`              | Missing sentiment implies neutral impact   |
| News Volume/Impact      | Fill with `0`              | No news = no influence                     |
| Calendar Features       | Always present             | Extracted directly from date               |

### 🧪 Example Implementation

```python
# Forward fill & backfill all numeric values
df = df.ffill().bfill()

# Neutral fill for sentiment-related features
for col in ['sentiment_score', 'average_sentiment_last_24h',
            'news_volatility_impact', 'news_count_last_24h']:
    if col in df.columns:
        df[col] = df[col].fillna(0)
```


✅ Result
- All rows are preserved
- No data leakage from future timestamps
- Model remains robust and interpretable

## 🧩 Module Descriptions

### `loader.py`
Loads the merged dataset containing historical stock prices and sentiment features.  
Also includes a basic preprocessing function to sort by date and remove rows without essential price data (e.g., missing close prices).

### `features.py`
Adds support for configurable feature selection and robust handling of missing values.  
Includes forward/backward filling for technical features and neutral value filling for sentiment data, ensuring no row is dropped.







