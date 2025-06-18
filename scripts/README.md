## Microsoft (MSFT) Data Collection & Processing

We collected Microsoft (MSFT) data from various sources, including news articles, stock prices, and technical indicators, to enhance our market analysis. The dataset integrates sentiment analysis, price movements, and macroeconomic factors like interest rates and oil prices using Financial Modeling Prep’s API.

### Order of Execution:

1. **News Data Collection**  
   We gathered news articles related to MSFT and performed sentiment analysis to derive sentiment scores. The sentiment scores represent the general market sentiment towards MSFT based on news coverage. This data serves as a feature to better understand market behavior and forecast stock movements. (folder: **news**)

2. **Stock Prices & Technical Indicators**  
   Price data for MSFT was extracted, along with key technical indicators such as MACD, Bollinger Bands, RSI, and moving averages. These indicators provide insights into the stock’s trend, momentum, and volatility, enhancing our market analysis and prediction capabilities. (folder: **price_tech_indicators**)

3. **Data Merging**  
   All collected data, including sentiment scores, stock prices, and technical indicators, were merged into a unified dataset. This integration allows for a comprehensive view of MSFT’s market activity, consolidating all relevant factors to aid in forecasting stock price movements and understanding market trends. (folder: **merge**)

4. **Data Preprocessing, Sequence Creation & Splitting for LSTM**  
   The dataset is loaded, and the `publishedDate` is converted to a `datetime` format. The data is then aggregated by date, calculating average sentiment scores and stock price statistics. Numeric columns are normalized for uniform scaling. Sequences of 30 consecutive days are created as input features (X) for the LSTM model, with the target variable (y) being the closing price for the next day. The data is split into training, validation, and test sets while preserving the chronological order, ensuring realistic model evaluation. The prepared datasets are then saved for model training.
 (folder: **data_prep**)

---

The final dataset offers a comprehensive view of MSFT’s market trends, combining historical stock price data, sentiment analysis, and technical indicators to aid in better-informed investment decisions. 🚀

