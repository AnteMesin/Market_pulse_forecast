# Data Preparation for LSTM Model

After data collection, we need to prepare the data for an LSTM (Long Short-Term Memory) model, which is designed to predict the stock closing prices based on historical data and sentiment scores. 

The data preparation steps include:

1. **Loading and Preprocessing the Dataset**  
   The dataset is loaded, and the `publishedDate` column is converted to a datetime object for proper time-series analysis.

2. **Data Aggregation**  
   Next, the data is aggregated by date to calculate daily statistics for stock prices and sentiment scores:
   
   - **sentiment_score**: The average sentiment score for the day.
   - **open**: The opening stock price.
   - **high**: The highest stock price during the day.
   - **low**: The lowest stock price during the day.
   - **close**: The closing stock price.
   - **volume**: The total trading volume for the day.

3. **Normalization**  
   To ensure that the features are on a similar scale, we normalize the data using the `MinMaxScaler` from `sklearn`. This scales all features (except the date column) to a range of [0, 1].

4. **Sequence Creation for LSTM**  
   To train the LSTM model, we create sequences of 30 consecutive days (a window of 30 days) to predict the next day's closing price. The features (X) are the stock prices and sentiment scores for the past 30 days, and the target (y) is the closing price for the next day.

   **How LSTM Works and Why It Needs Sequences:**

   LSTM models are a type of recurrent neural network (RNN) designed to handle sequential data, making them well-suited for time-series problems like stock price prediction. Traditional feedforward neural networks do not have the capacity to remember previous inputs, while LSTMs are specifically designed to retain information over long periods. They are able to "remember" past events (like previous stock prices or sentiment scores) and use that memory to make predictions about future events.

   **Why Sequences Are Needed:**

   LSTMs need sequences of data because they rely on past information to predict future outcomes. In this case, by feeding the model a sequence of stock prices and sentiment scores from the past 30 days, the model can learn patterns and trends that will help it predict the stock's closing price for the next day. The LSTM works by processing one time step at a time, storing important information in its internal state (memory) to help make predictions. Without sequences, the model would not have the historical context needed to make accurate predictions.

   **Summary:**  
   The process involves creating sequences that include data from the previous 30 days, allowing the model to learn patterns in stock prices and sentiment scores over time to predict future closing prices.

5. **Data Splitting into Training, Validation, and Test Sets**  
   The data is split into training, validation, and test sets while preserving the temporal order of the data.

6. **Saving the Processed Data**  
   The processed data (X_train, y_train, X_val, y_val, X_test, and y_test) is saved as `.npy` files for easy loading during model training.



## Issues

### 1. Why 193 rows? how many unique days?

Further investigations shows there are missing date in merged_data.

<img width="199" alt="image" src="https://github.com/user-attachments/assets/2e9ff068-5af6-4456-8d03-3d7e6f506c8f" />

- Possible reason: Need to check if **inner join caused data loss while preparing merged data.**
- Resolution: check the python script in the folder merge and fix the code.

