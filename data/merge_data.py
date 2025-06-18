# merge_data.py
import pandas as pd

def merge_stock_and_sentiment(stock_path, sentiment_path, output_path="data\\merged_data.csv"):
    # Load stock data
    stock_df = pd.read_csv(stock_path)
    stock_df["Date"] = pd.to_datetime(stock_df["Date"])

    # Load sentiment data
    sentiment_df = pd.read_csv(sentiment_path)
    sentiment_df["date"] = pd.to_datetime(sentiment_df["publishedDate"]).dt.date

    # Aggregate sentiment per day
    sentiment_agg = sentiment_df.groupby("date").agg({
        "sentiment_score": "mean",
        "news_count_last_24h": "sum",
        "average_sentiment_last_24h": "mean",
        "news_volatility_impact": "mean"
    }).reset_index()
    sentiment_agg["date"] = pd.to_datetime(sentiment_agg["date"])

    # Merge on date
    merged = pd.merge(stock_df, sentiment_agg, how="left", left_on="Date", right_on="date")
    merged.drop(columns=["date"], inplace=True)

    # Fill missing sentiment data
    sentiment_cols = [
        "sentiment_score",
        "news_count_last_24h",
        "average_sentiment_last_24h",
        "news_volatility_impact"
    ]
    merged[sentiment_cols] = merged[sentiment_cols].ffill().fillna(0)

    # Fill technical indicators (like moving averages, RSI, etc.)
    merged = merged.ffill().bfill()

    # Save merged file
    merged.to_csv(output_path, index=False)
    print(f"✅ Merged data saved to: {output_path} ({len(merged)} rows)")


if __name__ == "__main__":
    merge_stock_and_sentiment("data\\msft_data.csv", 'data\\msft_stock_news_sentiment.csv')
