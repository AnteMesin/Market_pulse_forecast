# data/features.py
import pandas as pd

def print_data_shape_summary(df, stage, initial_cols=None):
    print(f"[{stage}] → Rows: {len(df)}, Columns: {len(df.columns)}")
    if initial_cols:
        dropped = set(initial_cols) - set(df.columns)
        if dropped:
            print(f"⚠️ Dropped columns: {list(dropped)}")

def handle_missing_values(df):
    """
    Fill missing values using appropriate strategies.
    """
    # Forward fill and backfill numeric columns
    df = df.ffill().bfill()

    # Fill sentiment-related columns with neutral values
    sentiment_cols = [
        "sentiment_score", "average_sentiment_last_24h",
        "news_volatility_impact", "news_count_last_24h"
    ]
    for col in sentiment_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    return df

def select_features(df, config):
    """
    Select features strictly based on 'feature_flags' in the config.

    - Ignores group-level keys like 'use_price', 'use_technicals', etc.
    - Only includes features explicitly marked as `true` in 'feature_flags'
    - Automatically adds 'day_of_week' and 'month_of_year' if enabled
    - Always keeps 'Date' and 'close_price'
    """
    print_data_shape_summary(df, "Before feature selection")

    selected = ['Date', 'close_price']  # Always include
    feature_flags = config


    # Add calendar features if explicitly enabled
    if feature_flags.get("day_of_week", False):
        df["day_of_week"] = pd.to_datetime(df["Date"]).dt.dayofweek
    if feature_flags.get("month_of_year", False):
        df["month_of_year"] = pd.to_datetime(df["Date"]).dt.month

    # Select only features explicitly marked as true

    print(feature_flags.items())

    for feature, enabled in feature_flags.items():
        if enabled and feature in df.columns:
            selected.append(feature)


    initial_cols = list(df.columns)
    df_selected = df[selected].copy()
    df_selected = handle_missing_values(df_selected)

    print_data_shape_summary(df_selected, "After missing value handling", initial_cols=initial_cols)
    return df_selected


if __name__ == "__main__":
    import yaml
    from loader import load_merged_data, preprocess_data

    # ✅ Load from settings.yaml
    with open("config\\settings.yml", "r") as f:
        full_config = yaml.safe_load(f)

    config = full_config
    

    # Load and preprocess
    raw_df = load_merged_data("data\\merged_data.csv")
    df_scaled, _ = preprocess_data(raw_df)

    # Feature selection based on group and flags
    df_selected = select_features(df_scaled, config)

