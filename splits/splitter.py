import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

def chronological_split(df, test_size=0.2):
    split_idx = int(len(df) * (1 - test_size))
    return df.iloc[:split_idx], df.iloc[split_idx:]

def expanding_window_split(df, window_size, test_size=30):
    splits = []
    for i in range(len(df) - window_size - test_size + 1):
        train_df = df[: window_size + i]
        test_df = df[window_size + i : window_size + i + test_size]
        splits.append((train_df, test_df))
    return splits

def rolling_window_split(df, window_size, test_size=1):
    splits = []
    for i in range(len(df) - window_size - test_size + 1):
        train_df = df[i : i + window_size]
        test_df = df[i + window_size : i + window_size + test_size]
        splits.append((train_df, test_df))
    return splits

def time_series_kfold_split(df, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    splits = []
    for train_idx, test_idx in tscv.split(df):
        train = df.iloc[train_idx]
        test = df.iloc[test_idx]
        splits.append((train, test))
    return splits

if __name__ == "__main__":
    from data.loader import load_merged_data, preprocess_data
    from data.features import select_features

    df = preprocess_data(load_merged_data("data/merged_data.csv"))
    df = select_features(df, {
        "use_price": True,
        "use_technicals": True,
        "use_sentiment": True,
        "use_calendar": True
    })

    train, test = chronological_split(df)
    print("Train size:", len(train))
    print("Test size:", len(test))
