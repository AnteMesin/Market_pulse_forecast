# data/loader.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

filepath = "data\\merged_data.csv" # merged file path



def load_merged_data(filepath):
    """
    Load the merged stock and sentiment data.
    """
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def preprocess_data(df):
    df["day_of_week"] = df["Date"].dt.dayofweek
    df["month_of_year"] = df["Date"].dt.month

    df = df.ffill().bfill()

    scaler = MinMaxScaler()
    features_to_scale = df.drop(columns=["Date"])
    scaled_values = scaler.fit_transform(features_to_scale)

    df_scaled = pd.DataFrame(scaled_values, columns=features_to_scale.columns)
    df_scaled["Date"] = df["Date"].values  # restore original date

    return df_scaled, scaler

if __name__ == "__main__":
    data = load_merged_data("data/merged_data.csv")
    df, scaler, timestamps = preprocess_data(data)
    print(df.head())
    print(df.columns)