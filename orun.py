# run.py
import random
import argparse
import yaml
import torch
import numpy as np
import os
import ta
from data.loader import load_merged_data, preprocess_data
from data.features import select_features
from splits.splitter import (
    chronological_split, expanding_window_split,
    rolling_window_split, time_series_kfold_split
)
from models.lstm_model import LSTMModel
from training.trainer import train_model, evaluate_model, prepare_dataloaders
from explainability.shap_explainer import explain_with_shap
from explainability.lime_explainer import explain_with_lime
from explainability.saliency import compute_saliency
from explainability.counterfactuals import plot_counterfactual_sensitivity

import yaml
from data_prep import run_data_preparation


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensures reproducibility in CUDA convolution operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)  # ✅ Set the seed right after imports

def load_config(path="config\\settings.yml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def get_split(df, config):
    method = config["split"]["method"]

    if method == "chronological":
        return chronological_split(df, config["split"]["test_size"])

    elif method == "expanding":
        test_size = config["split"].get("test_size", 1)
        return expanding_window_split(df, config["split"]["window_size"], test_size=test_size)[-1]

    elif method == "rolling":
        # Use test_size from config if provided, default to 1 for backward compatibility
        test_size = config["split"].get("test_size", 1)
        return rolling_window_split(df, config["split"]["window_size"], test_size=test_size)[-1]

    elif method == "kfold":
        return time_series_kfold_split(df, config["split"]["n_splits"])[-1]

    else:
        raise ValueError("Unsupported split method")


def prepare_data(sequence_length, df, target_col="close_price"):
    df_X = df.drop(columns=["Date", target_col])
    y = df[target_col].values

    X, targets = [], []
    for i in range(len(df_X) - sequence_length):
        X.append(df_X.iloc[i:i+sequence_length].values)
        targets.append(y[i+sequence_length])
    return np.array(X), np.array(targets)

def main(mode, method, config):
    #with open("config\\settings.yml", "r") as f:
    #    full_config = yaml.safe_load(f)

    #config = full_config

    df_raw = load_merged_data("data\\merged_data.csv")
    df, scaler = preprocess_data(df_raw)

    # ✅ Step 1: Add lag features and momentum
    df["close_lag1"] = df["close_price"].shift(1)
    df["close_lag2"] = df["close_price"].shift(2)
    df["momentum"] = df["close_price"] - df["close_lag2"]

    # ✅ Step 2: Add MACD diff
    if "macd" in df.columns and "macd_signal" in df.columns:
        df["macd_diff"] = df["macd"] - df["macd_signal"]

    # ✅ Step 3: Add rolling std (volatility)
    df["rolling_std_14"] = df["close_price"].rolling(14).std()


    # ✅ Step 4: Drop rows with NaNs from lagging/rolling
    df = df.dropna().reset_index(drop=True)


    df = select_features(df, config["feature_flags"])
    train_df, test_df = get_split(df, config)

    seq_len = config["tuning"]["sequence_length"]
    X_train, y_train = prepare_data(seq_len, train_df, target_col="close_price")
    X_test, y_test = prepare_data(seq_len, test_df, target_col="close_price")

    train_loader, _ = prepare_dataloaders(train_df, test_df, seq_len, config["model"]["batch_size"])
    actual_input_size = next(iter(train_loader))[0].shape[-1]

    model = LSTMModel(
        input_size=actual_input_size,
        hidden_size=config["model"]["lstm_units"],
        num_layers=config["model"]["lstm_layers"],
        dropout=config["model"]["dropout"],
        use_attention=config["explainability"]["use_attention"]
    )

    if mode == "data-prep":
        print("🚀 Running data preparation pipeline...")
        run_data_preparation()  # This function should exist inside data_prep.py

    elif mode == "train":
        train_loader, _ = prepare_dataloaders(train_df, test_df, seq_len, config["model"]["batch_size"])
        train_model(model, train_loader, None, config, scaler)  # ⛔ removed timestamps
        os.makedirs("artifacts", exist_ok=True)
        torch.save(model.state_dict(), "artifacts/best_model.pt")

    elif mode == "evaluate":
        model.load_state_dict(torch.load("artifacts/best_model.pt"))
        _, test_loader = prepare_dataloaders(train_df, test_df, seq_len, config["model"]["batch_size"])
        evaluate_model(model, test_loader, scaler)  # ⛔ removed timestamps

    elif mode == "tune":
        from tuning.search import run_tuning_across_splits
        run_tuning_across_splits(df, config, scaler)

    elif mode == "explain":
        model.load_state_dict(torch.load("artifacts/best_model.pt"))
        model.eval()
        if method == "shap":
            feature_columns = df.drop(columns=["Date", "close_price"]).columns.tolist()
            explain_with_shap(model, X_test.astype(np.float32), df_columns=feature_columns)

        elif method == "lime":
            feature_columns = df.drop(columns=["Date", "close_price"]).columns.tolist()
            explain_with_lime(model, X_train, X_test, feature_names=feature_columns)

        elif method == "saliency":
            input_tensor = torch.tensor(X_test[:1], dtype=torch.float32)
            feature_columns = df.drop(columns=["Date", "close_price"]).columns.tolist()
            compute_saliency(model, input_tensor, feature_columns)

        elif method == "counterfactual":
            feature_columns = df.drop(columns=["Date", "close_price"]).columns.tolist()
            plot_counterfactual_sensitivity(model, X_test[0], delta=0.1, feature_names=feature_columns)

        else:
            raise ValueError("Unknown explanation method")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["data-prep","train", "evaluate", "explain", "tune"], required=True)
    parser.add_argument("--method", default="shap", help="explanation method: shap/lime/saliency/counterfactual")
    parser.add_argument("--split_method", choices=["chronological", "expanding", "rolling", "kfold"], default="chronological")

    args = parser.parse_args()

    with open("config\\settings.yml", "r") as f:
        full_config = yaml.safe_load(f)

    # Inject chosen split config dynamically
    full_config["split"] = full_config["splits"][args.split_method]
    print(f"🔀 Using split method: {args.split_method}")

    main(args.mode, args.method, full_config)
