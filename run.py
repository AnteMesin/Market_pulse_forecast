import random
import argparse
import yaml
import torch
import numpy as np
import os
import ta
from datetime import datetime
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
from protocol_logger import log_checkpoints

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

def inverse_transform(scaler, data, column_index):
    """
    Inverse transform a single column from a multi-column MinMaxScaler.
    """
    dummy = np.zeros((len(data), scaler.n_features_in_))
    dummy[:, column_index] = data
    return scaler.inverse_transform(dummy)[:, column_index]


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

def main(mode, method, config, enable_hash_logging=False):
    df_raw = load_merged_data("data\\merged_data.csv")
    df, scaler = preprocess_data(df_raw)
    df["close_lag1"] = df["close_price"].shift(1)
    df["close_lag2"] = df["close_price"].shift(2)
    df["momentum"] = df["close_price"] - df["close_lag2"]
    if "macd" in df.columns and "macd_signal" in df.columns:
        df["macd_diff"] = df["macd"] - df["macd_signal"]
    df["rolling_std_14"] = df["close_price"].rolling(14).std()
    df = df.dropna().reset_index(drop=True)
    df = select_features(df, config["feature_flags"])
    train_df, test_df = get_split(df, config)
    seq_len = config["tuning"]["sequence_length"]
    X_train, y_train = prepare_data(seq_len, train_df, target_col="close_price")
    X_test, y_test = prepare_data(seq_len, test_df, target_col="close_price")
    train_loader, test_loader = prepare_dataloaders(train_df, test_df, seq_len, config["model"]["batch_size"])
    actual_input_size = next(iter(train_loader))[0].shape[-1]
    model = LSTMModel(
        input_size=actual_input_size,
        hidden_size=config["model"]["lstm_units"],
        num_layers=config["model"]["lstm_layers"],
        dropout=config["model"]["dropout"],
        use_attention=config["explainability"]["use_attention"]
    )

    history = None
    eval_metrics = None
    predictions_df = None
    shap_summary = None
    attention_weights = None
    attention_image = None
    tuning_result_dict = None

    if mode == "data-prep":
        print("🚀 Running data preparation pipeline...")
        from data_prep import run_data_preparation
        run_data_preparation()

    elif mode == "train":
        history = train_model(model, train_loader, None, config, scaler)
        os.makedirs("artifacts", exist_ok=True)
        torch.save(model.state_dict(), "artifacts/best_model.pt")

    elif mode == "evaluate":
        model.load_state_dict(torch.load("artifacts/best_model.pt"))
        rmse, mae = evaluate_model(model, test_loader, scaler)
        eval_metrics = {"rmse": float(rmse), "mae": float(mae)}

        # Regenerate predictions manually for final_predictions_hash
        model.eval()
        predictions, actuals = [], []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device).float(), y_batch.to(device).float()
                outputs = model(X_batch)
                predictions.extend(outputs.squeeze().cpu().numpy())
                actuals.extend(y_batch.squeeze().cpu().numpy())

        col_index = list(scaler.feature_names_in_).index("close_price")
        y_true_inv = inverse_transform(scaler, np.array(actuals), col_index)
        y_pred_inv = inverse_transform(scaler, np.array(predictions), col_index)

        import pandas as pd
        predictions_df = pd.DataFrame({"actual": y_true_inv, "predicted": y_pred_inv})

    elif mode == "tune":
        from tuning.search import run_tuning_across_splits
        tuning_result_dict = run_tuning_across_splits(df, config, scaler)

    elif mode == "explain":
        model.load_state_dict(torch.load("artifacts/best_model.pt"))
        model.eval()
        feature_columns = df.drop(columns=["Date", "close_price"]).columns.tolist()
        if method == "shap":
            shap_summary = explain_with_shap(model, X_test.astype(np.float32), df_columns=feature_columns)
        elif method == "lime":
            explain_with_lime(model, X_train, X_test, feature_names=feature_columns)
        elif method == "saliency":
            input_tensor = torch.tensor(X_test[:1], dtype=torch.float32)
            attention_weights = compute_saliency(model, input_tensor, feature_columns)
        elif method == "counterfactual":
            attention_image = plot_counterfactual_sensitivity(model, X_test[0], delta=0.1, feature_names=feature_columns)
        else:
            raise ValueError("Unknown explanation method")

    if enable_hash_logging:
        checkpoint_data = {
            "config_settings": config,
            "feature_flags_hash": config.get("feature_flags"),
            "split_strategy_hash": config.get("split"),
            "train_dataset_hash": train_df.to_json(),
            "val_dataset_hash": test_df.to_json(),
            "tuned_hyperparams_hash": tuning_result_dict,
            "model_weights_hash": {k: v.tolist() for k, v in model.state_dict().items()},
            "training_metrics_hash": history,
            "evaluation_metrics_hash": eval_metrics,
            "attention_weights_hash": attention_weights,
            "shap_explanation_hash": shap_summary.to_json() if shap_summary is not None else None,
            "attention_plot_hash": attention_image.getvalue() if attention_image else None,
            "final_predictions_hash": predictions_df.to_json() if predictions_df is not None else None
        }
        log_checkpoints({k: v for k, v in checkpoint_data.items() if v is not None})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["data-prep","train", "evaluate", "explain", "tune"], required=True)
    parser.add_argument("--method", default="shap", help="explanation method: shap/lime/saliency/counterfactual")
    parser.add_argument("--split_method", choices=["chronological", "expanding", "rolling", "kfold"], default="chronological")
    parser.add_argument("--enable_hash_logging", action="store_true", help="Enable checkpoint hash logging to Sepolia")

    args = parser.parse_args()
    with open("config\\settings.yml", "r") as f:
        full_config = yaml.safe_load(f)
    full_config["split"] = full_config["splits"][args.split_method]
    print(f"🔀 Using split method: {args.split_method}")
    main(args.mode, args.method, full_config, enable_hash_logging=args.enable_hash_logging)
