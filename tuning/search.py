# tuning/search.py
import random
from copy import deepcopy
from training.trainer import train_model, evaluate_model, prepare_dataloaders
from models.lstm_model import LSTMModel
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.close("all")

def random_search(train_df, test_df, config, scaler, n_trials=5, sequence_length=30):
    best_score = float('inf')
    best_config = None

    for i in range(n_trials):
        trial_config = deepcopy(config)
        trial_config['model']['lstm_units'] = random.choice([32, 64, 128, 256])
        trial_config['model']['lstm_layers'] = random.choice([1, 2, 3, 4])
        trial_config['model']['dropout'] = random.choice([0.1, 0.2, 0.3, 0.4])
        trial_config['model']['learning_rate'] = random.choice([0.001, 0.0005, 0.0001])

        print(f"\n🔍 Trial {i+1}: {trial_config['model']}")

        # Safeguard: skip if test set too small
        if len(test_df) <= sequence_length + 1:
            print("⚠️ Skipping trial due to insufficient test data.")
            continue

        train_loader, test_loader = prepare_dataloaders(
            train_df, test_df, sequence_length, batch_size=trial_config['model']['batch_size']
        )

        if len(train_loader) == 0 or len(test_loader) == 0:
            print("⚠️ Skipping trial due to empty dataloaders.")
            continue

        input_size = train_df.drop(columns=["Date", "close_price"]).shape[1]

        model = LSTMModel(
            input_size=input_size,
            hidden_size=trial_config['model']['lstm_units'],
            num_layers=trial_config['model']['lstm_layers'],
            dropout=trial_config['model']['dropout'],
            use_attention=trial_config['explainability']['use_attention']
        )

        # ❌ prevent plots from opening during tuning
        plt.ioff()
        model = train_model(model, train_loader, test_loader, trial_config, scaler)
        rmse, _ = evaluate_model(model, test_loader, scaler)
        plt.close('all')

        if rmse < best_score:
            best_score = rmse
            best_config = deepcopy(trial_config['model'])

    if best_config:
        print(f"\n✅ Best RMSE: {best_score:.4f} with config: {best_config}")
    else:
        print("❌ No valid trial succeeded.")
    return best_config


def run_tuning_across_splits(df, config, scaler):
    from splits.splitter import (
        chronological_split, expanding_window_split,
        rolling_window_split, time_series_kfold_split
    )

    all_best_configs = {}

    for method in ["chronological", "expanding", "rolling", "kfold"]:
        print(f"\n🧪 Running tuning with split method: {method}")
        split_config = config["splits"][method]
        config["split"] = split_config  # Override active split

        try:
            if method == "chronological":
                train_df, test_df = chronological_split(df, split_config["test_size"])
            elif method == "expanding":
                train_df, test_df = expanding_window_split(df, split_config["window_size"], test_size=split_config.get("test_size", 1))[-1]
            elif method == "rolling":
                train_df, test_df = rolling_window_split(df, split_config["window_size"], test_size=split_config.get("test_size", 1))[-1]
            elif method == "kfold":
                train_df, test_df = time_series_kfold_split(df, split_config["n_splits"])[-1]
            else:
                continue

            best_model_config = random_search(
                train_df, test_df,
                config=config,
                scaler=scaler,
                n_trials=config["tuning"]["n_trials"],
                sequence_length=config["tuning"]["sequence_length"]
            )

            if best_model_config:
                all_best_configs[method] = best_model_config

        except Exception as e:
            print(f"⚠️ Skipping {method} due to error: {e}")
            continue

    print("\n📊 Summary of Best Configs by Split Method:")
    for method, cfg in all_best_configs.items():
        print(f"- {method}: {cfg}")
