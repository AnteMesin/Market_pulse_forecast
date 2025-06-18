# training/trainer.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def create_sequence_old(data, sequence_length):
    sequences, targets = [], []
    close_price_index = 4  # Explicitly set to column index of 'close_price'
    for i in range(len(data) - sequence_length):
        seq = data[i:i+sequence_length]
        target = data[i+sequence_length][close_price_index]
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets)

def create_sequences(X, y, sequence_length):
    sequences, targets = [], []
    for i in range(len(X) - sequence_length):
        sequences.append(X[i:i+sequence_length])
        targets.append(y[i+sequence_length])
    return np.array(sequences), np.array(targets)

def inverse_transform(scaler, values, column_index):
    dummy = np.zeros((len(values), scaler.n_features_in_))
    dummy[:, column_index] = values.ravel()  # Ensure shape matches
    return scaler.inverse_transform(dummy)[:, column_index]

def prepare_dataloaders_old(train_df, test_df, sequence_length, batch_size):
    train_values = pd.get_dummies(train_df.drop(columns=['Date', 'close_price'])).values.astype(np.float32)
    test_values = pd.get_dummies(test_df.drop(columns=['Date', 'close_price'])).values.astype(np.float32)


    features_used = train_df.drop(columns=['Date', 'close_price']).columns.tolist()
    print("🧠 Features used for training:", features_used, len(features_used))


    X_train, y_train = create_sequences(train_values, sequence_length)
    X_test, y_test = create_sequences(test_values, sequence_length)

    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train).unsqueeze(-1))
    test_ds = TensorDataset(torch.tensor(X_test), torch.tensor(y_test).unsqueeze(-1))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def prepare_dataloaders(train_df, test_df, sequence_length, batch_size):
    # Save close_price before dropping
    y_train_full = train_df['close_price'].values.astype(np.float32)
    y_test_full = test_df['close_price'].values.astype(np.float32)

    # Use features only (no target)
    X_train_full = pd.get_dummies(train_df.drop(columns=['Date', 'close_price'])).values.astype(np.float32)
    X_test_full = pd.get_dummies(test_df.drop(columns=['Date', 'close_price'])).values.astype(np.float32)

    # Create correct sequences
    X_train, y_train = create_sequences(X_train_full, y_train_full, sequence_length)
    X_test, y_test = create_sequences(X_test_full, y_test_full, sequence_length)

    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train).unsqueeze(-1))
    test_ds = TensorDataset(torch.tensor(X_test), torch.tensor(y_test).unsqueeze(-1))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def plot_predictions(y_true, y_pred, timestamps=None, title="Prediction vs Actual"):
    plt.figure(figsize=(10, 4))
    x_axis = timestamps if timestamps is not None else range(len(y_true))
    plt.plot(x_axis, y_true, label='Actual')
    plt.plot(x_axis, y_pred, label='Predicted')
    plt.title(title)
    plt.xlabel("Date" if timestamps is not None else "Time")
    plt.ylabel("Close Price")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def train_model(model, train_loader, val_loader, config, scaler):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['model']['learning_rate'])

    all_preds, all_targets = [], []

    for epoch in range(config['model']['epochs']):
        model.train()
        running_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device).float(), y_batch.to(device).float()

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if epoch == config['model']['epochs'] - 1:
                all_preds.append(outputs.detach().cpu().numpy())
                all_targets.append(y_batch.detach().cpu().numpy())

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{config['model']['epochs']}], Loss: {avg_loss:.4f}")

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    column_index = list(scaler.feature_names_in_).index("close_price")
    y_true_inv = inverse_transform(scaler, all_targets, column_index)
    y_pred_inv = inverse_transform(scaler, all_preds, column_index)
    plot_predictions(y_true_inv, y_pred_inv, title="Train Set Fitted values vs Actual")

    return model

def evaluate_model(model, test_loader, scaler):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    predictions, actuals = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device).float(), y_batch.to(device).float()
            outputs = model(X_batch)
            predictions.extend(outputs.squeeze().cpu().numpy())
            actuals.extend(y_batch.squeeze().cpu().numpy())

    # Inverse transform to original price scale
    column_index = list(scaler.feature_names_in_).index("close_price")
    y_true_inv = inverse_transform(scaler, np.array(actuals), column_index)
    y_pred_inv = inverse_transform(scaler, np.array(predictions), column_index)

    # ✅ Report real-world errors
    rmse = root_mean_squared_error(y_true_inv, y_pred_inv)
    mae = mean_absolute_error(y_true_inv, y_pred_inv)
    print(f"✅ Test RMSE (actual): {rmse:.4f}")
    print(f"✅ Test MAE  (actual): {mae:.4f}")

    plot_predictions(y_true_inv, y_pred_inv, title="Test Set: Prediction vs Actual")
    return rmse, mae
