## 🏋️ Training Module (`trainer.py`)

This module handles the full training and evaluation workflow for the LSTM model.

It is responsible for:
- Preparing time series sequences
- Wrapping data into PyTorch DataLoaders
- Training the LSTM model
- Evaluating predictions using metrics like RMSE and MAE

---

### 📂 How It Fits Into the Project

```text
data/loader.py           → Load & clean dataset
data/features.py         → Select + fill feature columns
splits/splitter.py       → Create train/test splits
models/lstm_model.py     → Define LSTM model
training/trainer.py      → Train and evaluate model ✅
```

### 🔁 Components

1. create_sequences()

Splits continuous time series data into input sequences and targets:

```python
X, y = create_sequences(data, sequence_length=30)
```
Each input X has shape [seq_len, num_features], and y is the target value right after the sequence.

2. prepare_dataloaders()

- Converts Pandas DataFrames to PyTorch TensorDatasets
- Removes the Date column
- Batches data using DataLoader

```python
train_loader, test_loader = prepare_dataloaders(train_df, test_df, sequence_length=30, batch_size=32)
```

3. train_model()

- Standard LSTM training loop using Adam optimizer
- Prints epoch-wise loss

```python
trained_model = train_model(model, train_loader, val_loader, config)
```

4. evaluate_model()

- Makes predictions on test data
- Computes RMSE and MAE

```python
rmse, mae = evaluate_model(model, test_loader)
```



### 🧠 How It Uses the LSTM Model

The LSTM model is defined in:

```python
from models.lstm_model import LSTMModel
```

After preparing your input shape, you instantiate it with:

```python
model = LSTMModel(
    input_size=num_features,
    hidden_size=64,
    num_layers=2,
    use_attention=True
)
```

Then you pass this model into train_model() and evaluate_model().


### ✅ Output

Once training completes, you get printed logs like:

```text
Epoch [1/50], Loss: 0.0134
...
Test RMSE: 1.5027
Test MAE : 1.2451
```

