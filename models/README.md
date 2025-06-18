## 🧠 LSTM Model (`lstm_model.py`)

This module defines a configurable LSTM-based architecture for stock price forecasting.

### 📌 Key Features

- Built using **PyTorch**
- Supports **optional attention mechanism** to focus on important time steps
- Accepts dynamic `input_size`, `hidden_size`, `num_layers`, and `dropout` values
- Outputs a **single value per sequence** (e.g., next day's price)

### 🧱 Architecture

```text
Input → LSTM (stacked) → [Attention or Final Time Step] → Fully Connected → Output

- If use_attention=True, attention weights are learned to weight each time step
- If use_attention=False, only the final LSTM output is used
```

### 🔧 Class: LSTMModel

```python
LSTMModel(
    input_size: int,
    hidden_size: int,
    num_layers: int,
    output_size: int = 1,
    dropout: float = 0.2,
    use_attention: bool = False
)
```

### 🧪 Example

```python
from models.lstm_model import LSTMModel
import torch

model = LSTMModel(input_size=10, hidden_size=64, num_layers=2, use_attention=True)
dummy_input = torch.randn(8, 30, 10)  # (batch_size, seq_len, input_size)
output = model(dummy_input)          # Output shape: (8, 1)
```
### 🧠 Use Case

This model is ideal for time series forecasting tasks like:

- Predicting stock prices
- Forecasting market indicators
- Learning temporal dependencies in financial data