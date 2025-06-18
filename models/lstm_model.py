# models/lstm_model.py
import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size=1, dropout=0.2, use_attention=False):
        super(LSTMModel, self).__init__()
        self.use_attention = use_attention
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        
        if use_attention:
            self.attn = nn.Linear(hidden_size, 1)  # attention weights
        
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # lstm_out: [batch, seq_len, hidden_size]

        if self.use_attention:
            # Apply attention over time steps
            attn_weights = torch.softmax(self.attn(lstm_out), dim=1)  # [batch, seq_len, 1]
            context = torch.sum(attn_weights * lstm_out, dim=1)       # [batch, hidden_size]
        else:
            # Use only the last hidden state
            context = lstm_out[:, -1, :]  # [batch, hidden_size]

        out = self.fc(context)  # [batch, output_size]
        return out

if __name__ == "__main__":
    model = LSTMModel(input_size=10, hidden_size=64, num_layers=2, use_attention=True)
    dummy_input = torch.randn(8, 30, 10)  # (batch_size, seq_len, input_size)
    output = model(dummy_input)
    print(output.shape)  # Expected: (8, 1)
