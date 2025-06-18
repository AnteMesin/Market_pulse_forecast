## 🎯 Hyperparameter Tuning & Evaluation

The model undergoes a **random search** to identify the best-performing hyperparameters. Each trial experiments with a different combination of:

- `lstm_units`: [32, 64, 128]
- `lstm_layers`: [1, 2, 3]
- `dropout`: [0.1, 0.2, 0.3]
- `learning_rate`: [0.001, 0.0005, 0.0001]

Each configuration is evaluated using **Root Mean Square Error (RMSE)** as the performance metric.

### 🔁 Split Strategies Used

We assess each configuration under different **data splitting strategies** to ensure robust performance:

1. **Chronological Split** – Traditional 80/20 time split.
2. **Expanding Window** – Growing train set simulates accumulating historical data.
3. **Rolling Window** – Fixed-size window slides forward, useful for concept drift.
4. **Time Series K-Fold** – Multiple rolling folds to validate on all segments.

### 🧪 Objective

- Identify the best combination of hyperparameters and data split strategy that **generalizes well**.
- Ensure stability across different **time-based splits**, avoiding data leakage.

### ✅ Output

After `n_trials`, the tuning script returns:

- The **lowest RMSE** achieved
- The **best hyperparameter configuration**
- This guides final training and helps explain **why the selected model is optimal**

### 💡 Next Step

You can visualize the results by:
- Plotting RMSE per trial
- Comparing performance across different split strategies
- Logging results for each fold/split combination
