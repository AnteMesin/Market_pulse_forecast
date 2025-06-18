# explainability/shap_explainer.py
import shap
import torch
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd

def explain_with_shap(model, X_sample, df_columns):
    model.eval()



    # Remove 'close_price' from features (index 3)
    df_columns = [col for col in df_columns if col not in ("Date", "close_price")]

     # ✅ Ensure dimensions are correct
    assert X_sample.ndim == 3, "X_sample must be [batch, seq_len, features]"

    expected_input_size = model.lstm.input_size
    actual_input_size = X_sample.shape[2]

    if actual_input_size > expected_input_size:
        diff = actual_input_size - expected_input_size
        X_sample = X_sample[:, :, :expected_input_size]
        df_columns = df_columns[:expected_input_size]

    elif actual_input_size < expected_input_size:
        raise ValueError(f"❌ Not enough features: expected {expected_input_size}, got {actual_input_size}")

    # Flatten 3D input → 2D by averaging over time steps
    X_flat = X_sample.mean(axis=1)

    def model_predict(x_flat):
        # Repeat the flattened input over time steps
        x_seq = np.repeat(x_flat[:, np.newaxis, :], X_sample.shape[1], axis=1)
        x_tensor = torch.tensor(x_seq, dtype=torch.float32)
        with torch.no_grad():
            preds = model(x_tensor).detach().numpy()
        return preds

    # Smart proportional splitting for small datasets
    n_total = X_flat.shape[0]

    if n_total < 6:
        raise ValueError("❌ Not enough data for SHAP explanation (need at least 6 samples).")

    # 70% background, 30% test (or at least 1 row)
    n_background = max(int(n_total * 0.7), 1)
    n_test = max(n_total - n_background, 1)

    background = X_flat[:n_background]
    test_sample = X_flat[n_background : n_background + n_test]


    # Use KernelExplainer for LSTM
    explainer = shap.KernelExplainer(model_predict, background[:20])
    shap_values = explainer.shap_values(test_sample)


    print("SHAP values per feature:", len(shap_values[0]))
#    print("Feature names:", len(feature_names))
#    print(feature_names)

    # SHAP summary plot
    # shap.summary_plot(shap_values, features=test_sample, feature_names=feature_names, show=True)


    # Convert SHAP values to numpy and squeeze if needed
    shap_array = np.array(shap_values).squeeze()  # Shape: (5, 27)
    mean_abs_shap = np.mean(np.abs(shap_array), axis=0)  # Shape: (27,)

    # ✅ Build DataFrame for plot
    shap_df = pd.DataFrame({
        "Feature": df_columns,
        "Mean |SHAP Value|": mean_abs_shap
    }).sort_values(by="Mean |SHAP Value|", ascending=False)

    # Interactive Plot
    fig = px.bar(
        shap_df,
        x="Mean |SHAP Value|",
        y="Feature",
        orientation="h",
        title="SHAP Feature Importance (Interactive)",
        height=600
    )

    fig.update_layout(yaxis=dict(autorange="reversed"))
    fig.show()

    threshold = 0.0002
    important_features = shap_df[shap_df["Mean |SHAP Value|"] > threshold]["Feature"].tolist()

    print('Keep these features:', important_features)