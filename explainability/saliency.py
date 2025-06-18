# explainability/saliency.py
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def compute_saliency(model, input_tensor, feature_names):
    input_tensor.requires_grad_()
    output = model(input_tensor)
    output.backward()

    # ➕ Saliency shape: (sequence_length, num_features)
    saliency = input_tensor.grad.abs().squeeze().detach().numpy()

    if saliency.ndim == 1:  # if batch size was 1
        saliency = saliency[np.newaxis, :]

    if saliency.ndim == 2:
        time_steps, num_features = saliency.shape
    else:
        raise ValueError(f"Expected saliency of shape [time, features], got {saliency.shape}")

    # ✅ Create heatmap
    plt.figure(figsize=(12, 6))
    sns.heatmap(saliency.T, cmap="YlOrRd", xticklabels=5, yticklabels=feature_names)
    plt.title("Saliency Heatmap: Feature Importance Over Time Steps")
    plt.xlabel("Time Step")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()

    return saliency
