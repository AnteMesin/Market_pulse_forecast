# explainability/counterfactuals.py
import torch
import numpy as np
import plotly.express as px


def plot_counterfactual_sensitivity(model, instance, delta=0.1, feature_names=None):
    instance = instance.copy()
    original_tensor = torch.tensor(instance[np.newaxis, :, :], dtype=torch.float32)

    with torch.no_grad():
        base_pred = model(original_tensor).item()

    sensitivity = []

    for i in range(instance.shape[1]):
        modified = instance.copy()
        modified[:, i] += delta

        modified_tensor = torch.tensor(modified[np.newaxis, :, :], dtype=torch.float32)

        with torch.no_grad():
            new_pred = model(modified_tensor).item()

        change = abs(new_pred - base_pred)
        name = feature_names[i] if feature_names else f"Feature {i}"
        sensitivity.append((name, change))

    # Sort and convert to DataFrame
    sensitivity = sorted(sensitivity, key=lambda x: x[1], reverse=True)
    names, diffs = zip(*sensitivity)

    # Plotly bar chart
    fig = px.bar(
        x=diffs,
        y=names,
        orientation='h',
        labels={"x": "Prediction Change", "y": "Feature"},
        title=f"Counterfactual Sensitivity (Δ = {delta})",
        height=600
    )
    fig.update_layout(yaxis=dict(autorange="reversed"))
    fig.show()
