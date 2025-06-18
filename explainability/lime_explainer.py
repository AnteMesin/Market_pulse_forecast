import torch
from lime import lime_tabular
import numpy as np
import pandas as pd
import plotly.express as px
import webbrowser


def explain_with_lime(model, X_train, X_sample, feature_names):
    # Flatten sequences for LIME
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_sample_flat = X_sample.reshape(X_sample.shape[0], -1)

    time_steps = X_train.shape[1]
    all_feature_names = [f"{name}_t{t}" for t in range(time_steps) for name in feature_names]

    # Create LIME explainer
    explainer = lime_tabular.LimeTabularExplainer(
        X_train_flat,
        mode="regression",
        feature_names=all_feature_names
    )

    # Explain the first sample
    i = 0
    exp = explainer.explain_instance(
    X_sample_flat[i],
    lambda x: model(torch.tensor(x.reshape(-1, X_sample.shape[1], X_sample.shape[2]), dtype=torch.float32)).detach().numpy().flatten(),
    num_features=50  # <-- add this to get more features
    )

    # Save explanation to HTML
    html_path = "explainability\\lime_explanation.html"
    exp.save_to_file(html_path)
    webbrowser.open(html_path)
    print(f"✅ LIME explanation saved to: {html_path}")

    # Extract LIME weights
    lime_weights = exp.as_list()
    lime_df = pd.DataFrame(lime_weights, columns=["Feature", "Importance"])

    # ✅ Clean and aggregate by base feature
    lime_df["BaseFeature"] = lime_df["Feature"].str.extract(r"([a-zA-Z0-9_]+)_t\d+")
    agg_df = lime_df.groupby("BaseFeature", as_index=False)["Importance"].mean()
    agg_df = agg_df.sort_values(by="Importance", ascending=False)

    print("\n📊 Aggregated LIME Feature Importances:")
    print(agg_df.head(10))

    # ✅ Plot
    fig = px.bar(
        agg_df.head(20),
        x="Importance",
        y="BaseFeature",
        orientation="h",
        title="Aggregated LIME Feature Importance (Top 20)",
        height=600
    )
    fig.update_layout(yaxis=dict(autorange="reversed"))
    fig.show()

    return agg_df
