from pathlib import Path

import matplotlib.pyplot as plt
import shap


def plot_shap(model,X_background,target_name,output_dir,include_bar=True):
    """This function generates SHAP summary plots for a fitted model."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    explainer = shap.Explainer(model, X_background)
    shap_values = explainer(X_background, check_additivity=False)

    plt.figure(figsize=(10, 7))
    shap.plots.beeswarm(shap_values, show=False)
    plt.title(f"SHAP Summary: {target_name}")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/shap_beeswarm_{target_name}.png",dpi=300,bbox_inches="tight")
    plt.close()

    if include_bar:
        plt.figure(figsize=(10, 7))
        shap.plots.bar(shap_values, show=False)
        plt.title(f"SHAP Feature Importance: {target_name}")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/shap_bar_{target_name}.png",dpi=300,bbox_inches="tight")
        plt.close()

    return shap_values