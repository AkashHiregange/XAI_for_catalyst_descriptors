import matplotlib.pyplot as plt
from pathlib import Path


def plot_performance(results, output_dir):
    """Plot train/test R2 and MAE for each target."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for model_name, model_results in results.groupby("model"):
        for metric in ("r2", "mae"):
            pivot = model_results.pivot(index="target",columns="split",values=metric)

            ax = pivot.plot(kind="bar",figsize=(12, 6),width=0.8)

            ax.set_title(f"{model_name}: {metric.upper()}")
            ax.set_xlabel("Target")
            ax.set_ylabel(metric.upper())
            ax.tick_params(axis="x", rotation=45)
            ax.legend(title="Split")

            if metric == "r2":
                ax.axhline(0.9,linestyle="--",linewidth=1,color="gray")

            ax.figure.tight_layout()
            ax.figure.savefig(f"{output_dir}/{model_name}_{metric}.png",dpi=300,bbox_inches="tight")
            plt.close(ax.figure)