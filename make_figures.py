"""Generate PNG figures from the reproduction CSVs in results/."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


COLORS = {"MLP": "#d62728", "RandomForest": "#1f77b4", "HistGB": "#2ca02c"}
MARKERS = {"MLP": "o", "RandomForest": "s", "HistGB": "^"}


def load(csv: str) -> pd.DataFrame:
    return pd.read_csv(Path("results") / csv)


def style(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)


def plot_remove_features():
    df = load("remove_features.csv")
    pivot = df.pivot(index="n_features_kept", columns="model", values="accuracy")
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for model in ["MLP", "RandomForest", "HistGB"]:
        ax.plot(
            pivot.index,
            pivot[model],
            color=COLORS[model],
            marker=MARKERS[model],
            label=model,
        )
    ax.invert_xaxis()
    style(ax, "Number of most-important features kept", "Test accuracy", "Remove least important features")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(Path("figures/remove_features.png"), dpi=200)
    plt.close(fig)


def plot_add_features():
    df = load("add_features.csv")
    pivot = df.pivot(index="n_added", columns="model", values="accuracy")
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for model in ["MLP", "RandomForest", "HistGB"]:
        ax.plot(
            pivot.index,
            pivot[model],
            color=COLORS[model],
            marker=MARKERS[model],
            label=model,
        )
    style(ax, "Number of Gaussian noise features added", "Test accuracy", "Add uninformative features")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(Path("figures/add_features.png"), dpi=200)
    plt.close(fig)


def plot_rotations():
    df = load("rotations.csv")
    summary = df.groupby(["condition", "model"])["accuracy"].agg(["mean", "std"]).reset_index()
    conditions = ["original", "rotated"]
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    x = range(len(conditions))
    width = 0.26
    for i, model in enumerate(["MLP", "RandomForest", "HistGB"]):
        sub = summary[summary.model == model].set_index("condition").loc[conditions]
        ax.bar(
            [xi + (i - 1) * width for xi in x],
            sub["mean"],
            yerr=sub["std"],
            width=width,
            color=COLORS[model],
            label=model,
            capsize=3,
        )
    ax.set_xticks(list(x))
    ax.set_xticklabels(["Original\nfeatures", "Randomly\nrotated"])
    style(ax, None, "Test accuracy", "Rotation invariance")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(Path("figures/rotations.png"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    Path("figures").mkdir(parents=True, exist_ok=True)
    plot_remove_features()
    plot_add_features()
    plot_rotations()
    print("figures written to figures/")
