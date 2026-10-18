"""
Quick data checks and exploratory plots.
"""

import matplotlib.pyplot as plt

import config


def check_data(df, dataset):
    print("Bananas per dosage (raw readings):")
    print(df.groupby(config.DOSAGE_COL)[config.GROUP_COL].nunique().to_string())
    print("\nBananas per dosage that reached spoilage (usable for training):")
    print(dataset.groupby(config.DOSAGE_COL)[config.GROUP_COL].nunique().to_string())
    print(f"\nTraining rows: {len(dataset)}")
    print(f"days_remaining range: {dataset[config.TARGET].min()} - {dataset[config.TARGET].max()}")
    print(f"colour_value range (should be 1-7): {df[config.COLOUR_COL].min()} - {df[config.COLOUR_COL].max()}\n")


def plot_trends(df, dataset, save_path=None):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for dosage, sub in df.groupby(config.DOSAGE_COL):
        axes[0].plot(*zip(*sub.groupby(config.DAY_COL)[config.ETHYLENE_COL].mean().items()),
                     marker="o", label=f"{dosage} g")
        axes[1].plot(*zip(*sub.groupby(config.DAY_COL)[config.COLOUR_COL].mean().items()),
                     marker="o", label=f"{dosage} g")
    axes[0].set(title="Mean ethylene over time", xlabel="Day", ylabel="Ethylene (ppm)")
    axes[1].set(title="Mean peel colour stage over time", xlabel="Day", ylabel="Colour (1-7)")

    for dosage, sub in dataset.groupby(config.DOSAGE_COL):
        axes[2].scatter(sub[config.ETHYLENE_COL], sub[config.TARGET], alpha=0.6, label=f"{dosage} g")
    axes[2].set(title="Ethylene vs days remaining", xlabel="Ethylene (ppm)", ylabel="Days remaining")

    for ax in axes:
        ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved trend plots to {save_path}")
    plt.close(fig)
