"""
Basic data checks and exploratory plots.
"""

import matplotlib.pyplot as plt

import config


def check_data(df):
    print("Missing values per column:")
    print(df.isna().sum())
    print()
    print("Rows per dosage:")
    print(df["kmno4_dosage"].value_counts())
    print()
    print("Distinct bananas per dosage:")
    print(df.groupby("kmno4_dosage")[config.GROUP_COL].nunique())
    print()
    print("colour_value range check (should be 1-7):", df["colour_value"].min(), "-", df["colour_value"].max())


def plot_trends(df, save_path=None):
    """Plots mean ethylene and colour value over time, by dosage."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    for dosage, sub in df.groupby("kmno4_dosage"):
        trend = sub.groupby("day")["ethylene_ppm"].mean()
        axes[0].plot(trend.index, trend.values, marker="o", label=f"{dosage}g")
    axes[0].set_title("Mean ethylene concentration over time")
    axes[0].set_xlabel("Day"); axes[0].set_ylabel("Ethylene (ppm)"); axes[0].legend()

    for dosage, sub in df.groupby("kmno4_dosage"):
        trend = sub.groupby("day")["colour_value"].mean()
        axes[1].plot(trend.index, trend.values, marker="o", label=f"{dosage}g")
    axes[1].set_title("Mean peel colour stage over time")
    axes[1].set_xlabel("Day"); axes[1].set_ylabel("Colour value (1-7)"); axes[1].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved trend plots to {save_path}")
    plt.close(fig)
