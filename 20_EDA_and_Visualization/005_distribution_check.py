import numpy as np
from matplotlib import pyplot as plt

from data import Data

names = np.array(["Study", "Game", "Exercise", "Sleep", "Score"])
data = Data.students[:, [3, 4, 5, 6, 7]].astype(float)

fig, axes = plt.subplots(2, 3, figsize=(11, 6))

for ax, values, name in zip(axes.ravel(), data.T, names):
    values = values[~np.isnan(values)]
    unique_values, counts = np.unique(values, return_counts=True)
    mode = unique_values[np.argmax(counts)]
    mean = np.mean(values)
    std = np.std(values)

    if std == 0:
        skewness = 0
        kurtosis = 0
    else:
        z_scores = (values - mean) / std
        skewness = np.mean(z_scores ** 3)
        kurtosis = np.mean(z_scores ** 4) - 3

    ax.hist(values, bins=8)
    ax.axvline(mean, color="red", label="Mean")
    ax.axvline(np.median(values), color="green", label="Median")
    ax.axvline(mode, color="orange", label="Mode")
    ax.text(0.02, 0.95,
            f"Skew: {skewness:.2f}\nKurtosis: {kurtosis:.2f}",
            transform=ax.transAxes,
            va="top",
            fontsize=8)
    ax.set_title(name)
    ax.set_xlabel(name)
    ax.set_ylabel("Count")
    ax.legend()

axes.ravel()[-1].axis("off")

plt.tight_layout()
plt.show()
