import numpy as np
from matplotlib import pyplot as plt

from data import Data

score = Data.students[:, 7].astype(float)
features = Data.students[:, [3, 4, 5, 6]].astype(float)
names = np.array(["Study", "Game", "Exercise", "Sleep"])

fig, axes = plt.subplots(2, 2, figsize=(8, 6))

for ax, values, name in zip(axes.ravel(), features.T, names):
    valid = ~np.isnan(values) & ~np.isnan(score)
    correlation = np.corrcoef(values[valid], score[valid])[0, 1]
    ax.scatter(values[valid], score[valid])
    ax.set_title(f"{name} vs Score, r={correlation:.2f}")
    ax.set_xlabel(name)
    ax.set_ylabel("Score")

plt.tight_layout()
plt.show()
