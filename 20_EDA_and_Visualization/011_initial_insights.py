import numpy as np
from matplotlib import pyplot as plt

from data import Data

names = np.array(["Study", "Game", "Exercise", "Sleep"])
features = Data.students[:, [3, 4, 5, 6]].astype(float)
score = Data.students[:, 7].astype(float)
correlations = []

for col in range(features.shape[1]):
    values = features[:, col]
    valid = ~np.isnan(values) & ~np.isnan(score)
    correlations.append(np.corrcoef(values[valid], score[valid])[0, 1])

correlations = np.array(correlations)

fig, axes = plt.subplots(2, 2, figsize=(9, 6))

axes[0, 0].bar(names, correlations)
axes[0, 0].axhline(0, color="black")
axes[0, 0].set_title("Correlation with Score")

valid = ~np.isnan(features[:, 0]) & ~np.isnan(score)
axes[0, 1].scatter(features[valid, 0], score[valid])
axes[0, 1].set_title("Study vs Score")
axes[0, 1].set_xlabel("Study")
axes[0, 1].set_ylabel("Score")

valid = ~np.isnan(features[:, 1]) & ~np.isnan(score)
axes[1, 0].scatter(features[valid, 1], score[valid])
axes[1, 0].set_title("Game vs Score")
axes[1, 0].set_xlabel("Game")
axes[1, 0].set_ylabel("Score")

valid = ~np.isnan(features[:, 3]) & ~np.isnan(score)
axes[1, 1].scatter(features[valid, 3], score[valid])
axes[1, 1].set_title("Sleep vs Score")
axes[1, 1].set_xlabel("Sleep")
axes[1, 1].set_ylabel("Score")

plt.tight_layout()
plt.show()
