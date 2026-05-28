import numpy as np
from matplotlib import pyplot as plt

from data import Data

names = Data.columns[[3, 4, 5, 6, 7]]
data = Data.students[:, [3, 4, 5, 6, 7]].astype(float)
data = data[~np.isnan(data).any(axis=1)]
correlation = np.corrcoef(data, rowvar=False)

plt.figure(figsize=(7, 5))
plt.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)
plt.xticks(range(len(names)), names, rotation=30)
plt.yticks(range(len(names)), names)
plt.colorbar()

for row in range(len(names)):
    for col in range(len(names)):
        text_color = "white" if abs(correlation[row, col]) > 0.6 else "black"
        plt.text(col, row, f"{correlation[row, col]:.2f}",
                 ha="center", va="center", fontsize=8, color=text_color)

plt.title("007 Correlation Heatmap")
plt.tight_layout()
plt.show()
