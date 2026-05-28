import numpy as np
from matplotlib import pyplot as plt

from data import Data

study = Data.students[:, 3].astype(float)
game = Data.students[:, 4].astype(float)
exercise = Data.students[:, 5].astype(float)
sleep = Data.students[:, 6].astype(float)
score = Data.students[:, 7].astype(float)

sleep_distance = (sleep - 7) ** 2

names = np.array([
    "Study", "Game", "Exercise", "Sleep", "Sleep_Distance", "Score"
])
data = np.column_stack([
    study, game, exercise, sleep, sleep_distance, score
])
data = data[~np.isnan(data).any(axis=1)]
correlation = np.corrcoef(data, rowvar=False)

plt.figure(figsize=(8, 5))
plt.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)
plt.xticks(range(len(names)), names, rotation=30)
plt.yticks(range(len(names)), names)
plt.colorbar()

for row in range(len(names)):
    for col in range(len(names)):
        text_color = "white" if abs(correlation[row, col]) > 0.6 else "black"
        plt.text(col, row, f"{correlation[row, col]:.2f}",
                 ha="center", va="center", fontsize=8, color=text_color)

plt.title("008 Engineered Correlation")
plt.tight_layout()
plt.show()
