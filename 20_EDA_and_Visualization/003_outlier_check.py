import numpy as np
from matplotlib import pyplot as plt

from data import Data

names = Data.columns[[3, 4, 5, 6, 7]]
data = Data.students[:, [3, 4, 5, 6, 7]].astype(float)
student_ids = Data.students[:, 0].astype(int)

fig, axes = plt.subplots(2, len(names), figsize=(15, 6))

for col, name in enumerate(names):
    raw_values = data[:, col]
    valid = ~np.isnan(raw_values)
    values = raw_values[valid]
    ids = student_ids[valid]

    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    iqr_mask = (values < lower) | (values > upper)
    iqr_outliers = values[iqr_mask]
    iqr_outlier_ids = ids[iqr_mask]

    z_scores = (values - np.mean(values)) / np.std(values)
    z_mask = np.abs(z_scores) > 3
    colors = np.where(z_mask, "red", "black")

    axes[0, col].boxplot(values, showfliers=False)
    axes[0, col].scatter(np.ones_like(iqr_outliers),
                         iqr_outliers,
                         color="red")
    for outlier_id, value in zip(iqr_outlier_ids, iqr_outliers):
        axes[0, col].text(1.05, value, str(outlier_id),
                          ha="left", va="center", fontsize=8)
    axes[0, col].axhline(lower, color="red", linestyle="--")
    axes[0, col].axhline(upper, color="red", linestyle="--")
    axes[0, col].set_title(f"{name}: IQR")
    axes[0, col].set_xticks([])

    y = np.zeros(len(z_scores))
    axes[1, col].scatter(z_scores, y, c=colors)
    for outlier_id, z_score, y_value in zip(ids[z_mask], z_scores[z_mask], y[z_mask]):
        axes[1, col].text(z_score, y_value, str(outlier_id),
                          ha="left", va="bottom", fontsize=8)
    axes[1, col].axvline(3, color="red", linestyle="--")
    axes[1, col].axvline(-3, color="red", linestyle="--")
    axes[1, col].set_title(f"{name}: Z-score")
    axes[1, col].set_xlabel("Z-score")
    axes[1, col].set_yticks([])

plt.tight_layout()
plt.show()
