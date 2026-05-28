import numpy as np
from matplotlib import pyplot as plt

from data import Data

names = Data.columns[[2, 3, 4, 5, 6, 7, 8]]
data = Data.students[:, [2, 3, 4, 5, 6, 7, 8]].astype(float)
stat_names = np.array([
    "Count", "Missing", "Mean", "Std", "Min", "Q1", "Median", "Q3", "Max",
    "Skew", "Kurtosis"
])
stat_table = []

for col in range(data.shape[1]):
    values = data[:, col]
    valid_values = values[~np.isnan(values)]
    mean = np.mean(valid_values)
    std = np.std(valid_values)

    if std == 0:
        skewness = 0
        kurtosis = 0
    else:
        z_scores = (valid_values - mean) / std
        skewness = np.mean(z_scores ** 3)
        kurtosis = np.mean(z_scores ** 4) - 3

    stat_table.append([
        len(valid_values),
        np.sum(np.isnan(values)),
        mean,
        std,
        np.min(valid_values),
        np.percentile(valid_values, 25),
        np.median(valid_values),
        np.percentile(valid_values, 75),
        np.max(valid_values),
        skewness,
        kurtosis,
    ])

stat_table = np.round(np.array(stat_table), 2).astype(object)

plt.figure(figsize=(13.5, 4.5))
table = plt.table(cellText=stat_table,
                  rowLabels=names,
                  colLabels=stat_names,
                  loc="center")
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1, 1.4)
plt.axis("off")
plt.title("004 Descriptive Statistics")
plt.show()
