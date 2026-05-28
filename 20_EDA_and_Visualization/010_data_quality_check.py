import numpy as np
from matplotlib import pyplot as plt

from data import Data

students = Data.students
numeric = students[:, [2, 3, 4, 5, 6, 7]].astype(float)
ids = students[:, 0].astype(int)

duplicate_id_count = len(ids) - len(np.unique(ids))
missing_count = np.sum(students != students)
age_error_count = np.sum((numeric[:, 0] < 15) | (numeric[:, 0] > 19))
hour_error_count = np.sum((numeric[:, 1:5] < 0) | (numeric[:, 1:5] > 24))
score_error_count = np.sum((numeric[:, 5] < 0) | (numeric[:, 5] > 100))
outlier_count = 0

for col in range(numeric.shape[1]):
    values = numeric[:, col]
    values = values[~np.isnan(values)]
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outlier_count += np.sum((values < lower) | (values > upper))

quality_names = np.array([
    "Duplicate ID", "Missing", "Age Error", "Hour Error", "Score Error", "Outlier"
])
quality_counts = np.array([
    duplicate_id_count,
    missing_count,
    age_error_count,
    hour_error_count,
    score_error_count,
    outlier_count,
])

plt.bar(quality_names, quality_counts)
plt.title("010 Data Quality Check")
plt.xlabel("Check")
plt.ylabel("Problem Count")
plt.xticks(rotation=30)
plt.ylim(0, max(1, np.max(quality_counts) + 1))
plt.show()
