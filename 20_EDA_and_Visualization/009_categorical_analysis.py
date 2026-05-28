import numpy as np
from matplotlib import pyplot as plt

from data import Data

age = Data.students[:, 2].astype(int)
score = Data.students[:, 7].astype(float)
age_groups, counts = np.unique(age, return_counts=True)
score_means = []

for group in age_groups:
    score_means.append(np.nanmean(score[age == group]))

score_means = np.array(score_means)

fig, axes = plt.subplots(1, 2, figsize=(9, 4))

axes[0].bar(age_groups, counts)
axes[0].set_title("Age Counts")
axes[0].set_xlabel("Age")
axes[0].set_ylabel("Students")

axes[1].bar(age_groups, score_means)
axes[1].set_title("Mean Score by Age")
axes[1].set_xlabel("Age")
axes[1].set_ylabel("Mean Score")
axes[1].set_ylim(0, 100)

plt.tight_layout()
plt.show()
