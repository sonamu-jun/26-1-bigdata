import numpy as np
from matplotlib import pyplot as plt

from data import Data

columns = Data.columns
students = Data.students
data_types = np.array([
    "int", "str", "int", "float",
    "float", "float", "float", "float", "float",
])
types = np.array([
    "numeric", "categorical", "numeric", "numeric",
    "numeric", "numeric", "numeric", "numeric", "numeric",
])
row_count = students.shape[0]
column_count = students.shape[1]
numeric_count = np.sum(types == "numeric")
categorical_count = np.sum(types == "categorical")
missing_count = np.sum(students != students)
duplicate_id_count = row_count - len(np.unique(students[:, 0]))
duplicate_row_count = row_count - len({tuple(row) for row in students})
normalized_column_values = []
duplicate_of = []

for col in range(column_count):
    values = students[:, col]
    normalized = tuple("NaN" if value != value else value for value in values)

    if normalized in normalized_column_values:
        original_index = normalized_column_values.index(normalized)
        duplicate_of.append(columns[original_index])
    else:
        duplicate_of.append("-")

    normalized_column_values.append(normalized)

duplicate_column_count = np.sum(np.array(duplicate_of) != "-")

summary_table = np.array([
    ["Rows", row_count],
    ["Columns", column_count],
    ["Numeric Columns", numeric_count],
    ["Categorical Columns", categorical_count],
    ["Missing Values", missing_count],
    ["Duplicate IDs", duplicate_id_count],
    ["Duplicate Rows", duplicate_row_count],
    ["Duplicate Columns", duplicate_column_count],
], dtype=object)

column_table = np.column_stack([
    columns,
    data_types,
    types,
    [len(np.unique(students[:, col])) for col in range(len(columns))],
    [np.sum(students[:, col] != students[:, col]) for col in range(len(columns))],
    duplicate_of,
])

fig, axes = plt.subplots(2, 1, figsize=(11, 7.5))

summary = axes[0].table(cellText=summary_table,
                        colLabels=["Item", "Value"],
                        loc="center")
summary.auto_set_font_size(False)
summary.set_fontsize(9)
summary.scale(1, 1.4)
axes[0].axis("off")
axes[0].set_title("Dataset Summary")

columns_table = axes[1].table(cellText=column_table,
                              colLabels=[
                                  "Column", "Data Type", "Variable Type",
                                  "Unique Count", "Missing Count", "Duplicate Of"
                              ],
                              loc="center")
columns_table.auto_set_font_size(False)
columns_table.set_fontsize(8)
columns_table.scale(1, 1.4)
axes[1].axis("off")
axes[1].set_title("Column Structure")

plt.tight_layout()
plt.show()
