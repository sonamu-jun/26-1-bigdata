import numpy as np
import pandas as pd


scores = np.array([72, 85, 90, 66, 78, 92, 88, 76, 95, 80])

df = pd.DataFrame({
    "Student": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],
    "Score": scores
})

print("Dataset")
print(df)


# ============================================================
# Measures of Central Tendency
# ============================================================

print("\nMeasures of Central Tendency")

print("Mean:", df["Score"].mean())
print("Median:", df["Score"].median())

print("Mode:")
print(df["Score"].mode())


# ============================================================
# Mode Example
# ============================================================

scores_with_mode = [70, 80, 80, 85, 90, 90, 90, 95]

df_mode = pd.DataFrame({
    "Score": scores_with_mode
})

print("\nMode Example")
print(df_mode["Score"].mode())


# ============================================================
# Measures of Dispersion
# ============================================================

print("\nMeasures of Dispersion")

maximum_score = df["Score"].max()
minimum_score = df["Score"].min()
score_range = maximum_score - minimum_score

print("Maximum:", maximum_score)
print("Minimum:", minimum_score)
print("Range:", score_range)

print("Population Variance:", df["Score"].var(ddof=0))
print("Sample Variance:", df["Score"].var(ddof=1))

print("Population Standard Deviation:", df["Score"].std(ddof=0))
print("Sample Standard Deviation:", df["Score"].std(ddof=1))


# ============================================================
# Measures of Position
# ============================================================

print("\nMeasures of Position")

q1 = df["Score"].quantile(0.25)
q2 = df["Score"].quantile(0.50)
q3 = df["Score"].quantile(0.75)
iqr = q3 - q1
p90 = df["Score"].quantile(0.90)

print("Q1:", q1)
print("Q2:", q2)
print("Q3:", q3)
print("IQR:", iqr)
print("90th Percentile:", p90)


# ============================================================
# Shape of Distribution
# ============================================================

print("\nShape of Distribution")

print("Skewness:", df["Score"].skew())
print("Kurtosis:", df["Score"].kurt())


# ============================================================
# Summary Statistics
# ============================================================

print("\nSummary Statistics")

print(df["Score"].describe())