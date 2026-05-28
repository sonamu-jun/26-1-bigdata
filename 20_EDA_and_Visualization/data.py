import numpy as np


np.random.seed(7)

original_count = 980

ids = np.arange(1, original_count + 1)
names = np.array([f"Student{i:04d}" for i in ids])
age = np.random.choice([16, 17, 18], size=original_count, p=[0.3, 0.4, 0.3])

study = np.random.normal(5.0, 1.7, original_count)
study = np.clip(study, 0.5, 9.5)

game = 8.5 - 0.75 * study + np.random.normal(0, 1.2, original_count)
game = np.clip(game, 0.2, 10.0)

exercise = np.random.normal(1.8, 0.8, original_count)
exercise = np.clip(exercise, 0.0, 4.5)

sleep = np.random.normal(7.0, 1.0, original_count)
sleep = np.clip(sleep, 3.5, 10.5)

score = (
    40
    + 6.2 * study
    - 3.0 * game
    - 3.2 * (sleep - 7) ** 2
    + 0.2 * exercise
    + 1.0 * (age - 17)
    + np.random.normal(0, 5, original_count)
)
score = np.clip(score, 0, 100)

study = np.round(study, 1)
game = np.round(game, 1)
exercise = np.round(exercise, 1)
sleep = np.round(sleep, 1)
score = np.round(score, 0)

outlier_rows = np.array([20, 120, 240, 360, 480, 600, 720, 840])
study[outlier_rows[[0, 7]]] = [12.0, 11.5]
game[outlier_rows[[1, 2]]] = [12.8, 14.0]
exercise[outlier_rows[[3, 4]]] = [8.5, 9.2]
sleep[outlier_rows[[5, 6]]] = [2.0, 12.0]
score[outlier_rows] = [100, 5, 18, 63, 69, 38, 42, 98]

study[[35, 135, 235, 335, 435, 535]] = np.nan
game[[60, 160, 260, 460, 660, 860]] = np.nan
exercise[[80, 180, 280, 580, 780]] = np.nan
sleep[[100, 200, 300, 500, 700]] = np.nan
score[[110, 210, 310, 510, 710]] = np.nan

students = np.empty((original_count, 8), dtype=object)
students[:, 0] = ids
students[:, 1] = names
students[:, 2] = age
students[:, 3] = study
students[:, 4] = game
students[:, 5] = exercise
students[:, 6] = sleep
students[:, 7] = score

duplicate_rows = students[900:920]
students = np.vstack([students, duplicate_rows])

columns = np.array([
    "ID", "Name", "Age", "Study", "Game", "Exercise", "Sleep", "Score",
    "Study_Copy"
])
students = np.column_stack([students, students[:, 3]])


class Data:
    columns = columns
    students = students
