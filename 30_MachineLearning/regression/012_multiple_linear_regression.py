from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt


# ============================================================
# 1. 기본 설정
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "regression_samples.csv"


# ============================================================
# 2. 데이터 파일 확인
# ============================================================
if not DATA_PATH.exists():
    raise FileNotFoundError(f"Missing data file: {DATA_PATH}")


# ============================================================
# 3. 데이터 불러오기
# ============================================================
data = np.loadtxt(DATA_PATH, delimiter=",", skiprows=1)

# 모델 입력값 설정
study_hours = data[:, 0]
sleep_hours = data[:, 1]

# 모델 정답값 설정
score = data[:, 3]


# ============================================================
# 4. 모델 학습
# ============================================================
# 독립변수 행렬 생성
independent_variable_matrix = np.column_stack([
    study_hours,
    sleep_hours,
    np.ones_like(study_hours),
])

# 회귀 계수 계산
coef_study, coef_sleep, intercept = np.linalg.lstsq(
    independent_variable_matrix,
    score,
    rcond=None,
)[0]

# 예측값 계산
score_pred = coef_study * study_hours + coef_sleep * sleep_hours + intercept


# ============================================================
# 5. 모델 평가
# ============================================================
# 오차 계산
error = score - score_pred

# 평가 지표 계산
mae = np.mean(np.abs(error))
rmse = np.sqrt(np.mean(error ** 2))
residual_sum = np.sum((score - score_pred) ** 2)
total_sum = np.sum((score - np.mean(score)) ** 2)
r2 = 1.0 - residual_sum / total_sum


# ============================================================
# 6. 결과 출력
# ============================================================
equation_text = (
    f"score = {coef_study:.4f}*study_hours\n"
    f"      + {coef_sleep:.4f}*sleep_hours + {intercept:.4f}"
)

print("[Multiple Linear Regression]")
print(equation_text)
print(f"MAE  = {mae:.4f}")
print(f"RMSE = {rmse:.4f}")
print(f"R2   = {r2:.4f}")


# ============================================================
# 7. 결과 시각화
# ============================================================
# 회귀 평면 데이터 생성
study_grid = np.linspace(study_hours.min(), study_hours.max(), 35)
sleep_grid = np.linspace(sleep_hours.min(), sleep_hours.max(), 35)
study_mesh, sleep_mesh = np.meshgrid(study_grid, sleep_grid)
score_mesh = coef_study * study_mesh + coef_sleep * sleep_mesh + intercept

fig = plt.figure(figsize=(13, 5), constrained_layout=True)

# 회귀 평면 그래프 설정
ax = fig.add_subplot(1, 2, 1, projection="3d")
ax.scatter(study_hours, sleep_hours, score, color="tab:blue", alpha=0.55)
ax.plot_surface(study_mesh, sleep_mesh, score_mesh, cmap="viridis", alpha=0.5)
ax.set_title("Multiple Linear Regression Plane")
ax.set_xlabel("study_hours")
ax.set_ylabel("sleep_hours")
ax.set_zlabel("score_multiple")

# 회귀식 박스 설정
ax.text2D(
    0.04,
    0.96,
    equation_text,
    transform=ax.transAxes,
    va="top",
    bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
)

# 실제값-예측값 그래프 설정
ax_result = fig.add_subplot(1, 2, 2)
ax_result.scatter(score, score_pred, alpha=0.65, color="tab:green")

min_value = min(score.min(), score_pred.min())
max_value = max(score.max(), score_pred.max())
padding = max((max_value - min_value) * 0.05, 1.0)
lower = min_value - padding
upper = max_value + padding

ax_result.plot([lower, upper], [lower, upper], color="tab:red")
ax_result.set_xlim(lower, upper)
ax_result.set_ylim(lower, upper)
ax_result.set_aspect("equal", adjustable="box")
ax_result.set_title("Actual vs Predicted")
ax_result.set_xlabel("actual score")
ax_result.set_ylabel("predicted score")
ax_result.grid(alpha=0.25)

# 평가 지표 박스 설정
result_text = (
    f"MAE  = {mae:.4f}\n"
    f"RMSE = {rmse:.4f}\n"
    f"R2   = {r2:.4f}"
)
ax_result.text(
    0.04,
    0.96,
    result_text,
    transform=ax_result.transAxes,
    va="top",
    bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
)

plt.show()
