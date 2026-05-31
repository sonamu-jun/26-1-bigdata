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

# 모델 정답값 설정
score = data[:, 2]


# ============================================================
# 4. 모델 학습
# ============================================================
# 독립변수 행렬 생성
independent_variable_matrix = np.column_stack([
    study_hours,
    np.ones_like(study_hours),
])

# 회귀 계수 계산
slope, intercept = np.linalg.lstsq(independent_variable_matrix, score, rcond=None)[0]

# 예측값 계산
score_pred = slope * study_hours + intercept


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
equation_text = f"score = {slope:.4f}*study_hours + {intercept:.4f}"

print("[Simple Linear Regression]")
print(equation_text)
print(f"MAE  = {mae:.4f}")
print(f"RMSE = {rmse:.4f}")
print(f"R2   = {r2:.4f}")


# ============================================================
# 7. 결과 시각화
# ============================================================
sorted_index = np.argsort(study_hours)
fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

# 회귀선 그래프 설정
axes[0].scatter(study_hours, score, alpha=0.65, label="samples", color="tab:blue")
axes[0].plot(
    study_hours[sorted_index],
    score_pred[sorted_index],
    color="tab:red",
    linewidth=2.5,
    label="fitted line",
)
axes[0].set_title("Simple Linear Regression")
axes[0].set_xlabel("study_hours")
axes[0].set_ylabel("score_simple")
axes[0].legend(loc="lower right")
axes[0].grid(alpha=0.25)

# 회귀식 박스 설정
axes[0].text(
    0.04,
    0.96,
    equation_text,
    transform=axes[0].transAxes,
    va="top",
    bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
)

# 실제값-예측값 그래프 설정
axes[1].scatter(score, score_pred, alpha=0.65, color="tab:green")
min_value = min(score.min(), score_pred.min())
max_value = max(score.max(), score_pred.max())
padding = max((max_value - min_value) * 0.05, 1.0)
lower = min_value - padding
upper = max_value + padding

axes[1].plot([lower, upper], [lower, upper], color="tab:red")
axes[1].set_xlim(lower, upper)
axes[1].set_ylim(lower, upper)
axes[1].set_aspect("equal", adjustable="box")
axes[1].set_title("Actual vs Predicted")
axes[1].set_xlabel("actual score")
axes[1].set_ylabel("predicted score")
axes[1].grid(alpha=0.25)

# 평가 지표 박스 설정
result_text = (
    f"MAE  = {mae:.4f}\n"
    f"RMSE = {rmse:.4f}\n"
    f"R2   = {r2:.4f}"
)
axes[1].text(
    0.04,
    0.96,
    result_text,
    transform=axes[1].transAxes,
    va="top",
    bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
)

plt.show()
