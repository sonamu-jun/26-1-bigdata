from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt


# ============================================================
# 1. 기본 설정
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "nonlinear_regression_samples.csv"

LEARNING_RATE = 0.05
N_EPOCHS = 3000


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
practice_days = data[:, 0]

# 모델 정답값 설정
skill_score = data[:, 1]


# ============================================================
# 4. 비선형 입력값 생성
# ============================================================
# 로그 입력값 생성
log_feature = np.log1p(practice_days)

# 입력값 표준화
feature_mean = np.mean(log_feature)
feature_std = np.std(log_feature)

if feature_std == 0:
    raise ValueError("feature_std is 0, so gradient descent cannot be used.")

feature_scaled = (log_feature - feature_mean) / feature_std


# ============================================================
# 5. 그래디언트 디센트 학습
# ============================================================
# 모델 파라미터 초기화
weight_scaled = 0.0
bias_scaled = np.mean(skill_score)
n_samples = len(skill_score)

for epoch in range(N_EPOCHS):
    # 예측값 계산
    skill_pred_scaled = weight_scaled * feature_scaled + bias_scaled

    # 오차 계산
    error = skill_pred_scaled - skill_score

    # 기울기 계산
    weight_gradient = (2 / n_samples) * np.sum(error * feature_scaled)
    bias_gradient = (2 / n_samples) * np.sum(error)

    # 모델 파라미터 수정
    weight_scaled = weight_scaled - LEARNING_RATE * weight_gradient
    bias_scaled = bias_scaled - LEARNING_RATE * bias_gradient


# ============================================================
# 6. 회귀 계수 변환
# ============================================================
coef_log = weight_scaled / feature_std
intercept = bias_scaled - weight_scaled * feature_mean / feature_std

skill_pred = coef_log * log_feature + intercept


# ============================================================
# 7. 모델 평가
# ============================================================
# 오차 계산
error = skill_score - skill_pred

# 평가 지표 계산
mae = np.mean(np.abs(error))
rmse = np.sqrt(np.mean(error ** 2))
residual_sum = np.sum((skill_score - skill_pred) ** 2)
total_sum = np.sum((skill_score - np.mean(skill_score)) ** 2)
r2 = 1.0 - residual_sum / total_sum


# ============================================================
# 8. 결과 출력
# ============================================================
intercept_sign = "+" if intercept >= 0 else "-"
equation_text = (
    f"skill_score = {coef_log:.4f}*log1p(practice_days)\n"
    f"            {intercept_sign} {abs(intercept):.4f}"
)

print("[Nonlinear Regression with Gradient Descent]")
print(equation_text)
print(f"MAE  = {mae:.4f}")
print(f"RMSE = {rmse:.4f}")
print(f"R2   = {r2:.4f}")


# ============================================================
# 9. 결과 시각화
# ============================================================
# 회귀 곡선 데이터 생성
x_curve = np.linspace(practice_days.min(), practice_days.max(), 300)
y_curve = coef_log * np.log1p(x_curve) + intercept

fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

# 회귀 곡선 그래프 설정
axes[0].scatter(practice_days, skill_score, alpha=0.65, label="samples", color="tab:blue")
axes[0].plot(
    x_curve,
    y_curve,
    color="tab:red",
    linewidth=2.5,
    label="gradient descent fit",
)
axes[0].set_title("Nonlinear Regression with Gradient Descent")
axes[0].set_xlabel("practice_days")
axes[0].set_ylabel("skill_score")
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
axes[1].scatter(skill_score, skill_pred, alpha=0.65, color="tab:green")

min_value = min(skill_score.min(), skill_pred.min())
max_value = max(skill_score.max(), skill_pred.max())
padding = max((max_value - min_value) * 0.05, 1.0)
lower = min_value - padding
upper = max_value + padding

axes[1].plot([lower, upper], [lower, upper], color="tab:red")
axes[1].set_xlim(lower, upper)
axes[1].set_ylim(lower, upper)
axes[1].set_aspect("equal", adjustable="box")
axes[1].set_title("Actual vs Predicted")
axes[1].set_xlabel("actual skill_score")
axes[1].set_ylabel("predicted skill_score")
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
