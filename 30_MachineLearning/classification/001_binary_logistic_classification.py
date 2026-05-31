from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D


# ============================================================
# 1. 기본 설정
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
TRAIN_DATA_PATH = BASE_DIR / "data" / "classification_training.csv"
VALIDATION_DATA_PATH = BASE_DIR / "data" / "classification_valid.csv"
TEST_DATA_PATH = BASE_DIR / "data" / "classification_test.csv"

LEARNING_RATE = 0.2
N_EPOCHS = 4000
EPSILON = 1e-12


def sigmoid(linear_score):
    clipped_score = np.clip(linear_score, -30, 30)
    return 1 / (1 + np.exp(-clipped_score))


def make_polynomial_features(features_scaled):
    practice_scaled = features_scaled[:, 0]
    quiz_scaled = features_scaled[:, 1]

    return np.column_stack([
        practice_scaled,
        quiz_scaled,
        practice_scaled ** 2,
        practice_scaled * quiz_scaled,
        quiz_scaled ** 2,
    ])


def binary_cross_entropy(target, probability):
    safe_probability = np.clip(probability, EPSILON, 1 - EPSILON)
    return -np.mean(
        target * np.log(safe_probability)
        + (1 - target) * np.log(1 - safe_probability)
    )


def load_dataset(data_path):
    data = np.loadtxt(data_path, delimiter=",", skiprows=1)
    features = data[:, :2]
    target = data[:, 2]
    return features, target


# ============================================================
# 2. 데이터 파일 확인
# ============================================================
for data_path in [TRAIN_DATA_PATH, VALIDATION_DATA_PATH, TEST_DATA_PATH]:
    if not data_path.exists():
        raise FileNotFoundError(f"Missing data file: {data_path}")


# ============================================================
# 3. 데이터 불러오기
# ============================================================
features_train, passed_train = load_dataset(TRAIN_DATA_PATH)
features_validation, passed_validation = load_dataset(VALIDATION_DATA_PATH)
features_test, passed_test = load_dataset(TEST_DATA_PATH)
features_all = np.vstack([
    features_train,
    features_validation,
    features_test,
])
practice_hours = features_all[:, 0]
quiz_score = features_all[:, 1]


# ============================================================
# 4. 입력값 표준화
# ============================================================
# 학습 데이터 기준으로 입력값 표준화
feature_mean = np.mean(features_train, axis=0)
feature_std = np.std(features_train, axis=0)

if np.any(feature_std == 0):
    raise ValueError("feature_std has 0, so gradient descent cannot be used.")

features_train_scaled = (features_train - feature_mean) / feature_std
features_validation_scaled = (features_validation - feature_mean) / feature_std
features_test_scaled = (features_test - feature_mean) / feature_std

# 다항 입력값 생성
features_train_poly = make_polynomial_features(features_train_scaled)
features_validation_poly = make_polynomial_features(features_validation_scaled)
features_test_poly = make_polynomial_features(features_test_scaled)


# ============================================================
# 5. 그래디언트 디센트 학습
# ============================================================
# 모델 파라미터 초기화
weights = np.zeros(features_train_poly.shape[1])
bias = 0.0
n_samples = len(passed_train)
train_loss_history = []
validation_loss_history = []
best_validation_loss = np.inf
best_epoch = 0
best_weights = weights.copy()
best_bias = bias

for epoch in range(N_EPOCHS):
    # 선형값 계산
    linear_score = features_train_poly @ weights + bias

    # 예측 확률 계산
    passed_probability = sigmoid(linear_score)
    safe_probability = np.clip(passed_probability, EPSILON, 1 - EPSILON)

    # 체인룰 1단계: 손실 함수가 예측 확률에 미치는 영향 계산
    probability_gradient = (
        -(passed_train / safe_probability)
        + ((1 - passed_train) / (1 - safe_probability))
    )

    # 체인룰 2단계: 시그모이드가 선형값에 미치는 영향 계산
    sigmoid_gradient = passed_probability * (1 - passed_probability)

    # 체인룰 3단계: 손실 함수가 선형값에 미치는 영향 계산
    linear_gradient = probability_gradient * sigmoid_gradient

    # 모델 파라미터 기울기 계산
    weight_gradient = features_train_poly.T @ linear_gradient / n_samples
    bias_gradient = np.mean(linear_gradient)

    # 모델 파라미터 수정
    weights = weights - LEARNING_RATE * weight_gradient
    bias = bias - LEARNING_RATE * bias_gradient

    # 학습 손실, 검증 손실 계산
    train_probability = sigmoid(features_train_poly @ weights + bias)
    validation_probability = sigmoid(features_validation_poly @ weights + bias)
    train_loss = binary_cross_entropy(passed_train, train_probability)
    validation_loss = binary_cross_entropy(passed_validation, validation_probability)

    train_loss_history.append(train_loss)
    validation_loss_history.append(validation_loss)

    # 검증 손실이 가장 낮은 모델 저장
    if validation_loss < best_validation_loss:
        best_validation_loss = validation_loss
        best_epoch = epoch
        best_weights = weights.copy()
        best_bias = bias

# 검증 손실이 가장 낮았던 모델 선택
weights = best_weights.copy()
bias = best_bias


# ============================================================
# 6. 모델 평가
# ============================================================
# 테스트 데이터 예측값 계산
linear_score = features_test_poly @ weights + bias
passed_probability = sigmoid(linear_score)
passed_pred = (passed_probability >= 0.5).astype(int)
passed_true = passed_test.astype(int)

# 테스트 데이터 평가 지표 계산
accuracy = np.mean(passed_pred == passed_true)
true_positive = np.sum((passed_true == 1) & (passed_pred == 1))
false_positive = np.sum((passed_true == 0) & (passed_pred == 1))
false_negative = np.sum((passed_true == 1) & (passed_pred == 0))

precision = true_positive / max(true_positive + false_positive, 1)
recall = true_positive / max(true_positive + false_negative, 1)
f1_score = 2 * precision * recall / max(precision + recall, EPSILON)


# ============================================================
# 7. 결과 시각화
# ============================================================
model_text = (
    "features = x1, x2, x1^2, x1*x2, x2^2\n"
    "x1 = scaled practice_hours\n"
    "x2 = scaled quiz_score\n"
    f"best epoch = {best_epoch + 1}\n"
    f"best validation BCE = {best_validation_loss:.4f}"
)

# 결정 경계 데이터 생성
x_grid = np.linspace(practice_hours.min() - 0.5, practice_hours.max() + 0.5, 200)
y_grid = np.linspace(quiz_score.min() - 3, quiz_score.max() + 3, 200)
x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)
grid_features = np.column_stack([
    x_mesh.ravel(),
    y_mesh.ravel(),
])
grid_features_scaled = (grid_features - feature_mean) / feature_std
grid_features_poly = make_polynomial_features(grid_features_scaled)
grid_probability = sigmoid(grid_features_poly @ weights + bias)
grid_probability = grid_probability.reshape(x_mesh.shape)
passed_train_true = passed_train.astype(int)
passed_validation_true = passed_validation.astype(int)
passed_test_true = passed_test.astype(int)

fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

# 결정 경계 그래프 설정
axes[0].contour(
    x_mesh,
    y_mesh,
    grid_probability,
    levels=[0.5],
    colors="tab:red",
    linewidths=2.5,
)
axes[0].scatter(
    features_train[passed_train_true == 0, 0],
    features_train[passed_train_true == 0, 1],
    alpha=0.18,
    color="tab:blue",
)
axes[0].scatter(
    features_train[passed_train_true == 1, 0],
    features_train[passed_train_true == 1, 1],
    alpha=0.18,
    color="tab:green",
)
axes[0].scatter(
    features_validation[passed_validation_true == 0, 0],
    features_validation[passed_validation_true == 0, 1],
    alpha=0.8,
    color="tab:blue",
    edgecolors="gray",
    marker="D",
    s=65,
)
axes[0].scatter(
    features_validation[passed_validation_true == 1, 0],
    features_validation[passed_validation_true == 1, 1],
    alpha=0.8,
    color="tab:green",
    edgecolors="gray",
    marker="D",
    s=65,
)
axes[0].scatter(
    features_test[passed_test_true == 0, 0],
    features_test[passed_test_true == 0, 1],
    alpha=0.9,
    color="tab:blue",
    edgecolors="black",
    marker="X",
    s=90,
)
axes[0].scatter(
    features_test[passed_test_true == 1, 0],
    features_test[passed_test_true == 1, 1],
    alpha=0.9,
    color="tab:green",
    edgecolors="black",
    marker="X",
    s=90,
)
axes[0].set_title("Polynomial Binary Logistic Classification")
axes[0].set_xlabel("practice_hours")
axes[0].set_ylabel("quiz_score")
legend_items = [
    Line2D([0], [0], color="tab:red", linewidth=2.5, label="p=0.5 boundary"),
    Line2D(
        [0],
        [0],
        marker="o",
        color="gray",
        linestyle="",
        label="train sample",
    ),
    Line2D(
        [0],
        [0],
        marker="D",
        color="gray",
        linestyle="",
        label="validation sample",
    ),
    Line2D(
        [0],
        [0],
        marker="X",
        color="black",
        linestyle="",
        label="test sample",
    ),
    Line2D(
        [0],
        [0],
        marker="o",
        color="tab:blue",
        linestyle="",
        label="not passed",
    ),
    Line2D(
        [0],
        [0],
        marker="o",
        color="tab:green",
        linestyle="",
        label="passed",
    ),
]
axes[0].legend(handles=legend_items, loc="lower right")
axes[0].grid(alpha=0.25)

# 평가 지표 박스 설정
result_text = (
    f"Test Accuracy  = {accuracy:.4f}\n"
    f"Test Precision = {precision:.4f}\n"
    f"Test Recall    = {recall:.4f}\n"
    f"Test F1-score  = {f1_score:.4f}"
)
axes[0].text(
    0.04,
    0.96,
    result_text,
    transform=axes[0].transAxes,
    va="top",
    bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
)

# 손실 함수 그래프 설정
axes[1].plot(train_loss_history, color="tab:purple", linewidth=2, label="train")
axes[1].plot(
    validation_loss_history,
    color="tab:orange",
    linewidth=2,
    label="validation",
)
axes[1].axvline(
    best_epoch,
    color="tab:red",
    linestyle="--",
    linewidth=1.5,
    label="best validation",
)
axes[1].set_title("Train / Validation Binary Cross Entropy Loss")
axes[1].set_xlabel("epoch")
axes[1].set_ylabel("loss")
axes[1].legend(loc="upper right")
axes[1].grid(alpha=0.25)

# 모델식 박스 설정
axes[1].text(
    0.04,
    0.96,
    model_text,
    transform=axes[1].transAxes,
    va="top",
    bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
)

plt.show()
