from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D


# ============================================================
# 1. 기본 설정
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
TRAIN_DATA_PATH = BASE_DIR / "data" / "classification_training.csv"
VALIDATION_DATA_PATH = BASE_DIR / "data" / "classification_valid.csv"
TEST_DATA_PATH = BASE_DIR / "data" / "classification_test.csv"

LEARNING_RATE = 0.1
N_EPOCHS = 5000
EPSILON = 1e-12
CLASS_NAMES = np.array(["basic", "intermediate", "advanced"])
CLASS_COLORS = ["#4C78A8", "#F58518", "#54A24B"]


def softmax(linear_scores):
    shifted_scores = linear_scores - np.max(linear_scores, axis=1, keepdims=True)
    exp_scores = np.exp(shifted_scores)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


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


def multiclass_cross_entropy(target_one_hot, class_probability):
    safe_probability = np.clip(class_probability, EPSILON, 1)
    return -np.mean(np.sum(target_one_hot * np.log(safe_probability), axis=1))


def load_dataset(data_path):
    data = np.loadtxt(data_path, delimiter=",", skiprows=1)
    features = data[:, :2]
    target = data[:, 3].astype(int)
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
features_train, skill_level_train = load_dataset(TRAIN_DATA_PATH)
features_validation, skill_level_validation = load_dataset(VALIDATION_DATA_PATH)
features_test, skill_level_test = load_dataset(TEST_DATA_PATH)
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
# 5. 정답값 원-핫 인코딩
# ============================================================
n_classes = len(CLASS_NAMES)
skill_level_train_one_hot = np.eye(n_classes)[skill_level_train]
skill_level_validation_one_hot = np.eye(n_classes)[skill_level_validation]


# ============================================================
# 6. 그래디언트 디센트 학습
# ============================================================
# 모델 파라미터 초기화
weights = np.zeros((features_train_poly.shape[1], n_classes))
bias = np.zeros(n_classes)
n_samples = len(skill_level_train)
train_loss_history = []
validation_loss_history = []
best_validation_loss = np.inf
best_epoch = 0
best_weights = weights.copy()
best_bias = bias.copy()

for epoch in range(N_EPOCHS):
    # 클래스별 선형값 계산
    class_score = features_train_poly @ weights + bias

    # 클래스별 예측 확률 계산
    class_probability = softmax(class_score)

    # 소프트맥스와 크로스 엔트로피를 체인룰로 정리한 기울기 계산
    score_gradient = class_probability - skill_level_train_one_hot

    # 모델 파라미터 기울기 계산
    weight_gradient = features_train_poly.T @ score_gradient / n_samples
    bias_gradient = np.mean(score_gradient, axis=0)

    # 모델 파라미터 수정
    weights = weights - LEARNING_RATE * weight_gradient
    bias = bias - LEARNING_RATE * bias_gradient

    # 학습 손실, 검증 손실 계산
    train_probability = softmax(features_train_poly @ weights + bias)
    validation_probability = softmax(features_validation_poly @ weights + bias)
    train_loss = multiclass_cross_entropy(
        skill_level_train_one_hot,
        train_probability,
    )
    validation_loss = multiclass_cross_entropy(
        skill_level_validation_one_hot,
        validation_probability,
    )

    train_loss_history.append(train_loss)
    validation_loss_history.append(validation_loss)

    # 검증 손실이 가장 낮은 모델 저장
    if validation_loss < best_validation_loss:
        best_validation_loss = validation_loss
        best_epoch = epoch
        best_weights = weights.copy()
        best_bias = bias.copy()

# 검증 손실이 가장 낮았던 모델 선택
weights = best_weights.copy()
bias = best_bias.copy()


# ============================================================
# 7. 모델 평가
# ============================================================
# 테스트 데이터 예측값 계산
class_score = features_test_poly @ weights + bias
class_probability = softmax(class_score)
skill_level_pred = np.argmax(class_probability, axis=1)

# 테스트 데이터 평가 지표 계산
accuracy = np.mean(skill_level_pred == skill_level_test)

class_precision = np.zeros(n_classes)
class_recall = np.zeros(n_classes)
class_f1_score = np.zeros(n_classes)

for class_index in range(n_classes):
    true_positive = np.sum(
        (skill_level_test == class_index) & (skill_level_pred == class_index)
    )
    false_positive = np.sum(
        (skill_level_test != class_index) & (skill_level_pred == class_index)
    )
    false_negative = np.sum(
        (skill_level_test == class_index) & (skill_level_pred != class_index)
    )

    class_precision[class_index] = true_positive / max(
        true_positive + false_positive,
        1,
    )
    class_recall[class_index] = true_positive / max(
        true_positive + false_negative,
        1,
    )
    class_f1_score[class_index] = (
        2
        * class_precision[class_index]
        * class_recall[class_index]
        / max(class_precision[class_index] + class_recall[class_index], EPSILON)
    )

precision = np.mean(class_precision)
recall = np.mean(class_recall)
f1_score = np.mean(class_f1_score)


# ============================================================
# 8. 결과 시각화
# ============================================================
logit_text_lines = ["z = w1*x1 + w2*x2 + w3*x1^2 + w4*x1*x2 + w5*x2^2 + b"]
for class_index, class_name in enumerate(CLASS_NAMES):
    logit_text_lines.append(
        f"z_{class_name} = "
        f"{weights[0, class_index]:.2f}*x1 "
        f"+ {weights[1, class_index]:.2f}*x2 "
        f"+ {weights[2, class_index]:.2f}*x1^2 "
        f"+ {weights[3, class_index]:.2f}*x1*x2 "
        f"+ {weights[4, class_index]:.2f}*x2^2 "
        f"+ {bias[class_index]:.2f}"
    )
logit_text = "\n".join(logit_text_lines)

# 결정 영역 데이터 생성
x_grid = np.linspace(practice_hours.min() - 0.5, practice_hours.max() + 0.5, 200)
y_grid = np.linspace(quiz_score.min() - 3, quiz_score.max() + 3, 200)
x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)
grid_features = np.column_stack([
    x_mesh.ravel(),
    y_mesh.ravel(),
])
grid_features_scaled = (grid_features - feature_mean) / feature_std
grid_features_poly = make_polynomial_features(grid_features_scaled)
grid_probability = softmax(grid_features_poly @ weights + bias)
grid_pred = np.argmax(grid_probability, axis=1).reshape(x_mesh.shape)

fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

# 결정 영역 그래프 설정
axes[0].contourf(
    x_mesh,
    y_mesh,
    grid_pred,
    levels=np.arange(n_classes + 1) - 0.5,
    cmap=ListedColormap(CLASS_COLORS),
    alpha=0.2,
)

for class_index, class_name in enumerate(CLASS_NAMES):
    train_mask = skill_level_train == class_index
    axes[0].scatter(
        features_train[train_mask, 0],
        features_train[train_mask, 1],
        alpha=0.18,
        color=CLASS_COLORS[class_index],
    )

for class_index, class_name in enumerate(CLASS_NAMES):
    validation_mask = skill_level_validation == class_index
    axes[0].scatter(
        features_validation[validation_mask, 0],
        features_validation[validation_mask, 1],
        alpha=0.8,
        color=CLASS_COLORS[class_index],
        edgecolors="gray",
        marker="D",
        s=65,
    )

for class_index, class_name in enumerate(CLASS_NAMES):
    test_mask = skill_level_test == class_index
    axes[0].scatter(
        features_test[test_mask, 0],
        features_test[test_mask, 1],
        alpha=0.9,
        color=CLASS_COLORS[class_index],
        edgecolors="black",
        marker="X",
        s=90,
    )

axes[0].set_title("Polynomial Multiclass Logistic Classification")
axes[0].set_xlabel("practice_hours")
axes[0].set_ylabel("quiz_score")
legend_items = [
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
]

for class_index, class_name in enumerate(CLASS_NAMES):
    legend_items.append(
        Line2D(
            [0],
            [0],
            marker="o",
            color=CLASS_COLORS[class_index],
            linestyle="",
            label=class_name,
        )
    )

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
axes[1].set_title("Train / Validation Multiclass Cross Entropy Loss")
axes[1].set_xlabel("epoch")
axes[1].set_ylabel("loss")
axes[1].legend(loc="upper right")
axes[1].grid(alpha=0.25)

# 클래스별 로짓 식 박스 설정
axes[1].text(
    0.04,
    0.96,
    logit_text,
    transform=axes[1].transAxes,
    va="top",
    fontsize=8,
    bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
)

plt.show()
