from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from sklearn.tree import DecisionTreeClassifier, plot_tree


# ============================================================
# 1. 기본 설정
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
TRAIN_DATA_PATH = BASE_DIR / "data" / "classification_training.csv"
VALIDATION_DATA_PATH = BASE_DIR / "data" / "classification_valid.csv"
TEST_DATA_PATH = BASE_DIR / "data" / "classification_test.csv"

MAX_DEPTH_CANDIDATES = np.arange(1, 8)
MIN_SAMPLES_LEAF = 3
RANDOM_STATE = 7
FEATURE_NAMES = ["practice_hours", "quiz_score"]
CLASS_NAMES = ["basic", "intermediate", "advanced"]
CLASS_COLORS = ["#4C78A8", "#F58518", "#54A24B"]


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
# 4. 모델 학습
# ============================================================
# 후보 모델 검증
train_error_history = []
validation_error_history = []
best_validation_error = np.inf
best_depth = None
best_tree_model = None

for max_depth in MAX_DEPTH_CANDIDATES:
    # 결정 트리 모델 생성
    candidate_tree_model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=RANDOM_STATE,
    )

    # 결정 트리 모델 학습
    candidate_tree_model.fit(features_train, skill_level_train)

    # 학습 오류, 검증 오류 계산
    train_pred = candidate_tree_model.predict(features_train)
    validation_pred = candidate_tree_model.predict(features_validation)
    train_error = 1 - np.mean(train_pred == skill_level_train)
    validation_error = 1 - np.mean(validation_pred == skill_level_validation)

    train_error_history.append(train_error)
    validation_error_history.append(validation_error)

    # 검증 오류가 가장 낮은 모델 저장
    if validation_error < best_validation_error:
        best_validation_error = validation_error
        best_depth = max_depth
        best_tree_model = candidate_tree_model

# 검증 오류가 가장 낮았던 모델 선택
tree_model = best_tree_model

# 테스트 데이터 예측값 계산
skill_level_pred = tree_model.predict(features_test)


# ============================================================
# 5. 모델 평가
# ============================================================
n_classes = len(CLASS_NAMES)
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
        / max(class_precision[class_index] + class_recall[class_index], 1e-12)
    )

precision = np.mean(class_precision)
recall = np.mean(class_recall)
f1_score = np.mean(class_f1_score)


# ============================================================
# 6. 결과 시각화
# ============================================================
# 결정 영역 데이터 생성
x_grid = np.linspace(practice_hours.min() - 0.5, practice_hours.max() + 0.5, 200)
y_grid = np.linspace(quiz_score.min() - 3, quiz_score.max() + 3, 200)
x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)
grid_features = np.column_stack([
    x_mesh.ravel(),
    y_mesh.ravel(),
])
grid_pred = tree_model.predict(grid_features).reshape(x_mesh.shape)

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)

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

axes[0].set_title("Decision Tree Classification")
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
    f"Test F1-score  = {f1_score:.4f}\n"
    f"Best Depth     = {best_depth}"
)
axes[0].text(
    0.04,
    0.96,
    result_text,
    transform=axes[0].transAxes,
    va="top",
    bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
)

# 학습 오류, 검증 오류 그래프 설정
axes[1].plot(
    MAX_DEPTH_CANDIDATES,
    train_error_history,
    marker="o",
    color="tab:purple",
    linewidth=2,
    label="train",
)
axes[1].plot(
    MAX_DEPTH_CANDIDATES,
    validation_error_history,
    marker="o",
    color="tab:orange",
    linewidth=2,
    label="validation",
)
axes[1].axvline(
    best_depth,
    color="tab:red",
    linestyle="--",
    linewidth=1.5,
    label="best validation",
)
axes[1].set_title("Train / Validation Error by Max Depth")
axes[1].set_xlabel("max_depth")
axes[1].set_ylabel("error")
axes[1].set_xticks(MAX_DEPTH_CANDIDATES)
axes[1].legend(loc="best")
axes[1].grid(alpha=0.25)

# 결정 트리 그래프 설정
plot_tree(
    tree_model,
    feature_names=FEATURE_NAMES,
    class_names=CLASS_NAMES,
    filled=True,
    rounded=True,
    impurity=False,
    fontsize=8,
    ax=axes[2],
)
axes[2].set_title("Best Validation Decision Tree")

plt.show()
