from pathlib import Path
import runpy

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. 기본 설정
# ============================================================
RANDOM_STATE = 7
INITIAL_CENTER_SEED = 1
N_CLUSTERS = 4
ELBOW_K_VALUES = range(1, 9)
MAX_ITERATIONS = 20
PAUSE_SECONDS = 1


def run_kmeans(features_data, n_clusters, save_history=False):
    # K-Means는 중심점을 조금씩 옮기면서 데이터를 묶는 방법이다.
    # 1. 처음 중심점을 k개 고른다.
    rng = np.random.default_rng(INITIAL_CENTER_SEED)
    center_indices = rng.choice(len(features_data), size=n_clusters, replace=False)
    centers = features_data[center_indices].copy()
    initial_centers = centers.copy()

    previous_label = None
    step_history = []
    cluster_label = None

    # 2. 결과가 더 이상 바뀌지 않을 때까지 반복한다.
    for iteration in range(1, MAX_ITERATIONS + 1):
        # 2-1. 모든 데이터가 각 중심점과 얼마나 떨어져 있는지 계산한다.
        distances = np.linalg.norm(
            features_data[:, np.newaxis, :] - centers[np.newaxis, :, :],
            axis=2,
        )

        # 2-2. 가장 가까운 중심점의 번호를 군집 번호로 정한다.
        cluster_label = np.argmin(distances, axis=1)

        # 2-3. 같은 군집끼리 평균을 내서 중심점을 새로 잡는다.
        new_centers = []

        for cluster_index in range(n_clusters):
            cluster_points = features_data[cluster_label == cluster_index]

            if len(cluster_points) == 0:
                # 데이터가 하나도 없으면 중심점을 그대로 둔다.
                cluster_center = centers[cluster_index]
            else:
                cluster_center = cluster_points.mean(axis=0)

            new_centers.append(cluster_center)

        new_centers = np.array(new_centers)

        if save_history:
            step_history.append({
                "iteration": iteration,
                "cluster_label": cluster_label.copy(),
                "old_centers": centers.copy(),
                "new_centers": new_centers.copy(),
            })

        # 2-4. 이전과 같은 군집 결과가 나오면 멈춘다.
        if (
            previous_label is not None
            and np.array_equal(cluster_label, previous_label)
        ):
            centers = new_centers
            break

        centers = new_centers
        previous_label = cluster_label.copy()

    # 3. SSE를 계산한다. SSE가 작을수록 군집이 더 잘 모인 것이다.
    distances = np.linalg.norm(
        features_data[:, np.newaxis, :] - centers[np.newaxis, :, :],
        axis=2,
    )
    closest_distances = np.min(distances, axis=1)
    sse = np.sum(closest_distances ** 2)

    return {
        "initial_centers": initial_centers,
        "centers": centers,
        "cluster_label": cluster_label,
        "sse": sse,
        "step_history": step_history,
    }


# ============================================================
# 2. 예제 데이터 생성
# ============================================================
features, _ = make_blobs(
    n_samples=300,
    centers=[[-5, -2], [-2, 3], [2, -1], [5, 3]],
    cluster_std=[1.4, 1.25, 1.35, 1.2],
    random_state=RANDOM_STATE,
)

# 두 번째 변수의 단위를 크게 만들어 표준화가 필요하게 만든다.
features[:, 1] = features[:, 1] * 5 + 30


# ============================================================
# 3. 표준화
# ============================================================
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)


# ============================================================
# 4. 엘보 방법으로 k 확인
# ============================================================
elbow_k_values = []
elbow_sse_values = []

for k_value in ELBOW_K_VALUES:
    # k를 바꿔 가며 K-Means를 돌리고 SSE를 저장한다.
    result = run_kmeans(features_scaled, k_value)
    elbow_k_values.append(k_value)
    elbow_sse_values.append(result["sse"])


# ============================================================
# 5. K-Means 반복
# ============================================================
result = run_kmeans(features_scaled, N_CLUSTERS, save_history=True)
initial_centers = result["initial_centers"]
step_history = result["step_history"]


# ============================================================
# 6. 시각화 파일 실행
# ============================================================
visualization_path = Path(__file__).with_name("012_kmeans_visualization.py")
visualization = runpy.run_path(visualization_path)
visualization["visualize_kmeans"](
    features_scaled,
    initial_centers,
    step_history,
    elbow_k_values,
    elbow_sse_values,
    N_CLUSTERS,
    PAUSE_SECONDS,
)
