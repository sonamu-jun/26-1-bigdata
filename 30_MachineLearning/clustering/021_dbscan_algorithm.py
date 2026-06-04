from pathlib import Path
import runpy

import numpy as np
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. 기본 설정
# ============================================================
RANDOM_STATE = 7
EPS = 0.28
MIN_SAMPLES = 5
NOISE_LABEL = -1
UNASSIGNED_LABEL = -99
ANIMATION_EVERY = 18
PAUSE_SECONDS = 1


def get_neighbor_indices(features_data, point_index):
    # 현재 점에서 eps 거리 안에 있는 점들의 번호를 찾는다.
    distances = np.linalg.norm(features_data - features_data[point_index], axis=1)
    neighbor_indices = np.where(distances <= EPS)[0]

    return neighbor_indices


def save_step(
    step_history,
    title,
    event,
    cluster_label,
    visited,
    current_index,
    neighbor_indices,
    step_count,
):
    # 시각화에 사용할 현재 상태를 저장한다.
    if neighbor_indices is not None:
        neighbor_indices = neighbor_indices.copy()

    step_history.append({
        "title": title,
        "event": event,
        "cluster_label": cluster_label.copy(),
        "visited": visited.copy(),
        "current_index": current_index,
        "neighbor_indices": neighbor_indices,
        "step_count": step_count,
    })


def run_dbscan(features_data):
    # DBSCAN은 가까이 붙어 있는 점들을 하나의 군집으로 묶는 방법이다.
    # 1. 처음에는 모든 점을 방문하지 않은 상태로 둔다.
    n_samples = len(features_data)
    cluster_label = np.full(n_samples, UNASSIGNED_LABEL)
    visited = np.zeros(n_samples, dtype=bool)
    cluster_id = 0
    step_count = 0
    step_history = []

    save_step(
        step_history,
        "Start",
        "start",
        cluster_label,
        visited,
        None,
        None,
        step_count,
    )

    # 2. 점을 하나씩 확인한다.
    for point_index in range(n_samples):
        # 2-1. 이미 확인한 점이면 넘어간다.
        if visited[point_index]:
            continue

        # 2-2. 현재 점을 확인했다고 표시하고, 주변 점을 찾는다.
        visited[point_index] = True
        neighbor_indices = get_neighbor_indices(features_data, point_index)

        # 2-3. 주변 점이 적으면 군집을 만들 수 없으므로 노이즈로 둔다.
        # 여기서 주변 점 개수에는 현재 점 자기 자신도 포함된다.
        if len(neighbor_indices) < MIN_SAMPLES:
            cluster_label[point_index] = NOISE_LABEL
            step_count += 1

            save_step(
                step_history,
                f"Step {step_count}: noise",
                "noise",
                cluster_label,
                visited,
                point_index,
                neighbor_indices,
                step_count,
            )
            continue

        # 2-4. 주변 점이 충분하면 새 군집을 시작한다.
        cluster_label[point_index] = cluster_id
        seed_queue = []
        seed_set = {point_index}

        # 2-5. 주변 점들을 앞으로 확인할 목록에 넣는다.
        for neighbor_index in neighbor_indices:
            neighbor_index = int(neighbor_index)

            if neighbor_index != point_index:
                seed_queue.append(neighbor_index)
                seed_set.add(neighbor_index)

        save_step(
            step_history,
            f"Start cluster {cluster_id}",
            "start_cluster",
            cluster_label,
            visited,
            point_index,
            neighbor_indices,
            step_count,
        )

        # 2-6. 확인할 주변 점이 남아 있으면 군집을 계속 넓힌다.
        while seed_queue:
            neighbor_index = int(seed_queue.pop(0))
            next_neighbors = get_neighbor_indices(features_data, neighbor_index)

            # 2-6-1. 처음 보는 점이면 확인했다고 표시한다.
            if not visited[neighbor_index]:
                visited[neighbor_index] = True

                # 2-6-2. 이 점도 주변 점이 많으면, 그 주변 점도 목록에 추가한다.
                if len(next_neighbors) >= MIN_SAMPLES:
                    for next_index in next_neighbors:
                        next_index = int(next_index)

                        if next_index not in seed_set:
                            seed_queue.append(next_index)
                            seed_set.add(next_index)

            # 2-6-3. 아직 군집이 없거나 노이즈였던 점은 현재 군집에 넣는다.
            if cluster_label[neighbor_index] in [UNASSIGNED_LABEL, NOISE_LABEL]:
                cluster_label[neighbor_index] = cluster_id

            step_count += 1

            save_step(
                step_history,
                f"Step {step_count}: cluster {cluster_id}",
                "expand",
                cluster_label,
                visited,
                neighbor_index,
                next_neighbors,
                step_count,
            )

        save_step(
            step_history,
            f"Cluster {cluster_id} complete",
            "cluster_complete",
            cluster_label,
            visited,
            None,
            None,
            step_count,
        )

        # 2-7. 한 군집이 끝났으므로 다음 군집 번호로 넘어간다.
        cluster_id += 1

    # 3. 모든 점을 확인했으면 결과를 돌려준다.
    save_step(
        step_history,
        "DBSCAN Result",
        "result",
        cluster_label,
        visited,
        None,
        None,
        step_count,
    )

    return {
        "cluster_label": cluster_label,
        "visited": visited,
        "step_count": step_count,
        "step_history": step_history,
    }


# ============================================================
# 2. 예제 데이터 생성
# ============================================================
features, _ = make_moons(
    n_samples=260,
    noise=0.08,
    random_state=RANDOM_STATE,
)

rng = np.random.default_rng(RANDOM_STATE)
noise_features = rng.uniform(
    low=[-1.7, -0.8],
    high=[2.7, 1.3],
    size=(30, 2),
)

features = np.vstack([features, noise_features])
features[:, 1] = features[:, 1] * 6 + 10


# ============================================================
# 3. 표준화
# ============================================================
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)


# ============================================================
# 4. DBSCAN 반복
# ============================================================
result = run_dbscan(features_scaled)
step_history = result["step_history"]


# ============================================================
# 5. 시각화 파일 실행
# ============================================================
visualization_path = Path(__file__).with_name("022_dbscan_visualization.py")
visualization = runpy.run_path(visualization_path)
visualization["visualize_dbscan"](
    features_scaled,
    step_history,
    EPS,
    ANIMATION_EVERY,
    PAUSE_SECONDS,
)
