from matplotlib import pyplot as plt
from matplotlib.patches import Circle


def visualize_dbscan(
    features_scaled,
    step_history,
    eps,
    animation_every,
    pause_seconds,
):
    # 021_dbscan_algorithm.py에서 계산한 값을 받아 시각화한다.
    plt.ion()
    _, ax = plt.subplots(num="DBSCAN Iteration", figsize=(6, 5))
    plt.show(block=False)

    for step in step_history:
        if step["event"] == "expand" and step["step_count"] % animation_every != 0:
            continue

        ax.clear()

        cluster_label = step["cluster_label"]
        current_index = step["current_index"]
        neighbor_indices = step["neighbor_indices"]

        ax.scatter(
            features_scaled[:, 0],
            features_scaled[:, 1],
            c=cluster_label,
            s=15,
        )

        if neighbor_indices is not None:
            ax.scatter(
                features_scaled[neighbor_indices, 0],
                features_scaled[neighbor_indices, 1],
                color="black",
                marker="x",
                s=40,
            )

        if current_index is not None:
            current_point = features_scaled[current_index]
            ax.scatter(current_point[0], current_point[1], color="red", s=80)
            ax.add_patch(Circle(current_point, eps, fill=False, color="red"))

        ax.set_title(step["title"])
        plt.pause(pause_seconds)

    plt.ioff()
    plt.show()
