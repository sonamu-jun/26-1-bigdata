from matplotlib import pyplot as plt


def visualize_kmeans(
    features_scaled,
    initial_centers,
    step_history,
    elbow_k_values,
    elbow_sse_values,
    n_clusters,
    pause_seconds,
):
    # 011_kmeans_algorithm.py에서 계산한 값을 받아 시각화한다.
    plt.ion()
    _, elbow_ax = plt.subplots(num="K-Means Elbow Method", figsize=(6, 5))
    _, kmeans_ax = plt.subplots(num="K-Means Iteration", figsize=(6, 5))

    selected_k_index = elbow_k_values.index(n_clusters)
    selected_sse = elbow_sse_values[selected_k_index]

    elbow_ax.plot(elbow_k_values, elbow_sse_values, marker="o")
    elbow_ax.scatter(n_clusters, selected_sse, color="red", s=80, zorder=3)
    elbow_ax.axvline(n_clusters, color="red", linestyle="--", linewidth=1)
    elbow_ax.set_xticks(elbow_k_values)
    elbow_ax.set_xlabel("Number of clusters (k)")
    elbow_ax.set_ylabel("SSE")
    elbow_ax.set_title(f"Elbow Method: selected k = {n_clusters}")

    kmeans_ax.scatter(features_scaled[:, 0], features_scaled[:, 1], s=15)
    kmeans_ax.scatter(initial_centers[:, 0], initial_centers[:, 1],
                      color="red", marker="x", s=100)
    kmeans_ax.set_title("Initial Centers")
    plt.show(block=False)
    plt.pause(pause_seconds)

    for step in step_history:
        kmeans_ax.clear()

        cluster_label = step["cluster_label"]
        new_centers = step["new_centers"]

        kmeans_ax.scatter(
            features_scaled[:, 0],
            features_scaled[:, 1],
            c=cluster_label,
            cmap="tab10",
            vmin=0,
            vmax=n_clusters - 1,
            s=15,
        )
        kmeans_ax.scatter(new_centers[:, 0], new_centers[:, 1],
                          color="red", marker="x", s=100)
        kmeans_ax.set_title(f"Iteration {step['iteration']}")
        plt.pause(pause_seconds)

    kmeans_ax.set_title("K-Means Result")
    plt.ioff()
    plt.show()
