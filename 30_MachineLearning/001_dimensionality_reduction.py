import numpy as np
from matplotlib import pyplot as plt

# 1. 예제용 3차원 데이터 생성
rng = np.random.default_rng(seed=7)
n_samples = 120

x = rng.normal(loc=0, scale=2.0, size=n_samples)
y = 0.7 * x + rng.normal(loc=0, scale=0.9, size=n_samples)
z = 1.1 * x - 0.4 * y + rng.normal(loc=0, scale=0.35, size=n_samples)

data_3d = np.column_stack([x, y, z])


# 2. 평균 중심화
mean_3d = np.mean(data_3d, axis=0)
centered_data = data_3d - mean_3d


# 3. 공분산 행렬 산정
cov_matrix = np.cov(centered_data, rowvar=False)


# 4. 고유값, 고유벡터 계산
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)


# 5. 고유값 내림차순 정렬
sort_order = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sort_order]
eigenvectors = eigenvectors[:, sort_order]


# 6. 주성분 2개 선택
n_components = 2
principal_components = eigenvectors[:, :n_components]


# 7. 2차원 좌표로 투영
data_2d = centered_data @ principal_components


# 8. 설명 분산 비율 확인
explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
kept_variance = np.sum(explained_variance_ratio[:n_components])

print("원본 데이터 모양:", data_3d.shape)
print("PCA 후 데이터 모양:", data_2d.shape)
print("각 주성분의 설명 분산 비율:", explained_variance_ratio)
print(f"2차원으로 유지한 정보 비율: {kept_variance:.2%}")


# 9. 원본 3D와 PCA 2D 시각화
fig = plt.figure(figsize=(12, 5), constrained_layout=True)

ax_3d = fig.add_subplot(1, 2, 1, projection="3d")
ax_3d.scatter(data_3d[:, 0], data_3d[:, 1], data_3d[:, 2],
              c=data_2d[:, 0], cmap="viridis", alpha=0.8)
ax_3d.set_title("Original Data in 3D")
ax_3d.set_xlabel("X")
ax_3d.set_ylabel("Y")
ax_3d.set_zlabel("Z")

ax_2d = fig.add_subplot(1, 2, 2)
scatter = ax_2d.scatter(data_2d[:, 0], data_2d[:, 1],
                        c=data_2d[:, 0], cmap="viridis", alpha=0.8)
ax_2d.axhline(0, color="gray", linewidth=1)
ax_2d.axvline(0, color="gray", linewidth=1)
ax_2d.set_title(f"PCA Result in 2D ({kept_variance:.1%} kept)")
ax_2d.set_xlabel("Principal Component 1")
ax_2d.set_ylabel("Principal Component 2")
ax_2d.set_aspect("equal", adjustable="box")

fig.colorbar(scatter, ax=[ax_3d, ax_2d], shrink=0.75, label="PC1 value")
plt.show()
