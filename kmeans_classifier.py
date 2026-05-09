import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.model_selection import train_test_split

from preprocess_faces import prepare_dataset, natural_key, load_gray_image


class KMeansCustom:
    def __init__(self, n_clusters=28, max_iter=100, random_state=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.centroids_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = 0

    def fit(self, X):
        rng = np.random.RandomState(self.random_state)
        n_samples = X.shape[0]

        indices = rng.choice(n_samples, self.n_clusters, replace=False)
        self.centroids_ = X[indices].copy()

        for iteration in range(self.max_iter):
            distances = self._compute_distances(X, self.centroids_)
            self.labels_ = np.argmin(distances, axis=1)

            new_centroids = np.zeros_like(self.centroids_)
            for k in range(self.n_clusters):
                mask = self.labels_ == k
                if mask.sum() > 0:
                    new_centroids[k] = X[mask].mean(axis=0)
                else:
                    new_centroids[k] = X[rng.choice(n_samples)]

            if np.allclose(self.centroids_, new_centroids):
                self.n_iter_ = iteration + 1
                self.centroids_ = new_centroids
                break

            self.centroids_ = new_centroids
            self.n_iter_ = iteration + 1

        self.inertia_ = self._compute_inertia(X)
        return self

    def predict(self, X):
        if self.centroids_ is None:
            raise ValueError("Model has not been fitted yet")
        distances = self._compute_distances(X, self.centroids_)
        return np.argmin(distances, axis=1)

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_

    def _compute_distances(self, X, centroids):
        X_sq = (X ** 2).sum(axis=1, keepdims=True)
        centroids_sq = (centroids ** 2).sum(axis=1)
        cross_term = X @ centroids.T
        distances = X_sq - 2 * cross_term + centroids_sq
        return np.maximum(distances, 0.0)

    def _compute_inertia(self, X):
        distances = self._compute_distances(X, self.centroids_)
        return np.sum(np.min(distances, axis=1))


def build_label_map(y_true, y_pred_cluster):
    n_classes = int(np.max(y_true)) + 1
    n_clusters = int(np.max(y_pred_cluster)) + 1

    contingency = np.zeros((n_classes, n_clusters), dtype=np.int64)
    for yt, yp in zip(y_true, y_pred_cluster):
        contingency[int(yt), int(yp)] += 1

    row_ind, col_ind = linear_sum_assignment(-contingency)
    return {int(c): int(r) for r, c in zip(row_ind, col_ind)}


def apply_label_map(y_pred_cluster, label_map):
    return np.array([label_map.get(int(c), -1) for c in y_pred_cluster], dtype=np.int64)


def mapped_accuracy(y_true, y_pred_cluster, label_map):
    y_pred = apply_label_map(y_pred_cluster, label_map)
    return (y_true == y_pred).mean()


def show_cluster_examples(X_train, y_train, train_clusters, idx_train, image_paths, centroids, n_groups=4, n_per_group=5):
    cluster_ids, counts = np.unique(train_clusters, return_counts=True)
    order = np.argsort(counts)[::-1]
    selected_clusters = cluster_ids[order[:n_groups]]

    fig, axes = plt.subplots(
        len(selected_clusters),
        n_per_group + 1,
        figsize=(2.2 * (n_per_group + 1), 2.3 * len(selected_clusters)),
        gridspec_kw={"width_ratios": [2.6] + [1] * n_per_group},
    )

    if len(selected_clusters) == 1:
        axes = axes[np.newaxis, :]

    fig.patch.set_facecolor("white")

    # Column headers for image cells
    for col in range(1, n_per_group + 1):
        axes[0, col].set_title(f"Example {col}", fontsize=10, pad=8)

    for row, cluster_id in enumerate(selected_clusters):
        members = np.where(train_clusters == cluster_id)[0]
        centroid = centroids[int(cluster_id)]
        dists = ((X_train[members] - centroid) ** 2).sum(axis=1)
        nearest_local = members[np.argsort(dists)[:n_per_group]]

        # Row header cell (table-style)
        header_ax = axes[row, 0]
        header_ax.axis("off")
        header_text = (
            f"Cluster c{int(cluster_id)}\n"
            f"size: {len(members)}"
        )
        header_ax.text(
            0.03,
            0.5,
            header_text,
            ha="left",
            va="center",
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.45", "facecolor": "#f3f4f6", "edgecolor": "#d1d5db"},
        )

        for col in range(n_per_group):
            ax = axes[row, col + 1]
            ax.axis("off")
            if col >= len(nearest_local):
                continue

            sample_local_idx = nearest_local[col]
            global_idx = idx_train[sample_local_idx]
            img = load_gray_image(image_paths[global_idx])
            ax.imshow(img, cmap="gray")
            ax.set_title(f"p{int(y_train[sample_local_idx])}", fontsize=9, pad=4)

            # Draw soft cell border to emphasize table structure
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.8)
                spine.set_edgecolor("#d1d5db")

    fig.suptitle("K-Means Cluster Table: Nearest Training Images per Cluster", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94], w_pad=1.0, h_pad=1.1)
    out_path = "kmeans_cluster_examples.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    X, y, _ = prepare_dataset(
        folder="Train Set (Labeled)",
        fit_limit=800,
        n_components=100,
    )

    image_paths = sorted(glob.glob(os.path.join("Train Set (Labeled)", "*.pgm")), key=natural_key)
    all_indices = np.arange(X.shape[0])

    X_train, X_test, y_train, y_test, idx_train, _ = train_test_split(
        X, y, all_indices, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Train: {X_train.shape[0]}  |  Test: {X_test.shape[0]}  |  Features: {X.shape[1]}")

    kmeans = KMeansCustom(n_clusters=28, max_iter=300, random_state=42)
    kmeans.fit(X_train)

    train_clusters = kmeans.predict(X_train)
    test_clusters = kmeans.predict(X_test)

    label_map = build_label_map(y_train, train_clusters)

    train_acc = mapped_accuracy(y_train, train_clusters, label_map)
    test_acc = mapped_accuracy(y_test, test_clusters, label_map)

    train_ari = adjusted_rand_score(y_train, train_clusters)
    test_ari = adjusted_rand_score(y_test, test_clusters)
    train_nmi = normalized_mutual_info_score(y_train, train_clusters)
    test_nmi = normalized_mutual_info_score(y_test, test_clusters)

    print("=" * 60)
    print(f"Converged in: {kmeans.n_iter_} iterations")
    print(f"Train inertia: {kmeans.inertia_:.2f}")
    print("-" * 60)
    print(f"Train mapped accuracy: {train_acc * 100:.2f}%")
    print(f"Test  mapped accuracy: {test_acc * 100:.2f}%")
    print(f"Train ARI: {train_ari:.4f}  |  Test ARI: {test_ari:.4f}")
    print(f"Train NMI: {train_nmi:.4f}  |  Test NMI: {test_nmi:.4f}")
    print("=" * 60)

    image_out = show_cluster_examples(
        X_train,
        y_train,
        train_clusters,
        idx_train,
        image_paths,
        kmeans.centroids_,
        n_groups=4,
        n_per_group=5,
    )
    print(f"Saved cluster image grid to: {image_out}")
