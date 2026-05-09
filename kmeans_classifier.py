import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.model_selection import train_test_split

from preprocess_faces import prepare_dataset


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


if __name__ == "__main__":
    X, y, _ = prepare_dataset(
        folder="Train Set (Labeled)",
        fit_limit=800,
        n_components=100,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
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

