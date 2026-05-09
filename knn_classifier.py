import numpy as np
from sklearn.model_selection import train_test_split

from preprocess_faces import prepare_dataset


class KNNClassifier:
    def __init__(self, k=5, p=2):
        self.k = k
        self.p = p

    def fit(self, X, y):
        self._X_train = X
        self._y_train = y
        return self

    def predict(self, X):
        return knn_predict(self._X_train, self._y_train, X, k=self.k, p=self.p)


def knn_predict(X_train, y_train, X_test, k=5, p=2):
    n_test = X_test.shape[0]
    n_train = X_train.shape[0]
    predictions = np.empty(n_test, dtype=y_train.dtype)

    batch = max(1, 512 * 512 // max(n_train, 1))
    train_sq = (X_train ** 2).sum(axis=1) if p == 2 else None

    for start in range(0, n_test, batch):
        end = min(start + batch, n_test)
        chunk = X_test[start:end]

        if p == 2:
            test_sq = (chunk ** 2).sum(axis=1, keepdims=True)
            dists = np.maximum(test_sq - 2.0 * (chunk @ X_train.T) + train_sq, 0.0)
        else:
            diff = np.abs(chunk[:, np.newaxis, :] - X_train[np.newaxis, :, :])
            dists = (diff ** p).sum(axis=2)

        nn_idx = np.argpartition(dists, kth=min(k, n_train - 1), axis=1)[:, :k]

        for i, neighbours in enumerate(nn_idx):
            votes = y_train[neighbours]
            counts = np.bincount(votes, minlength=int(y_train.max()) + 1)
            predictions[start + i] = counts.argmax()

    return predictions


def accuracy(y_true, y_pred):
    return (y_true == y_pred).mean()


def classification_report(y_true, y_pred):
    labels = np.unique(y_true)
    lines = [f"{'Person':>8}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}  {'Support':>8}"]
    lines.append("-" * 46)
    macro_p, macro_r, macro_f = [], [], []

    for lbl in labels:
        tp = ((y_pred == lbl) & (y_true == lbl)).sum()
        fp = ((y_pred == lbl) & (y_true != lbl)).sum()
        fn = ((y_pred != lbl) & (y_true == lbl)).sum()
        support = (y_true == lbl).sum()

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        macro_p.append(prec)
        macro_r.append(rec)
        macro_f.append(f1)
        lines.append(f"p{lbl:>6}  {prec:>6.3f}  {rec:>6.3f}  {f1:>6.3f}  {support:>8}")

    lines.append("-" * 46)
    lines.append(
        f"{'macro avg':>8}  {np.mean(macro_p):>6.3f}  {np.mean(macro_r):>6.3f}"
        f"  {np.mean(macro_f):>6.3f}  {len(y_true):>8}"
    )
    return "\n".join(lines)


if __name__ == "__main__":
    X, y, _ = prepare_dataset(
        folder="Train Set (Labeled)",
        fit_limit=800,
        n_components=225,
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"\nTrain: {X_train.shape[0]} samples  |  Test: {X_test.shape[0]} samples")
    print(f"Classes: {np.unique(y).size}  |  Features: {X.shape[1]}\n")

    best_acc, best_k, best_p = 0.0, 1, 2

    for p, dist_name in [(1, "L1 (Manhattan)"), (2, "L2 (Euclidean)"), (3, "L3 (Minkowski p=3)")]:
        print(f"\n--- {dist_name} ---")
        for k in [1, 3, 5, 11]:
            print(f"  k={k:>2} ", end="", flush=True)
            y_pred = knn_predict(X_train, y_train, X_test, k=k, p=p)
            acc = accuracy(y_test, y_pred)
            print(f"accuracy: {acc * 100:.2f}%")
            if acc > best_acc:
                best_acc, best_k, best_p = acc, k, p

    dist_names = {1: "L1", 2: "L2", 3: "L3"}
    print(f"\n=== Best: {dist_names[best_p]}  k={best_k}  ({best_acc * 100:.2f}%) ===")
    print(f"=== Detailed report  ({dist_names[best_p]}, k={best_k}) ===\n")
    y_pred_best = knn_predict(X_train, y_train, X_test, k=best_k, p=best_p)
    print(classification_report(y_test, y_pred_best))
    print(f"\nOverall accuracy: {accuracy(y_test, y_pred_best) * 100:.2f}%")
