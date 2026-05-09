import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

from preprocess_faces import prepare_dataset


# ---------------------------------------------------------------------------
# SVM
# ---------------------------------------------------------------------------

class SVMClassifier:
    """sklearn-compatible wrapper around sklearn's SVC."""

    def __init__(self, kernel="rbf", C=1.0, gamma="scale", class_weight="balanced", random_state=42):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.class_weight = class_weight
        self.random_state = random_state
        self._clf = None

    def fit(self, X, y):
        self._clf = SVC(
            kernel=self.kernel,
            C=self.C,
            gamma=self.gamma,
            class_weight=self.class_weight,
            random_state=self.random_state,
        )
        self._clf.fit(X, y)
        return self

    def predict(self, X):
        return self._clf.predict(X)


if __name__ == "__main__":
    X, y, _ = prepare_dataset(
        folder="Train Set (Labeled)",
        fit_limit=800,
        n_components=123,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain: {X_train.shape[0]}  |  Test: {X_test.shape[0]}  |  Classes: {np.unique(y).size}  |  Features: {X.shape[1]}\n")

    # -----------------------------------------------------------------------
    # Hyperparameter grid
    # -----------------------------------------------------------------------
    configs = [
        # kernel   C       gamma
        ("linear", 0.1,    None),
        ("linear", 1.0,    None),
        ("linear", 10.0,   None),
        ("rbf",    0.1,    "scale"),
        ("rbf",    1.0,    "scale"),
        ("rbf",    5.0,    "scale"),
        ("rbf",    10.0,   "scale"),
        ("rbf",    50.0,   "scale"),
        ("rbf",    1.0,    "auto"),
        ("rbf",    5.0,    "auto"),
        ("poly",   1.0,    "scale"),
        ("poly",   5.0,    "scale"),
        ("poly",   10.0,   "scale"),
    ]

    results = []
    total = len(configs)
    for i, (kernel, C, gamma) in enumerate(configs, 1):
        label = f"kernel={kernel:<6}  C={C:<6}  gamma={str(gamma):<6}"
        print(f"[{i:>2}/{total}]  {label} ... ", end="", flush=True)

        kwargs = dict(kernel=kernel, C=C, class_weight="balanced", random_state=42)
        if gamma is not None:
            kwargs["gamma"] = gamma

        clf = SVC(**kwargs)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results.append((acc, label, clf, y_pred))
        print(f"accuracy: {acc * 100:.2f}%")

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    results.sort(key=lambda r: r[0], reverse=True)
    print("\n" + "=" * 56)
    print(f"{'Rank':<5}  {'Accuracy':>9}  Config")
    print("-" * 56)
    for rank, (acc, label, _, _) in enumerate(results, 1):
        marker = " <-- best" if rank == 1 else ""
        print(f"{rank:<5}  {acc * 100:>8.2f}%  {label}{marker}")
    print("=" * 56)

    # -----------------------------------------------------------------------
    # Detailed report for best config
    # -----------------------------------------------------------------------
    best_acc, best_label, best_clf, best_pred = results[0]
    print(f"\n=== Detailed report for best config: {best_label} ===\n")
    print(classification_report(y_test, best_pred, zero_division=0))
    print(f"Overall accuracy: {best_acc * 100:.2f}%")
