import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

from preprocess_faces import prepare_dataset


# ---------------------------------------------------------------------------
# Softmax Regression
# ---------------------------------------------------------------------------

class SoftmaxRegression:
    """
    Multi-class logistic (softmax) regression trained with mini-batch
    gradient descent and L2 regularisation.

    Parameters
    ----------
    lr          : learning rate
    reg         : L2 regularisation strength (lambda)
    max_iter    : number of full passes over the training set (epochs)
    batch_size  : mini-batch size  (None → full-batch gradient descent)
    random_state: seed for weight initialisation and batch shuffling
    """

    def __init__(
        self,
        lr=0.01,
        reg=1e-4,
        max_iter=3000,
        batch_size=64,
        random_state=42,
    ):
        self.lr           = lr
        self.reg          = reg
        self.max_iter     = max_iter
        self.batch_size   = batch_size
        self.random_state = random_state

        self.W      = None   # (n_features, n_classes)
        self.b      = None   # (n_classes,)
        self.classes_ = None

    @staticmethod
    def _softmax(Z):
        Z_shifted = Z - Z.max(axis=1, keepdims=True)
        exp_Z = np.exp(Z_shifted)
        return exp_Z / exp_Z.sum(axis=1, keepdims=True)

    @staticmethod
    def _one_hot(y, n_classes):
        """Convert integer label vector to one-hot matrix."""
        n = len(y)
        Y = np.zeros((n, n_classes), dtype=np.float64)
        Y[np.arange(n), y] = 1.0
        return Y

    def _cross_entropy_loss(self, P, Y_oh):
        """Mean cross-entropy + L2 penalty on W (not on bias)."""
        n = Y_oh.shape[0]
        # Clip to avoid log(0)
        log_likelihood = -np.sum(Y_oh * np.log(np.clip(P, 1e-15, 1.0))) / n
        l2_penalty     = 0.5 * self.reg * np.sum(self.W ** 2)
        return log_likelihood + l2_penalty


    def fit(self, X, y, X_val=None, y_val=None, verbose=False):
        """Train on (X, y).  y must be integer class labels 0..K-1."""
        rng = np.random.default_rng(self.random_state)
        self.val_acc_history_ = []

        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        # Map labels to 0..K-1
        label_to_idx = {lbl: i for i, lbl in enumerate(self.classes_)}
        y_idx        = np.array([label_to_idx[lbl] for lbl in y])

        # Xavier / Glorot initialisation
        scale   = np.sqrt(2.0 / (n_features + n_classes))
        self.W  = rng.normal(0.0, scale, (n_features, n_classes))
        self.b  = np.zeros(n_classes)

        bs = self.batch_size if self.batch_size is not None else n_samples

        for epoch in range(self.max_iter):
            # Shuffle training set each epoch
            perm = rng.permutation(n_samples)
            X_s, y_s = X[perm], y_idx[perm]

            for start in range(0, n_samples, bs):
                Xb = X_s[start : start + bs]        # (B, d)
                yb = y_s[start : start + bs]        # (B,)

                B   = Xb.shape[0]
                Z   = Xb @ self.W + self.b           # (B, K)
                P   = self._softmax(Z)               # (B, K)
                Y_oh = self._one_hot(yb, n_classes)  # (B, K)

                dZ   = (P - Y_oh) / B               # (B, K)
                dW   = Xb.T @ dZ + self.reg * self.W # (d, K)
                db   = dZ.sum(axis=0)               # (K,)

                self.W -= self.lr * dW
                self.b -= self.lr * db

            if X_val is not None:
                self.val_acc_history_.append((self.predict(X_val) == y_val).mean())

            if verbose and (epoch + 1) % 50 == 0:
                Z_full = X @ self.W + self.b
                P_full = self._softmax(Z_full)
                loss   = self._cross_entropy_loss(P_full, self._one_hot(y_idx, n_classes))
                print(f"  epoch {epoch + 1:>4}/{self.max_iter}  loss={loss:.4f}")

        return self

    def predict_proba(self, X):
        """Return softmax probability matrix of shape (n_samples, n_classes)."""
        Z = X @ self.W + self.b
        return self._softmax(Z)

    def predict(self, X):
        """Return predicted integer class labels."""
        proba      = self.predict_proba(X)
        idx        = proba.argmax(axis=1)
        return self.classes_[idx]



def accuracy(y_true, y_pred):
    return (y_true == y_pred).mean()


def classification_report(y_true, y_pred):
    labels = np.unique(y_true)
    lines  = [f"{'Person':>8}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}  {'Support':>8}"]
    lines.append("-" * 46)
    macro_p, macro_r, macro_f = [], [], []

    for lbl in labels:
        tp = ((y_pred == lbl) & (y_true == lbl)).sum()
        fp = ((y_pred == lbl) & (y_true != lbl)).sum()
        fn = ((y_pred != lbl) & (y_true == lbl)).sum()
        support = (y_true == lbl).sum()

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    X, y, _ = prepare_dataset(
        folder="Train Set (Labeled)",
        fit_limit=800,
        n_components=123,
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"\nTrain: {X_train.shape[0]}  |  Test: {X_test.shape[0]}"
          f"  |  Classes: {np.unique(y).size}  |  Features: {X.shape[1]}\n")

    # -----------------------------------------------------------------------
    # Hyperparameter grid
    # -----------------------------------------------------------------------
    configs = [
        # lr       reg      max_iter  batch_size
        (0.1,    1e-4,   1000,   64),
        (0.1,    1e-3,   1000,   64),
        (0.1,    1e-2,   1000,   64),
        (0.05,   1e-4,   1000,   64),
        (0.05,   1e-3,   1000,   64),
        (0.01,   1e-4,   1000,   64),
        (0.01,   1e-3,   1000,   64),
        (0.1,    1e-4,   1000,  128),
        (0.1,    1e-4,   1000,  None),   # full-batch
    ]

    results = []
    total = len(configs)
    for i, (lr, reg, max_iter, bs) in enumerate(configs, 1):
        label = f"lr={lr:<5}  reg={reg:<6}  iter={max_iter:<4}  bs={str(bs):<4}"
        print(f"[{i:>2}/{total}]  {label} ... ", end="", flush=True)

        clf = SoftmaxRegression(
            lr=lr, reg=reg, max_iter=max_iter,
            batch_size=bs, random_state=42,
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc    = accuracy(y_test, y_pred)
        results.append((acc, label, clf, y_pred))
        print(f"accuracy: {acc * 100:.2f}%")

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    results.sort(key=lambda r: r[0], reverse=True)
    print("\n" + "=" * 60)
    print(f"{'Rank':<5}  {'Accuracy':>9}  Config")
    print("-" * 60)
    for rank, (acc, label, _, _) in enumerate(results, 1):
        marker = " <-- best" if rank == 1 else ""
        print(f"{rank:<5}  {acc * 100:>8.2f}%  {label}{marker}")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # Detailed report for best config
    # -----------------------------------------------------------------------
    best_acc, best_label, best_clf, best_pred = results[0]
    print(f"\n=== Detailed report for best config: {best_label} ===\n")
    print(classification_report(y_test, best_pred))
    print(f"\nOverall accuracy: {best_acc * 100:.2f}%")

    # -----------------------------------------------------------------------
    # Plot test accuracy vs epochs for the best config
    # -----------------------------------------------------------------------
    print("\nRecording per-epoch test accuracy for best config...")
    clf_plot = SoftmaxRegression(
        lr=best_clf.lr, reg=best_clf.reg, max_iter=best_clf.max_iter,
        batch_size=best_clf.batch_size, random_state=42,
    )
    clf_plot.fit(X_train, y_train, X_val=X_test, y_val=y_test)

    epochs = range(1, best_clf.max_iter + 1)
    plt.figure(figsize=(9, 5))
    plt.plot(epochs, [v * 100 for v in clf_plot.val_acc_history_], linewidth=1.5)
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy (%)")
    plt.title(f"Test Accuracy vs Epochs\n{best_label}")
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()
