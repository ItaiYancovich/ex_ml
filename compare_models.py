from collections import Counter

import numpy as np
from sklearn.model_selection import train_test_split

from preprocess_faces import prepare_dataset
from knn_classifier import KNNClassifier
from svm_classifier import SVMClassifier
from softmax_classifier import SoftmaxRegression
from k_fold_validation import main as kfold_main


models = [
    (
        "KNN",
        KNNClassifier,
        [
            {"k": 1,  "p": 2},
            {"k": 3,  "p": 2},
            {"k": 5,  "p": 2},
            {"k": 7, "p": 2},
            {"k": 9,  "p": 2},
            {"k": 11,  "p": 2},
        ],
    ),
    (
        "SVM",
        SVMClassifier,
        [
            {"kernel": "linear", "C": 0.1,  "class_weight": "balanced"},
            {"kernel": "linear", "C": 1.0,  "class_weight": "balanced"},
            {"kernel": "linear", "C": 10.0, "class_weight": "balanced"},
            {"kernel": "rbf",    "C": 1.0,  "gamma": "scale", "class_weight": "balanced"},
            {"kernel": "rbf",    "C": 5.0,  "gamma": "scale", "class_weight": "balanced"},
            {"kernel": "rbf",    "C": 10.0, "gamma": "scale", "class_weight": "balanced"},
        ],
    ),
    (
        "Softmax",
        SoftmaxRegression,
        [
            {"lr": 0.01,  "reg": 1e-4, "max_iter": 500, "batch_size": 64},
            {"lr": 0.01,  "reg": 1e-3, "max_iter": 500, "batch_size": 64},
            {"lr": 0.001, "reg": 1e-4, "max_iter": 500, "batch_size": 64},
            {"lr": 0.005, "reg": 1e-4, "max_iter": 500, "batch_size": 128},
        ],
    ),
]


def main():
    X, y, _ = prepare_dataset(
        folder="Train Set (Labeled)",
        fit_limit=800,
        n_components=225,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Train: {X_train.shape[0]}  |  Test: {X_test.shape[0]}"
          f"  |  Classes: {np.unique(y).size}  |  Features: {X.shape[1]}\n")

    best_name, results, test_predictions = kfold_main(
        models,
        X_train,
        y_train,
        X_test=X_test,
        k_outer=5,
        k_inner=3,
    )

    print("=" * 60)
    print(f"{'Model':<12}  {'Avg CV Acc':>10}  {'Variance':>10}  Best HPs (most common)")
    print("-" * 60)

    for name, res in results.items():
        avg = res["avg_score"]
        var = res["variance"]
        hps = res["selected_hyperparameters"]

        tuples = [tuple(sorted(d.items())) for d in hps]
        most_common = dict(Counter(tuples).most_common(1)[0][0])
        print(f"{name:<12}  {avg * 100:>9.2f}%  {var:>10.6f}  {most_common}")

    print("=" * 60)
    print(f"\nBest model: {best_name}")

    if test_predictions is not None:
        final_acc = (test_predictions == y_test).mean()
        print(f"Hold-out test accuracy ({best_name}): {final_acc * 100:.2f}%")


if __name__ == "__main__":
    main()
