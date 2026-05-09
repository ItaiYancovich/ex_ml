import glob
import os
import re

import numpy as np
from sklearn.svm import SVC

from preprocess_faces import (
    load_gray_image,
    remove_gray_side_rectangles,
    clahe,
    apply_pipeline,
    prepare_dataset,
)


TRAIN_FOLDER = "Train Set (Labeled)"
TEST_FOLDER = os.path.join("Test Set (Unlabeled)-20260509T123733Z-3-001", "Test Set (Unlabeled)")
OUTPUT_FILE = "submission.csv"


def pic_key(path):
    name = os.path.basename(path)
    match = re.search(r"(\d+)", name)
    return int(match.group(1)) if match else 0


def load_test_features(test_folder, pipeline):
    test_paths = sorted(glob.glob(os.path.join(test_folder, "*.pgm")), key=pic_key)
    print(f"\nFound {len(test_paths)} test images.")

    common_shape = pipeline["common_shape"]
    kept_images = []

    for path in test_paths:
        image = load_gray_image(path)
        cropped, _, _ = remove_gray_side_rectangles(image)

        if cropped.shape != common_shape:
            print(f"  WARNING: {os.path.basename(path)} has shape {cropped.shape}, expected {common_shape}; skipping")
            continue

        kept_images.append(clahe(cropped))

    print(f"Kept {len(kept_images)} images after shape filtering.")
    print("Computing HOG + PCA features for test images...")
    return apply_pipeline(
        kept_images,
        pipeline["pipeline_mean"],
        pipeline["pipeline_std"],
        pipeline["pca_mean"],
        pipeline["pca_components"],
    )


def main():
    print("Loading training dataset and pipeline...")
    X_train, y_train, pipeline = prepare_dataset(
        folder=TRAIN_FOLDER,
        fit_limit=800,
        n_components=225,
    )
    print(f"Training set: {X_train.shape[0]} images, {X_train.shape[1]} features, {np.unique(y_train).size} classes")

    print("\nTraining SVM (kernel=rbf, C=5.0, gamma='scale') on full training set...")
    clf = SVC(kernel="rbf", C=5.0, gamma="scale", class_weight="balanced", random_state=42)
    clf.fit(X_train, y_train)
    print("Training complete.")

    X_test = load_test_features(TEST_FOLDER, pipeline)

    print("Predicting...")
    predictions = clf.predict(X_test)

    with open(OUTPUT_FILE, "w") as f:
        f.write(",".join(str(p) for p in predictions) + ",")

    print(f"\nPredictions written to {OUTPUT_FILE!r}")
    print(f"Total predictions: {len(predictions)}")
    print(f"Sample (first 20): {list(predictions[:20])}")


if __name__ == "__main__":
    main()
