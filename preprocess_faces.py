import glob
import os
import re
from collections import Counter

import matplotlib.image as mpimg
import numpy as np


TRAIN_FOLDER = "Train Set (Labeled)"


def natural_key(path):
    name = os.path.basename(path)
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", name)]


def load_gray_image(path):
    image = mpimg.imread(path)
    if image.ndim == 3:
        image = image[..., :3].mean(axis=2)

    image = image.astype(np.float32)
    if image.max() <= 1.0:
        image *= 255.0
    return image


def person_id(path):
    match = re.match(r"(p\d+)_i", os.path.basename(path))
    return match.group(1) if match else None


def find_gray_side_rectangles(image, gray_value=127.0, tolerance=2.0, std_tolerance=0.75):
    column_mean = image.mean(axis=0)
    column_std = image.std(axis=0)
    gray_columns = (np.abs(column_mean - gray_value) <= tolerance) & (column_std <= std_tolerance)

    left = 0
    while left < image.shape[1] and gray_columns[left]:
        left += 1

    right = 0
    while right < image.shape[1] - left and gray_columns[-1 - right]:
        right += 1

    max_side_width = image.shape[1] // 4
    if left > max_side_width:
        left = 0
    if right > max_side_width:
        right = 0

    return left, right


def remove_gray_side_rectangles(image):
    left, right = find_gray_side_rectangles(image)
    end = image.shape[1] - right if right else image.shape[1]
    return image[:, left:end], left, right


def clahe(image, tile_grid_size=(8, 8), clip_limit=2.0):
    from skimage.exposure import equalize_adapthist
    normalized = np.clip(image, 0.0, 255.0) / 255.0
    equalized = equalize_adapthist(normalized, kernel_size=tile_grid_size, clip_limit=clip_limit / 100.0, nbins=256)
    return (equalized * 255.0).astype(np.float32)


def compute_hog(image, cell_size=8, n_bins=9, block_size=2):
    from skimage.feature import hog as sk_hog
    features = sk_hog(
        image / 255.0,
        orientations=n_bins,
        pixels_per_cell=(cell_size, cell_size),
        cells_per_block=(block_size, block_size),
        block_norm="L2-Hys",
        visualize=False,
        feature_vector=True,
    )
    return features.astype(np.float32)


def hog_batch(images, cell_size=8, n_bins=9, block_size=2):
    features = []
    for image in images:
        features.append(compute_hog(image, cell_size=cell_size, n_bins=n_bins, block_size=block_size))
    return np.stack(features).astype(np.float32)


def fit_standard_scaler(matrix):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(matrix)
    std = scaler.scale_.astype(np.float32)
    std[std < 1e-6] = 1.0
    return scaler.mean_.astype(np.float32), std


def standard_scale(matrix, mean, std):
    return (matrix - mean) / std


def fit_pca(matrix, n_components):
    from sklearn.decomposition import PCA
    n = min(n_components, min(matrix.shape))
    pca = PCA(n_components=n, svd_solver="randomized", random_state=42)
    pca.fit(matrix)
    return (
        pca.mean_.astype(np.float32),
        pca.components_.astype(np.float32),
        pca.explained_variance_ratio_.astype(np.float32),
    )


def apply_pipeline(images, pipeline_mean, pipeline_std, pca_mean, pca_components):
    hog_matrix = hog_batch(images)
    scaled = standard_scale(hog_matrix, pipeline_mean, pipeline_std)
    scores = (scaled - pca_mean) @ pca_components.T
    return scores


def build_dataset(folder, pipeline_mean, pipeline_std, pca_mean, pca_components, common_shape):
    paths = sorted(glob.glob(os.path.join(folder, "*.pgm")), key=natural_key)
    kept_images, kept_labels = [], []
    total = len(paths)
    for i, path in enumerate(paths):
        if (i + 1) % max(1, total // 10) == 0 or (i + 1) == total:
            print(f"  loading+CLAHE {i + 1}/{total}", end="\r", flush=True)
        image = load_gray_image(path)
        cropped, _, _ = remove_gray_side_rectangles(image)
        if cropped.shape != common_shape:
            continue
        kept_images.append(clahe(cropped))
        pid = person_id(path)
        kept_labels.append(int(pid[1:]) if pid else -1)
    print()
    print(f"  Computing HOG for {len(kept_images)} images...")
    X = apply_pipeline(kept_images, pipeline_mean, pipeline_std, pca_mean, pca_components)
    y = np.array(kept_labels, dtype=np.int32)
    return X, y


def prepare_dataset(folder=TRAIN_FOLDER, fit_limit=800, n_components=225, cache_file="dataset_cache.npz"):
    if cache_file and os.path.exists(cache_file):
        print(f"Loading cached dataset from {cache_file!r}...")
        data = np.load(cache_file)
        X, y = data["X"], data["y"]
        pipeline = {
            "pipeline_mean": data["pipeline_mean"],
            "pipeline_std":  data["pipeline_std"],
            "pca_mean":      data["pca_mean"],
            "pca_components":data["pca_components"],
            "common_shape":  tuple(data["common_shape"]),
        }
        print(f"Loaded: {X.shape[0]} images, {X.shape[1]} features, {np.unique(y).size} people.")
        return X, y, pipeline

    paths = sorted(glob.glob(os.path.join(folder, "*.pgm")), key=natural_key)
    if not paths:
        raise FileNotFoundError(f"No .pgm images found in {folder!r}")

    fit_paths = load_fit_paths(paths, fit_limit)
    print(f"[1/7] Found {len(paths)} images total. Using {len(fit_paths)} to fit pipeline.")

    print(f"[2/7] Loading and cropping {len(fit_paths)} fit images...")
    fit_raw = [load_gray_image(p) for p in fit_paths]
    fit_cropped, fit_crop_paths = [], []
    for path, img in zip(fit_paths, fit_raw):
        cropped, *_ = remove_gray_side_rectangles(img)
        fit_cropped.append(cropped)
        fit_crop_paths.append(path)

    fit_crop_paths, fit_cropped, common_shape, _ = keep_common_shape(fit_crop_paths, fit_cropped)
    print(f"[3/7] Applying CLAHE to {len(fit_cropped)} fit images (shape {common_shape})...")
    fit_clahe = [clahe(img) for img in fit_cropped]
    print(f"[4/7] Computing HOG features for {len(fit_clahe)} fit images...")
    fit_hog = hog_batch(fit_clahe)
    print(f"[5/7] Fitting scaler and PCA ({n_components} components) on {fit_hog.shape[0]} HOG vectors ({fit_hog.shape[1]} features)...")
    pipeline_mean, pipeline_std = fit_standard_scaler(fit_hog)
    fit_scaled = standard_scale(fit_hog, pipeline_mean, pipeline_std)
    pca_mean, pca_components, _ = fit_pca(fit_scaled, n_components)

    print(f"[6/7] Transforming all {len(paths)} images...")
    X, y = build_dataset(folder, pipeline_mean, pipeline_std, pca_mean, pca_components, common_shape)
    print(f"[7/7] Done. Dataset: {X.shape[0]} images, {X.shape[1]} features, {np.unique(y).size} people.")

    pipeline = {
        "pipeline_mean":  pipeline_mean,
        "pipeline_std":   pipeline_std,
        "pca_mean":       pca_mean,
        "pca_components": pca_components,
        "common_shape":   np.array(common_shape),
    }

    if cache_file:
        np.savez_compressed(cache_file, X=X, y=y, **pipeline)
        print(f"Dataset and pipeline parameters saved to {cache_file!r}.")

    pipeline["common_shape"] = common_shape
    return X, y, pipeline


def keep_common_shape(paths, images):
    shape_counts = Counter(image.shape for image in images)
    common_shape, _ = shape_counts.most_common(1)[0]
    kept_paths = []
    kept_images = []
    for path, image in zip(paths, images):
        if image.shape == common_shape:
            kept_paths.append(path)
            kept_images.append(image)
    return kept_paths, kept_images, common_shape, shape_counts


def load_fit_paths(paths, fit_limit):
    if fit_limit <= 0:
        return paths
    return paths[:fit_limit]
