import argparse
import glob
import os
import re
from collections import Counter

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
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


def select_different_people(paths, sample_count):
    selected = []
    seen = set()
    for path in sorted(paths, key=natural_key):
        current_person = person_id(path)
        if current_person is None or current_person in seen:
            continue
        selected.append(path)
        seen.add(current_person)
        if len(selected) == sample_count:
            break
    return selected


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


def equalize_tile(tile, clip_limit=2.0, bins=256):
    tile_u8 = np.clip(np.rint(tile), 0, 255).astype(np.uint8)
    histogram = np.bincount(tile_u8.ravel(), minlength=bins).astype(np.float32)

    clip_value = max(1.0, clip_limit * tile_u8.size / bins)
    excess = np.maximum(histogram - clip_value, 0.0).sum()
    histogram = np.minimum(histogram, clip_value)
    histogram += excess / bins

    cdf = histogram.cumsum()
    cdf_min = cdf[np.flatnonzero(cdf)[0]] if np.any(cdf) else 0.0
    denominator = cdf[-1] - cdf_min
    if denominator <= 0:
        return np.arange(bins, dtype=np.float32)

    return np.clip((cdf - cdf_min) / denominator * 255.0, 0, 255)


def clahe(image, tile_grid_size=(8, 8), clip_limit=2.0):
    image_u8 = np.clip(np.rint(image), 0, 255).astype(np.uint8)
    height, width = image_u8.shape
    grid_rows, grid_cols = tile_grid_size

    y_edges = np.linspace(0, height, grid_rows + 1, dtype=int)
    x_edges = np.linspace(0, width, grid_cols + 1, dtype=int)
    y_centers = (y_edges[:-1] + y_edges[1:] - 1) / 2.0
    x_centers = (x_edges[:-1] + x_edges[1:] - 1) / 2.0

    luts = np.empty((grid_rows, grid_cols, 256), dtype=np.float32)
    for row in range(grid_rows):
        for col in range(grid_cols):
            tile = image_u8[y_edges[row] : y_edges[row + 1], x_edges[col] : x_edges[col + 1]]
            luts[row, col] = equalize_tile(tile, clip_limit=clip_limit)

    output = np.empty_like(image_u8, dtype=np.float32)
    for y in range(height):
        bottom = np.searchsorted(y_centers, y, side="right")
        top = max(0, bottom - 1)
        bottom = min(grid_rows - 1, bottom)
        if top == bottom:
            y_weight = 0.0
        else:
            y_weight = (y - y_centers[top]) / (y_centers[bottom] - y_centers[top])

        for x in range(width):
            right = np.searchsorted(x_centers, x, side="right")
            left = max(0, right - 1)
            right = min(grid_cols - 1, right)
            if left == right:
                x_weight = 0.0
            else:
                x_weight = (x - x_centers[left]) / (x_centers[right] - x_centers[left])

            value = image_u8[y, x]
            top_value = (1.0 - x_weight) * luts[top, left, value] + x_weight * luts[top, right, value]
            bottom_value = (1.0 - x_weight) * luts[bottom, left, value] + x_weight * luts[bottom, right, value]
            output[y, x] = (1.0 - y_weight) * top_value + y_weight * bottom_value

    return output


def flatten_images(images):
    return np.stack([image.ravel() for image in images]).astype(np.float32)


# ---------------------------------------------------------------------------
# HOG (Histogram of Oriented Gradients)
# ---------------------------------------------------------------------------

def compute_hog(image, cell_size=8, n_bins=9, block_size=2):
    """
    Compute HOG feature vector for a single grayscale image (float32, 0-255).

    Steps:
      1. Central-difference gradients (Gx, Gy).
      2. Magnitude and unsigned angle (0-180 deg) per pixel.
      3. Soft bilinear bin assignment → per-cell histograms.
      4. L2 block normalisation over (block_size * block_size) cell windows.

    Returns
    -------
    feature_vector : 1-D float32 array of length n_blocks_y*n_blocks_x*block_size^2*n_bins
    cell_hists     : float32 array (n_cells_y, n_cells_x, n_bins) — raw, un-normalised
    """
    image_f = image.astype(np.float32)
    gx = np.zeros_like(image_f)
    gy = np.zeros_like(image_f)
    gx[:, 1:-1] = image_f[:, 2:] - image_f[:, :-2]
    gy[1:-1, :] = image_f[2:, :] - image_f[:-2, :]

    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    angle = np.rad2deg(np.arctan2(gy, gx)) % 180.0  # unsigned 0-180

    height, width = image_f.shape
    n_cells_y = height // cell_size
    n_cells_x = width // cell_size

    bin_width = 180.0 / n_bins
    bin_idx = angle / bin_width - 0.5           # fractional bin index
    bin_lo = np.floor(bin_idx).astype(np.int32) % n_bins
    bin_hi = (bin_lo + 1) % n_bins
    weight_hi = (bin_idx - np.floor(bin_idx)).astype(np.float32)
    weight_lo = 1.0 - weight_hi

    cell_hists = np.zeros((n_cells_y, n_cells_x, n_bins), dtype=np.float32)
    for cy in range(n_cells_y):
        for cx in range(n_cells_x):
            y0, y1 = cy * cell_size, (cy + 1) * cell_size
            x0, x1 = cx * cell_size, (cx + 1) * cell_size
            mag  = magnitude[y0:y1, x0:x1].ravel()
            lo   = bin_lo[y0:y1, x0:x1].ravel()
            hi   = bin_hi[y0:y1, x0:x1].ravel()
            wlo  = weight_lo[y0:y1, x0:x1].ravel()
            whi  = weight_hi[y0:y1, x0:x1].ravel()
            np.add.at(cell_hists[cy, cx], lo, wlo * mag)
            np.add.at(cell_hists[cy, cx], hi, whi * mag)

    # L2 block normalisation
    n_blocks_y = n_cells_y - block_size + 1
    n_blocks_x = n_cells_x - block_size + 1
    eps = 1e-6
    blocks = []
    for by in range(n_blocks_y):
        for bx in range(n_blocks_x):
            block = cell_hists[by : by + block_size, bx : bx + block_size, :].ravel()
            blocks.append(block / np.sqrt(np.dot(block, block) + eps * eps))

    return np.concatenate(blocks).astype(np.float32), cell_hists


def hog_visualization(cell_hists, cell_size=8, n_bins=9):
    """
    Render HOG cell histograms as a grayscale image.
    In each cell, draw oriented lines whose brightness is proportional to that
    bin's magnitude — visually shows which edge directions dominate.
    """
    n_cells_y, n_cells_x, _ = cell_hists.shape
    vis = np.zeros((n_cells_y * cell_size, n_cells_x * cell_size), dtype=np.float32)
    angles = np.arange(n_bins) * np.pi / n_bins  # center angle of each bin

    for cy in range(n_cells_y):
        for cx in range(n_cells_x):
            hist = cell_hists[cy, cx]
            peak = hist.max()
            if peak < 1e-8:
                continue
            center_y = cy * cell_size + cell_size / 2.0
            center_x = cx * cell_size + cell_size / 2.0
            half = cell_size / 2.0 - 1.0
            for b in range(n_bins):
                weight = hist[b] / peak
                if weight < 0.01:
                    continue
                cos_a = np.cos(angles[b])
                sin_a = np.sin(angles[b])
                for t in np.linspace(-half, half, cell_size * 3):
                    py = int(round(center_y + t * sin_a))
                    px = int(round(center_x + t * cos_a))
                    if 0 <= py < vis.shape[0] and 0 <= px < vis.shape[1]:
                        if weight > vis[py, px]:
                            vis[py, px] = weight

    return vis * 255.0


def hog_batch(images, cell_size=8, n_bins=9, block_size=2):
    """Compute HOG for a list of images. Returns (feature_matrix, viz_images)."""
    features, vizs = [], []
    for image in images:
        feat, cell_hists = compute_hog(image, cell_size=cell_size, n_bins=n_bins, block_size=block_size)
        features.append(feat)
        vizs.append(hog_visualization(cell_hists, cell_size=cell_size, n_bins=n_bins))
    return np.stack(features).astype(np.float32), vizs


def hog_vector_to_display(vector):
    """Reshape a 1-D HOG (or scaled/reconstructed HOG) vector to a square image for display."""
    n = len(vector)
    side = int(np.ceil(np.sqrt(n)))
    padded = np.zeros(side * side, dtype=np.float32)
    padded[:n] = vector.astype(np.float32)
    return normalize_for_display(padded.reshape(side, side))


# ---------------------------------------------------------------------------

def fit_standard_scaler(matrix):
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    std[std < 1e-6] = 1.0
    return mean, std


def standard_scale(matrix, mean, std):
    return (matrix - mean) / std


def fit_pca(matrix, n_components):
    pca_mean = matrix.mean(axis=0)
    centered = matrix - pca_mean
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)

    max_components = min(n_components, vt.shape[0])
    components = vt[:max_components]
    explained_variance = (singular_values**2) / max(matrix.shape[0] - 1, 1)
    total_variance = explained_variance.sum()
    explained_ratio = explained_variance / total_variance if total_variance > 0 else explained_variance
    return pca_mean, components, explained_ratio[:max_components]


def pca_reconstruct(matrix, pca_mean, components):
    scores = (matrix - pca_mean) @ components.T
    return scores @ components + pca_mean


def normalize_for_display(image):
    image = image.astype(np.float32)
    minimum = image.min()
    maximum = image.max()
    if maximum - minimum < 1e-6:
        return np.zeros_like(image)
    return (image - minimum) / (maximum - minimum) * 255.0


def plot_pairs(before_images, after_images, labels, title, after_name):
    count = len(before_images)
    fig, axes = plt.subplots(2, count, figsize=(2.4 * count, 5.2), squeeze=False)
    fig.suptitle(title)

    for col, (before, after, label) in enumerate(zip(before_images, after_images, labels)):
        axes[0, col].imshow(before, cmap="gray", vmin=0, vmax=255)
        axes[0, col].set_title(label, fontsize=9)
        axes[0, col].axis("off")

        axes[1, col].imshow(after, cmap="gray", vmin=0, vmax=255)
        axes[1, col].set_title(after_name, fontsize=9)
        axes[1, col].axis("off")

    axes[0, 0].set_ylabel("Before", fontsize=10)
    axes[1, 0].set_ylabel("After", fontsize=10)
    plt.tight_layout()


def plot_pipeline(labels, rows, row_names, title):
    row_count = len(rows)
    col_count = len(labels)
    fig, axes = plt.subplots(row_count, col_count, figsize=(2.4 * col_count, 2.25 * row_count), squeeze=False)
    fig.suptitle(title)

    for row_index, (images, row_name) in enumerate(zip(rows, row_names)):
        for col_index, (image, label) in enumerate(zip(images, labels)):
            axes[row_index, col_index].imshow(image, cmap="gray", vmin=0, vmax=255)
            axes[row_index, col_index].axis("off")
            if row_index == 0:
                axes[row_index, col_index].set_title(label, fontsize=9)
            if col_index == 0:
                axes[row_index, col_index].set_ylabel(row_name, fontsize=10)

    plt.tight_layout()


def plot_pca_variance(explained_ratio, original_pixels, chosen_k=None, feature_name="pixels"):
    cumulative = np.cumsum(explained_ratio) * 100.0
    individual = explained_ratio * 100.0
    components = np.arange(1, len(cumulative) + 1)

    chosen_variance = cumulative[chosen_k - 1] if chosen_k and chosen_k <= len(cumulative) else None
    chosen_label = f"  chosen K={chosen_k} ({chosen_variance:.1f}%)" if chosen_variance is not None else ""

    fig, (ax_bar, ax_cum) = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    fig.suptitle(
        f"PCA Explained Variance  \u2014  original input has {original_pixels:,} {feature_name}\n"
        f"Curve shown for {len(explained_ratio)} components  |{chosen_label}",
        fontsize=11,
    )

    # --- top panel: per-component individual variance ---
    ax_bar.bar(components, individual, color="steelblue", alpha=0.7)
    ax_bar.set_ylabel("Individual explained\nvariance (%)", fontsize=9)
    ax_bar.grid(axis="y", alpha=0.3)

    # --- bottom panel: cumulative curve with threshold annotations ---
    ax_cum.plot(components, cumulative, color="steelblue", linewidth=2)
    ax_cum.fill_between(components, cumulative, alpha=0.15, color="steelblue")
    ax_cum.set_ylabel("Cumulative explained\nvariance (%)", fontsize=9)
    ax_cum.set_xlabel(f"Number of PCA components  (original: {original_pixels:,} {feature_name})", fontsize=9)
    ax_cum.set_ylim(0, 105)
    ax_cum.grid(True, alpha=0.3)

    thresholds = [80, 90, 95, 99]
    colors = ["#e6b800", "#e67e00", "#cc3300", "#660000"]
    for threshold, color in zip(thresholds, colors):
        idx = np.searchsorted(cumulative, threshold)
        if idx < len(cumulative):
            n_comp = components[idx]
            ax_cum.axhline(threshold, color=color, linestyle="--", linewidth=1.0, alpha=0.8)
            ax_cum.axvline(n_comp, color=color, linestyle=":", linewidth=1.0, alpha=0.6)
            ax_cum.annotate(
                f"{threshold}%: {n_comp} components",
                xy=(n_comp, threshold),
                xytext=(n_comp + max(1, len(components) // 30), threshold - 4),
                fontsize=8,
                color=color,
                arrowprops=dict(arrowstyle="->", color=color, lw=0.8),
            )

    compression_at_95 = np.searchsorted(cumulative, 95)
    if compression_at_95 < len(components):
        ratio = original_pixels / components[compression_at_95]
        ax_cum.text(
            0.98, 0.05,
            f"Compression at 95%: {ratio:.0f}x\n({original_pixels:,} {feature_name} \u2192 {components[compression_at_95]} values)",
            transform=ax_cum.transAxes,
            ha="right", va="bottom", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="gray", alpha=0.8),
        )

    # Mark the chosen K
    if chosen_k and chosen_k <= len(components) and chosen_variance is not None:
        ax_bar.axvline(chosen_k, color="green", linestyle="-", linewidth=1.5, alpha=0.9)
        ax_cum.axvline(chosen_k, color="green", linestyle="-", linewidth=1.5, alpha=0.9)
        ax_cum.annotate(
            f"K={chosen_k}\n{chosen_variance:.1f}%",
            xy=(chosen_k, chosen_variance),
            xytext=(chosen_k + max(1, len(components) // 20), chosen_variance - 8),
            fontsize=8, color="green", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="green", lw=1.0),
        )

    plt.tight_layout()


def apply_pipeline(images, pipeline_mean, pipeline_std, pca_mean, pca_components):
    """Transform a list of preprocessed (cropped+CLAHE) images into PCA feature vectors."""
    hog_matrix, _ = hog_batch(images)
    scaled = standard_scale(hog_matrix, pipeline_mean, pipeline_std)
    scores = (scaled - pca_mean) @ pca_components.T
    return scores  # shape: (n_images, n_components)


def build_dataset(folder, pipeline_mean, pipeline_std, pca_mean, pca_components, common_shape):
    """
    Load all .pgm images from *folder*, run the full pipeline, and return
    (X, y) where X is shape (n, n_components) and y holds integer person IDs.
    Only images that crop to *common_shape* are included.
    """
    paths = sorted(glob.glob(os.path.join(folder, "*.pgm")), key=natural_key)
    kept_images, kept_labels = [], []
    for path in paths:
        image = load_gray_image(path)
        cropped, _, _ = remove_gray_side_rectangles(image)
        if cropped.shape != common_shape:
            continue
        kept_images.append(clahe(cropped))
        pid = person_id(path)
        kept_labels.append(int(pid[1:]) if pid else -1)
    X = apply_pipeline(kept_images, pipeline_mean, pipeline_std, pca_mean, pca_components)
    y = np.array(kept_labels, dtype=np.int32)
    return X, y


def prepare_dataset(folder=TRAIN_FOLDER, fit_limit=800, n_components=123, test_size=0.2, random_state=42):
    """
    Full pipeline: load images → crop → CLAHE → HOG → scale → PCA → train/test split.

    Parameters
    ----------
    folder       : folder with .pgm images
    fit_limit    : images used to fit scaler + PCA (0 = all)
    n_components : PCA components to keep
    test_size    : fraction held out for testing
    random_state : random seed for reproducibility

    Returns
    -------
    X_train, X_test, y_train, y_test  — numpy arrays, X shape (n, n_components)
    """
    from sklearn.model_selection import train_test_split

    paths = sorted(glob.glob(os.path.join(folder, "*.pgm")), key=natural_key)
    if not paths:
        raise FileNotFoundError(f"No .pgm images found in {folder!r}")

    fit_paths = load_fit_paths(paths, fit_limit)

    # Fit phase: crop → CLAHE → HOG → scaler → PCA
    fit_raw = [load_gray_image(p) for p in fit_paths]
    fit_cropped, fit_crop_paths = [], []
    for path, img in zip(fit_paths, fit_raw):
        cropped, *_ = remove_gray_side_rectangles(img)
        fit_cropped.append(cropped)
        fit_crop_paths.append(path)

    fit_crop_paths, fit_cropped, common_shape, _ = keep_common_shape(fit_crop_paths, fit_cropped)
    fit_clahe = [clahe(img) for img in fit_cropped]
    fit_hog, _ = hog_batch(fit_clahe)
    pipeline_mean, pipeline_std = fit_standard_scaler(fit_hog)
    fit_scaled = standard_scale(fit_hog, pipeline_mean, pipeline_std)
    pca_mean, pca_components, _ = fit_pca(fit_scaled, n_components)

    # Transform all images
    X, y = build_dataset(folder, pipeline_mean, pipeline_std, pca_mean, pca_components, common_shape)
    print(f"Dataset ready: {X.shape[0]} images, {X.shape[1]} features, {np.unique(y).size} people.")

    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


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


def main():
    parser = argparse.ArgumentParser(description="Preprocess face images and visualize every step.")
    parser.add_argument("--folder", default=TRAIN_FOLDER, help="Folder containing .pgm training images.")
    parser.add_argument("--samples", type=int, default=10, help="Number of different people to show.")
    parser.add_argument("--fit-limit", type=int, default=800, help="Images used to fit scaling/PCA. Use 0 for all images.")
    parser.add_argument("--components", type=int, default=123, help="PCA components to keep.")
    parser.add_argument("--no-show", action="store_true", help="Run preprocessing without opening matplotlib windows.")
    args = parser.parse_args()

    paths = sorted(glob.glob(os.path.join(args.folder, "*.pgm")), key=natural_key)
    if not paths:
        raise FileNotFoundError(f"No .pgm images found in {args.folder!r}")

    sample_paths = select_different_people(paths, args.samples)
    if len(sample_paths) < args.samples:
        print(f"Only found {len(sample_paths)} different people for display.")

    fit_paths = load_fit_paths(paths, args.fit_limit)
    print(f"Found {len(paths)} images. Fitting scaler/PCA on {len(fit_paths)} images.")

    sample_raw = [load_gray_image(path) for path in sample_paths]
    labels = [person_id(path) or os.path.basename(path) for path in sample_paths]

    sample_cropped_all = []
    removed_columns = []
    for image in sample_raw:
        cropped, left, right = remove_gray_side_rectangles(image)
        sample_cropped_all.append(cropped)
        removed_columns.append((left, right))

    print("Side columns removed from displayed samples:")
    for label, (left, right) in zip(labels, removed_columns):
        print(f"  {label}: left={left}, right={right}")

    fit_raw = [load_gray_image(path) for path in fit_paths]
    raw_matrix = flatten_images(fit_raw)
    raw_mean, raw_std = fit_standard_scaler(raw_matrix)
    sample_raw_matrix = flatten_images(sample_raw)
    sample_scaled_only = standard_scale(sample_raw_matrix, raw_mean, raw_std)
    sample_scaled_only_images = [normalize_for_display(row.reshape(sample_raw[0].shape)) for row in sample_scaled_only]

    raw_pca_mean, raw_components, _ = fit_pca(raw_matrix, args.components)
    sample_pca_only = pca_reconstruct(sample_raw_matrix, raw_pca_mean, raw_components)
    sample_pca_only_images = [np.clip(row.reshape(sample_raw[0].shape), 0, 255) for row in sample_pca_only]

    sample_clahe_only = [clahe(image) for image in sample_raw]
    # Standalone HOG visualisation (applied after CLAHE, shown in isolation)
    _, sample_hog_viz_only = hog_batch(sample_clahe_only)

    fit_cropped = []
    fit_crop_paths = []
    crop_counts = Counter()
    for path, image in zip(fit_paths, fit_raw):
        cropped, left, right = remove_gray_side_rectangles(image)
        crop_counts[(left, right)] += 1
        fit_crop_paths.append(path)
        fit_cropped.append(cropped)

    fit_crop_paths, fit_cropped, common_shape, shape_counts = keep_common_shape(fit_crop_paths, fit_cropped)
    print(f"Crop pattern counts in fit set: {dict(crop_counts)}")
    print(f"Cropped shape counts in fit set: {dict(shape_counts)}")
    print(f"Using common cropped shape for final pipeline: {common_shape}")

    pipeline_samples = [
        (label, raw, cropped)
        for label, raw, cropped in zip(labels, sample_raw, sample_cropped_all)
        if cropped.shape == common_shape
    ]
    if not pipeline_samples:
        raise ValueError("None of the displayed samples match the common cropped shape used by the pipeline.")

    labels_for_pipeline = [label for label, _, _ in pipeline_samples]
    sample_raw_for_pipeline = [raw for _, raw, _ in pipeline_samples]
    sample_cropped = [cropped for _, _, cropped in pipeline_samples]

    fit_clahe = [clahe(image) for image in fit_cropped]
    print("Computing HOG features for fit set...")
    fit_hog_matrix, _ = hog_batch(fit_clahe)
    pipeline_mean, pipeline_std = fit_standard_scaler(fit_hog_matrix)
    fit_scaled = standard_scale(fit_hog_matrix, pipeline_mean, pipeline_std)
    # Fit with max(chosen_k, 500) so the variance plot always shows the full elbow curve.
    # SVD computes all singular values in one pass anyway, so cost is the same.
    plot_components = max(args.components, 500)
    pca_mean, all_components, all_explained_ratio = fit_pca(fit_scaled, plot_components)
    pca_components = all_components[: args.components]
    explained_ratio = all_explained_ratio[: args.components]

    sample_clahe_pipeline = [clahe(image) for image in sample_cropped]
    print("Computing HOG features for displayed samples...")
    sample_hog_matrix, sample_hog_vizs = hog_batch(sample_clahe_pipeline)
    sample_scaled_pipeline = standard_scale(sample_hog_matrix, pipeline_mean, pipeline_std)
    sample_scaled_pipeline_images = [hog_vector_to_display(row) for row in sample_scaled_pipeline]
    sample_pca_scaled = pca_reconstruct(sample_scaled_pipeline, pca_mean, pca_components)
    sample_pca_pipeline = sample_pca_scaled * pipeline_std + pipeline_mean
    sample_pca_pipeline_images = [hog_vector_to_display(row) for row in sample_pca_pipeline]

    print(
        f"Final PCA keeps {len(pca_components)} components "
        f"and explains {np.sum(explained_ratio) * 100.0:.2f}% of HOG variance "
        f"(input: {fit_hog_matrix.shape[1]:,} HOG features from {common_shape[0]}x{common_shape[1]} images)."
    )

    if args.no_show:
        return

    plot_pairs(sample_raw, sample_cropped_all, labels, "Step 1: Gray Side-Rectangle Removal Only", "Cropped")
    plot_pairs(sample_raw, sample_clahe_only, labels, "Step 2: CLAHE Only", "CLAHE")
    plot_pairs(sample_clahe_only, sample_hog_viz_only, labels, "Step 3: HOG Only  (CLAHE \u2192 gradient orientation map)", "HOG visualization")
    plot_pairs(sample_raw, sample_scaled_only_images, labels, "Step 4: Standard Scaling Only (on raw pixels — for reference)", "Scaled display")
    plot_pairs(sample_raw, sample_pca_only_images, labels, "Step 5: PCA Reconstruction Only (on raw pixels — for reference)", "PCA reconstruction")
    plot_pipeline(
        labels_for_pipeline,
        [
            sample_raw_for_pipeline,
            sample_cropped,
            sample_clahe_pipeline,
            sample_hog_vizs,
            sample_scaled_pipeline_images,
            sample_pca_pipeline_images,
        ],
        ["Original", "Cropped", "CLAHE", "HOG viz", "Scaled HOG", "PCA recon (HOG)"],
        "Full Process: Crop \u2192 CLAHE \u2192 HOG \u2192 Standard Scaling \u2192 PCA",
    )
    plot_pca_variance(
        all_explained_ratio,
        original_pixels=fit_hog_matrix.shape[1],
        chosen_k=args.components,
        feature_name="HOG features",
    )
    plt.show()


if __name__ == "__main__":
    main()