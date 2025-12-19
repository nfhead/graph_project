import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Dict


@dataclass
class DetectedCurve:
    local_id: str
    mask: np.ndarray                    # binary mask
    points: List[Dict]                  # downsampled points [{"x_px","y_px"}]
    polyline: List[Dict]                # ordered points along curve [{"x_px","y_px"}]
    bbox: Dict[str, int]
    color_rgb: List[int]
    color_name: str


def _cluster_by_color(img_bgr: np.ndarray, max_clusters: int = 5):
    h, w, _ = img_bgr.shape
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pixels = img_rgb.reshape(-1, 3)

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hsv_flat = hsv.reshape(-1, 3)
    v = hsv_flat[:, 2]
    s = hsv_flat[:, 1]

    bg_mask_flat = (v > 230) & (s < 25)
    fg_indices = np.where(~bg_mask_flat)[0]

    labels = np.full((h * w,), -1, dtype=np.int32)
    if len(fg_indices) == 0:
        return labels.reshape((h, w)), 0

    fg_pixels = pixels[fg_indices]
    max_samples = 5000
    if fg_pixels.shape[0] > max_samples:
        sample_idx = np.random.choice(fg_pixels.shape[0], max_samples, replace=False)
        sample_pixels = fg_pixels[sample_idx]
    else:
        sample_pixels = fg_pixels

    k = min(max_clusters, sample_pixels.shape[0])
    if k <= 1:
        labels[fg_indices] = 0
        return labels.reshape((h, w)), 1

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1.0)
    _compactness, sample_labels, centers = cv2.kmeans(
        sample_pixels.astype(np.float32),
        K=k,
        bestLabels=None,
        criteria=criteria,
        attempts=3,
        flags=cv2.KMEANS_PP_CENTERS
    )

    # assign all fg pixels to nearest center
    dists = np.linalg.norm(
        fg_pixels.astype(np.float32)[:, None, :] - centers[None, :, :],
        axis=2
    )
    full_fg_labels = np.argmin(dists, axis=1)
    labels[fg_indices] = full_fg_labels
    return labels.reshape((h, w)), k


def _rgb_to_name(rgb: np.ndarray) -> str:
    r, g, b = rgb.astype(int)
    if r + g + b < 30:
        return "black"
    if r > 200 and g > 200 and b > 200:
        return "white"
    if r > g + 40 and r > b + 40:
        return "red"
    if g > r + 30 and g > b + 30:
        return "green"
    if b > r + 30 and b > g + 30:
        return "blue"
    if r > 180 and g > 180 and b < 100:
        return "yellow"
    if r > 180 and b > 180 and g < 100:
        return "magenta"
    if g > 180 and b > 180 and r < 100:
        return "cyan"
    return "colored"


def _skeletonize(binary_mask: np.ndarray) -> np.ndarray:
    """
    Simple morphological skeletonization (no ximgproc dependency).
    Input: 0/255 uint8 mask
    Output: 0/255 skeleton
    """
    img = (binary_mask > 0).astype(np.uint8) * 255
    skel = np.zeros(img.shape, np.uint8)

    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False
    while not done:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()

        if cv2.countNonZero(img) == 0:
            done = True
    return skel


def _ordered_polyline_from_skeleton(skel: np.ndarray, max_points: int = 2000) -> List[Dict]:
    """
    Practical approximation:
    - take skeleton pixels
    - sort by x (then y) to create a monotonic-ish polyline
    Works well for many engineering plots where x increases left->right.
    """
    ys, xs = np.where(skel > 0)
    if len(xs) == 0:
        return []

    coords = list(zip(xs.tolist(), ys.tolist()))
    coords.sort(key=lambda p: (p[0], p[1]))

    # downsample if too large
    if len(coords) > max_points:
        step = max(1, len(coords) // max_points)
        coords = coords[::step]

    return [{"x_px": int(x), "y_px": int(y)} for x, y in coords]


def detect_curves(
    image_path: str,
    max_curves: int = 5,
    min_pixels_per_curve: int = 200
) -> List[DetectedCurve]:
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"[WARN] Could not read image: {image_path}")
        return []

    h, w, _ = img_bgr.shape
    labels, n_clusters = _cluster_by_color(img_bgr, max_clusters=max_curves)
    if n_clusters == 0:
        return []

    curves: List[DetectedCurve] = []
    curve_letter_ord = ord("A")

    for c_idx in range(n_clusters):
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[labels == c_idx] = 255

        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        num_labels, cc_labels, stats, _centroids = cv2.connectedComponentsWithStats(mask)

        best_area = 0
        best_label = None
        for lab in range(1, num_labels):
            area = stats[lab, cv2.CC_STAT_AREA]
            if area > best_area:
                best_area = area
                best_label = lab

        if best_label is None or best_area < min_pixels_per_curve:
            continue

        curve_mask = np.zeros_like(mask)
        curve_mask[cc_labels == best_label] = 255

        ys, xs = np.where(curve_mask > 0)
        if len(xs) == 0:
            continue

        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())

        # sample points (still useful)
        coords = list(zip(xs.tolist(), ys.tolist()))
        if len(coords) > 800:
            sample_idx = np.random.choice(len(coords), 800, replace=False)
            coords_sample = [coords[i] for i in sample_idx]
        else:
            coords_sample = coords
        points = [{"x_px": int(x), "y_px": int(y)} for x, y in coords_sample]

        bbox = {"x_min_px": x_min, "x_max_px": x_max, "y_min_px": y_min, "y_max_px": y_max}

        # mean curve color
        curve_pixels_bgr = img_bgr[ys, xs, :]
        curve_pixels_rgb = cv2.cvtColor(curve_pixels_bgr.reshape(-1, 1, 3), cv2.COLOR_BGR2RGB).reshape(-1, 3)
        mean_rgb = curve_pixels_rgb.mean(axis=0)
        color_name = _rgb_to_name(mean_rgb)

        # skeleton + polyline
        skel = _skeletonize(curve_mask)
        polyline = _ordered_polyline_from_skeleton(skel)

        local_id = f"curve_{chr(curve_letter_ord)}"
        curve_letter_ord += 1

        curves.append(
            DetectedCurve(
                local_id=local_id,
                mask=curve_mask,
                points=points,
                polyline=polyline,
                bbox=bbox,
                color_rgb=[int(mean_rgb[0]), int(mean_rgb[1]), int(mean_rgb[2])],
                color_name=color_name,
            )
        )

    return curves
