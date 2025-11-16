"""
flow_utils.py  (IMPROVED – Noise-Reduced Version)
-----------------------------------------------
Noise reduction upgrades:
- Bilateral + Gaussian smoothing on magnitude
- Normalization to [0,255]
- Percentile thresholding (recommended for RAFT)
- Larger morphological kernel
- Higher min-area filtering
- Optional Otsu/Top-p threshold modes
"""

import numpy as np
import cv2


# -------------------------------------------------------------
# Load flow (.npy)
# -------------------------------------------------------------
def load_flow_npy(flow_path):
    flow = np.load(flow_path)       # shape (H, W, 2)
    return flow


# -------------------------------------------------------------
# Compute |flow|
# -------------------------------------------------------------
def flow_magnitude(flow):
    u = flow[..., 0]
    v = flow[..., 1]
    mag = np.sqrt(u*u + v*v)
    return mag.astype(np.float32)


# -------------------------------------------------------------
# Percentile threshold (recommended for RAFT)
# -------------------------------------------------------------
def percentile_threshold(mag_norm, percentile=97.0):
    th = np.percentile(mag_norm, percentile)
    mask = (mag_norm > th).astype(np.uint8) * 255
    return mask


# -------------------------------------------------------------
# Clean mask (noise reduction)
# -------------------------------------------------------------
def clean_motion_mask(mask, kernel_size=9):
    """
    Larger kernel = more aggressive noise removal.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # Remove dots
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Fill holes
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


# -------------------------------------------------------------
# Connected-components filtering
# -------------------------------------------------------------
def extract_motion_blobs(mask, min_area=2000):
    """
    Removes very tiny noisy blobs from RAFT.
    """
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)

    blobs = []
    for i in range(1, num_labels):  # skip background
        x, y, w, h, area = stats[i]
        if area < min_area:         # << MUCH stronger filtering
            continue

        cx, cy = centroids[i]
        blobs.append({
            "bbox": (x, y, x+w, y+h),
            "centroid": (int(cx), int(cy)),
            "area": area
        })

    return blobs


# -------------------------------------------------------------
# FULL PIPELINE FOR FLOW → CLEAN MASK + BLOBS
# -------------------------------------------------------------
def get_motion_blobs_from_flow(
        flow_path,
        percentile=97.0,
        smooth_kernel=7,
        bilateral=True,
        min_area=2000):
    """
    Full noise-reduced pipeline:
    - Load flow
    - Magnitude
    - Smoothing (bilateral + Gaussian)
    - Normalize
    - Percentile threshold (recommended)
    - Morphology
    - Connected-components
    """

    flow = load_flow_npy(flow_path)
    mag = flow_magnitude(flow)

    # 1) Bilateral (best noise reducer for RAFT)
    if bilateral:
        mag = cv2.bilateralFilter(mag, 7, 50, 50)

    # 2) Gaussian smoothing
    k = smooth_kernel if smooth_kernel % 2 == 1 else smooth_kernel + 1
    mag = cv2.GaussianBlur(mag, (k, k), 0)

    # 3) Normalize to 0..255
    mag_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 4) Percentile threshold
    mask = percentile_threshold(mag_norm, percentile=percentile)

    # 5) Morph cleaning
    mask = clean_motion_mask(mask, kernel_size=9)

    # 6) Connected component filtering
    blobs = extract_motion_blobs(mask, min_area=min_area)

    return blobs, mask, mag_norm
