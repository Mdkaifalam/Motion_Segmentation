"""
flow_utils.py
--------------------------------
Utility functions for:
- Loading RAFT flow (.npy)
- Computing magnitude map
- Thresholding to motion mask
- Morphological cleaning
- Extracting connected components
- Returning blob bounding boxes + centroids

Used for Flow-SAM prompts.
"""

import numpy as np
import cv2


# -------------------------------------------------------------
# Load flow (HxWx2) from .npy file
# -------------------------------------------------------------
def load_flow_npy(flow_path):
    """
    Load a RAFT optical flow saved as numpy array (HxWx2).
    """
    flow = np.load(flow_path)  # shape (H, W, 2)
    return flow


# -------------------------------------------------------------
# Compute magnitude map
# -------------------------------------------------------------
def flow_magnitude(flow):
    """
    Compute |flow| magnitude sqrt(u^2 + v^2)
    """
    u = flow[..., 0]
    v = flow[..., 1]
    mag = np.sqrt(u**2 + v**2)
    return mag


# -------------------------------------------------------------
# Threshold magnitude to get initial motion mask
# -------------------------------------------------------------
def threshold_motion(mag, thresh=1.5):
    """
    Simple threshold on magnitude.
    Higher threshold -> fewer blobs.
    """
    mask = (mag > thresh).astype(np.uint8) * 255
    return mask


# -------------------------------------------------------------
# Morphological cleanup (optional but strongly recommended)
# -------------------------------------------------------------
def clean_motion_mask(mask, kernel_size=5):
    """
    Apply morphological opening + closing to reduce noise.
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Remove small noise dots
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Fill small holes / connect parts
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

    return cleaned


# -------------------------------------------------------------
# Extract connected-component blobs
# -------------------------------------------------------------
def extract_motion_blobs(mask, min_area=50):
    """
    Finds connected components in motion mask.
    Returns:
        blobs = [
            {
                'bbox': (x1, y1, x2, y2),
                'centroid': (cx, cy),
                'area': area
            },
            ...
        ]
    """

    # Ensure binary format
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)

    blobs = []

    for i in range(1, num_labels):  # skip background (0)
        x, y, w, h, area = stats[i]

        if area < min_area:
            continue

        cx, cy = centroids[i]

        blob = {
            "bbox": (x, y, x + w, y + h),
            "centroid": (int(cx), int(cy)),
            "area": area
        }

        blobs.append(blob)

    return blobs


# -------------------------------------------------------------
# Full pipeline from flow_path -> blobs
# -------------------------------------------------------------
def get_motion_blobs_from_flow(flow_path, mag_thresh=1.5, min_area=50):
    """
    Full pipeline:
    Load flow -> magnitude -> threshold -> clean -> blob extraction
    """
    flow = load_flow_npy(flow_path)
    mag = flow_magnitude(flow)
    mask = threshold_motion(mag, thresh=mag_thresh)
    cleaned = clean_motion_mask(mask)
    blobs = extract_motion_blobs(cleaned, min_area=min_area)
    return blobs, cleaned, mag
