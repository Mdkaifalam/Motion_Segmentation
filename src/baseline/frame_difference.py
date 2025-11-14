# ===============================
# frame_difference.py
# ===============================

import cv2
import numpy as np
import os
from davis_loader import load_davis_frames

def stabilize_frame(prev_gray, gray):
    # Detect feature points
    kp1, kp2 = cv2.goodFeaturesToTrack(prev_gray, 500, 0.01, 10), None
    kp2, st, err = None, None, None
    if kp1 is not None:
        kp2, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, kp1, None)
    if kp2 is None:
        return prev_gray  # fallback: no motion detected

    # Keep only good matches
    kp1_good = kp1[st == 1]
    kp2_good = kp2[st == 1]

    # Estimate homography (global motion)
    H, _ = cv2.findHomography(kp1_good, kp2_good, cv2.RANSAC, 5.0)
    if H is None:
        return prev_gray

    # Warp previous frame to align with current
    stabilized_prev = cv2.warpPerspective(prev_gray, H, (gray.shape[1], gray.shape[0]))
    return stabilized_prev


# def frame_diff_davis(sequence_dir, out_dir, T=20, kernel_size=7, A_min=500, use_stabilization=True):
#     os.makedirs(out_dir, exist_ok=True)
#     # frames, frame_names = load_davis_frames(sequence_dir)
#     # if len(frames) < 2:
#     #     print(f"Not enough frames in {sequence_dir}")
#     #     return

#     # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
#     # prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)

#     # for idx in range(1, len(frames)):
#     #     gray = cv2.cvtColor(frames[idx], cv2.COLOR_BGR2GRAY)
#     #     stabilized_prev = stabilize_frame(prev_gray, gray)
#     #     diff = cv2.absdiff(gray, stabilized_prev)
#     #     # diff = cv2.absdiff(gray, prev_gray)

#     #     # Threshold
#     #     _, mask = cv2.threshold(diff, T, 255, cv2.THRESH_BINARY)

#     #     # Denoise
#     #     mask = cv2.medianBlur(mask, 3)
#     #     mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
#     #     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

#     #     # Remove small blobs
#     #     num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
#     #     out = np.zeros_like(mask)
#     #     for i in range(1, num_labels):
#     #         if stats[i, cv2.CC_STAT_AREA] >= A_min:
#     #             out[labels == i] = 255

#     #     cv2.imwrite(os.path.join(out_dir, f"mask_{frame_names[idx]}"), out)
#     #     prev_gray = gray

#     frames, names = load_davis_frames(sequence_dir)
#     prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)

#     for i in range(1, len(frames)):
#         gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
#         if use_stabilization:
#             prev_gray = stabilize_frame(prev_gray, gray)

#         # Gradient-based difference
#         gx1 = cv2.Sobel(prev_gray, cv2.CV_32F, 1, 0, 3)
#         gy1 = cv2.Sobel(prev_gray, cv2.CV_32F, 0, 1, 3)
#         gx2 = cv2.Sobel(gray, cv2.CV_32F, 1, 0, 3)
#         gy2 = cv2.Sobel(gray, cv2.CV_32F, 0, 1, 3)
#         diff = cv2.magnitude(gx2 - gx1, gy2 - gy1)
#         diff = np.uint8(np.clip(diff, 0, 255))

#         # Adaptive threshold
#         T = np.percentile(diff, 98)
#         _, mask = cv2.threshold(diff, T, 255, cv2.THRESH_BINARY)

#         mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
#         mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
#         cv2.imwrite(os.path.join(out_dir, f"mask_{names[i]}"), mask)

#         prev_gray = gray

#     print(f"Frame Differencing completed for {sequence_dir}")

def frame_diff_davis(sequence_dir, out_dir, percentile=98, T = 20, kernel_size = 7, A_min=1500):
    os.makedirs(out_dir, exist_ok=True)
    frames, names = load_davis_frames(sequence_dir)
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask_buffer = []

    for i in range(1, len(frames)):
        gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        prev_gray_stab = stabilize_frame(prev_gray, gray)

        # --- gradient-based difference ---
        gx1 = cv2.Sobel(prev_gray_stab, cv2.CV_32F, 1, 0, 3)
        gy1 = cv2.Sobel(prev_gray_stab, cv2.CV_32F, 0, 1, 3)
        gx2 = cv2.Sobel(gray, cv2.CV_32F, 1, 0, 3)
        gy2 = cv2.Sobel(gray, cv2.CV_32F, 0, 1, 3)
        diff = cv2.magnitude(gx2 - gx1, gy2 - gy1)
        diff = np.uint8(np.clip(diff, 0, 255))

        # --- adaptive threshold ---
        T = np.percentile(diff, percentile)
        _, mask = cv2.threshold(diff, T, 255, cv2.THRESH_BINARY)

        # --- morphology (remove small dots) ---
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # --- connected component filtering ---
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        filtered = np.zeros_like(mask)
        for j in range(1, num_labels):
            if stats[j, cv2.CC_STAT_AREA] >= A_min:
                filtered[labels == j] = 255
        mask = filtered

        # --- mild temporal smoothing (3-frame median) ---
        mask_buffer.append(mask)
        if len(mask_buffer) > 3:
            mask_buffer.pop(0)
        if len(mask_buffer) == 3:
            mask = np.median(np.stack(mask_buffer, axis=0), axis=0).astype(np.uint8)

        cv2.imwrite(os.path.join(out_dir, f"mask_{names[i]}"), mask)
        prev_gray = gray

    print(f"Finished improved frame differencing for {sequence_dir}")