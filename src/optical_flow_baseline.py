# ===============================
# optical_flow_baseline.py
# ===============================

import cv2
import numpy as np
import os
from davis_loader import load_davis_frames

# ---------- 1️⃣ Basic Farnebäck Flow ----------
# def compensate_camera_motion(prev_gray, gray, max_corners=500):
#     """
#     Estimate global camera motion using feature matching and homography,
#     then warp the previous frame to align with the current one.
#     """
#     kp1 = cv2.goodFeaturesToTrack(prev_gray, max_corners, 0.01, 10)
#     if kp1 is None:
#         return prev_gray

#     kp2, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, kp1, None)
#     if kp2 is None or st is None:
#         return prev_gray

#     kp1_good, kp2_good = kp1[st == 1], kp2[st == 1]
#     if len(kp1_good) < 10 or len(kp2_good) < 10:
#         return prev_gray

#     H, _ = cv2.findHomography(kp1_good, kp2_good, cv2.RANSAC, 5.0)
#     if H is None:
#         return prev_gray

#     stabilized = cv2.warpPerspective(prev_gray, H, (gray.shape[1], gray.shape[0]))
#     return stabilized


# # ---------- 2️⃣ Improved Farnebäck Flow + Visualization ----------
# def farneback_davis_improved(sequence_dir, out_dir, percentile=94,
#                              smooth_kernel=7, A_min=1500):
#     """
#     Compute optical flow on DAVIS sequences with stabilization,
#     generate binary motion masks, and visualize as [Original | Heatmap | Mask].
#     """
#     os.makedirs(out_dir, exist_ok=True)
#     frames, names = load_davis_frames(sequence_dir)
#     if len(frames) < 2:
#         print("Not enough frames for optical flow.")
#         return

#     prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

#     for i in range(1, len(frames)):
#         gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)

#         # ---------- Step 1: Compensate camera motion ----------
#         stab_prev = compensate_camera_motion(prev_gray, gray)

#         # ---------- Step 2: Compute optical flow ----------
#         flow = cv2.calcOpticalFlowFarneback(
#             stab_prev, gray, None,
#             pyr_scale=0.5, levels=3, winsize=15,
#             iterations=3, poly_n=5, poly_sigma=1.2, flags=0
#         )

#         # ---------- Step 3: Magnitude & Smoothing ----------
#         mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
#         mag = cv2.bilateralFilter(mag.astype(np.float32), 9, 50, 50)
#         mag = cv2.GaussianBlur(mag, (smooth_kernel, smooth_kernel), 0)

#         # Normalize for visualization
#         mag_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
#         mag_norm = np.uint8(np.clip(mag_norm, 0, 255))

#         # ---------- Step 4: Threshold + Cleanup ----------
#         th = np.percentile(mag_norm, percentile)
#         mask = (mag_norm > th).astype(np.uint8) * 255
#         mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
#         mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

#         num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
#         filtered = np.zeros_like(mask)
#         for j in range(1, num_labels):
#             if stats[j, cv2.CC_STAT_AREA] >= A_min:
#                 filtered[labels == j] = 255

#         # ---------- Step 5: Create Heatmap Visualization ----------
#         heatmap = cv2.applyColorMap(mag_norm, cv2.COLORMAP_JET)

#         # Label panels
#         cv2.putText(frames[i], "Original", (20, 30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
#         cv2.putText(heatmap, "Flow Heatmap", (20, 30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

#         mask_color = cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)
#         cv2.putText(mask_color, "Motion Mask", (20, 30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

#         # ---------- Step 6: Combine side-by-side ----------
#         comparison = np.hstack([frames[i], heatmap, mask_color])

#         # ---------- Step 7: Save results ----------
#         # cv2.imwrite(os.path.join(out_dir, f"mask_{names[i]}"), filtered)
#         cv2.imwrite(os.path.join(out_dir, f"vis_compare_{names[i]}"), comparison)

#         prev_gray = gray

#     print("✅ Stabilized Farneback (heatmap + mask) complete.")

def compensate_camera_motion(prev_gray, gray, max_corners=500):
    """Estimate and compensate global camera motion using homography."""
    kp1 = cv2.goodFeaturesToTrack(prev_gray, max_corners, 0.01, 10)
    if kp1 is None:
        return prev_gray

    kp2, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, kp1, None)
    if kp2 is None or st is None:
        return prev_gray

    kp1_good, kp2_good = kp1[st == 1], kp2[st == 1]
    if len(kp1_good) < 10:
        return prev_gray

    H, _ = cv2.findHomography(kp1_good, kp2_good, cv2.RANSAC, 5.0)
    if H is None:
        return prev_gray

    stabilized = cv2.warpPerspective(prev_gray, H, (gray.shape[1], gray.shape[0]))
    return stabilized


def farneback_davis(sequence_dir, out_dir,
                             percentile=94, smooth_kernel=7, A_min=1500, fps=15):
    """
    Compute optical flow on DAVIS sequences with stabilization,
    visualize as [Original | Heatmap | Mask], and export to MP4.
    """
    os.makedirs(out_dir, exist_ok=True)
    frames, names = load_davis_frames(sequence_dir)
    if len(frames) < 2:
        print("Not enough frames for optical flow.")
        return

    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    h, w = frames[0].shape[:2]
    video_path = os.path.join(out_dir, "motion_comparison.mp4")

    # VideoWriter setup: width = 3×frame width (original, heatmap, mask)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_video = cv2.VideoWriter(video_path, fourcc, fps, (w * 3, h))

    for i in range(1, len(frames)):
        gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)

        # ---------- Step 1: Stabilize ----------
        stab_prev = compensate_camera_motion(prev_gray, gray)

        # ---------- Step 2: Optical Flow ----------
        flow = cv2.calcOpticalFlowFarneback(
            stab_prev, gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )

        # ---------- Step 3: Magnitude + Smoothing ----------
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mag = cv2.bilateralFilter(mag.astype(np.float32), 9, 50, 50)
        mag = cv2.GaussianBlur(mag, (smooth_kernel, smooth_kernel), 0)
        mag_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # ---------- Step 4: Threshold + Clean ----------
        th = np.percentile(mag_norm, percentile)
        mask = (mag_norm > th).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        filtered = np.zeros_like(mask)
        for j in range(1, num_labels):
            if stats[j, cv2.CC_STAT_AREA] >= A_min:
                filtered[labels == j] = 255

        # ---------- Step 5: Heatmap Visualization ----------
        heatmap = cv2.applyColorMap(mag_norm, cv2.COLORMAP_JET)

        # ---------- Step 6: Label Panels ----------
        original_panel = frames[i].copy()
        mask_color = cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)
        cv2.putText(original_panel, "Original", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(heatmap, "Flow Heatmap", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(mask_color, "Motion Mask", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # ---------- Step 7: Combine Panels ----------
        comparison = np.hstack([original_panel, heatmap, mask_color])

        # Save individual frame visualization (optional)
        cv2.imwrite(os.path.join(out_dir, f"vis_compare_{names[i]}"), comparison)

        # Write to MP4
        out_video.write(comparison)

        prev_gray = gray

    out_video.release()
    print(f"✅ Saved motion comparison video: {video_path}")
    