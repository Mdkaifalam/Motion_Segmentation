import cv2
import numpy as np
import os
from davis_loader import load_davis_frames

def mog2_background_davis(sequence_dir, out_dir,
                                 method='MOG2',
                                 A_min=1500,
                                 kernel_size=7,
                                 warmup=10):
    """
    Apply background subtraction (MOG2 or KNN) to a DAVIS sequence.

    Args:
        sequence_dir (str): path to DAVIS sequence folder (e.g., .../bear)
        out_dir (str): output mask directory
        method (str): 'MOG2' or 'KNN'
        A_min (int): min foreground blob area
        kernel_size (int): morphology kernel size
        warmup (int): number of frames to allow model to adapt before saving
    """
    os.makedirs(out_dir, exist_ok=True)
    frames, frame_names = load_davis_frames(sequence_dir)
    if len(frames) == 0:
        print("No frames found in", sequence_dir)
        return

    # --- Create background subtractor ---
    if method == 'MOG2':
        backSub = cv2.createBackgroundSubtractorMOG2(
            varThreshold=16, detectShadows=False)
    else:
        backSub = cv2.createBackgroundSubtractorKNN(
            dist2Threshold=1000, detectShadows=False)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (kernel_size, kernel_size))

    for idx, frame in enumerate(frames):
        fgmask = backSub.apply(frame)

        # Skip warm-up period
        if idx < warmup:
            continue

        # Binary clean-up
        _, mask = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
        mask = cv2.medianBlur(mask, 3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Remove tiny blobs
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        out = np.zeros_like(mask)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= A_min:
                out[labels == i] = 255

        cv2.imwrite(os.path.join(out_dir, f"mask_{frame_names[idx]}"), out)

    print(f"{method} background subtraction complete for {sequence_dir}")

def knn_background_davis(sequence_dir, out_dir,
                         A_min=2500,
                         kernel_size=7,
                         dist2Threshold=2000,
                         detectShadows=False,
                         learningRate=0.001,
                         warmup=10):
    """
    Improved KNN background subtraction for DAVIS sequences.
    """
    os.makedirs(out_dir, exist_ok=True)
    frames, frame_names = load_davis_frames(sequence_dir)
    if len(frames) == 0:
        print("No frames found in", sequence_dir)
        return

    backSub = cv2.createBackgroundSubtractorKNN(
        dist2Threshold=dist2Threshold,
        detectShadows=detectShadows
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    idx = 0
    for frame in frames:
        fgmask = backSub.apply(frame, learningRate=learningRate)

        if idx < warmup:
            idx += 1
            continue

        # ------------------------------------------------------------
        # 1️⃣ Visualize motion intensity (based on fgmask values)
        # ------------------------------------------------------------

        # make a smooth heatmap to represent intensity of change
        motion_intensity = cv2.normalize(fgmask, None, 0, 255, cv2.NORM_MINMAX)
        motion_intensity = np.uint8(motion_intensity)
        intensity_map = cv2.applyColorMap(motion_intensity, cv2.COLORMAP_JET)

        # ------------------------------------------------------------
        # 2️⃣ Threshold + clean mask (same logic you already have)
        # ------------------------------------------------------------
        _, mask = cv2.threshold(fgmask, 180, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        out = np.zeros_like(mask)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= A_min:
                out[labels == i] = 255

        # ------------------------------------------------------------
        # 3️⃣ Create RGB versions for visualization
        # ------------------------------------------------------------
        mask_color = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)

        # optional: add labels on each panel
        cv2.putText(frame, "Original", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(intensity_map, "Intensity", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(mask_color, "Mask", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # ------------------------------------------------------------
        # 4️⃣ Combine side by side: [Original | Intensity | Mask]
        # ------------------------------------------------------------
        comparison = np.hstack([frame, intensity_map, mask_color])

        # ------------------------------------------------------------
        # 5️⃣ Save all outputs
        # ------------------------------------------------------------
        cv2.imwrite(os.path.join(out_dir, f"mask_{frame_names[idx]}"), out)
        cv2.imwrite(os.path.join(out_dir, f"vis_compare_{frame_names[idx]}"), comparison)

        idx += 1

    # for frame in frames:
    #     fgmask = backSub.apply(frame, learningRate=learningRate)

    #     if idx < warmup:
    #         idx += 1
    #         continue
    #     def visualize_fgmask(fgmask):
    #         vis = np.zeros((fgmask.shape[0], fgmask.shape[1], 3), dtype=np.uint8)
    #         vis[fgmask == 0] = [0, 0, 0]         # background → black
    #         vis[fgmask == 127] = [255, 0, 0]     # shadow → blue/red
    #         vis[fgmask == 255] = [0, 255, 0]     # foreground → green
    #         return vis

    #     vis_mask = visualize_fgmask(fgmask)

    #     # --- Overlay on original frame for debugging ---
    #     overlay = cv2.addWeighted(frame, 0.7, vis_mask, 0.3, 0)

    #     # --- Save or display ---
    #     cv2.imwrite(os.path.join(out_dir, f"vis_{frame_names[idx]}"), overlay)

    #     # Threshold more gently
    #     _, mask = cv2.threshold(fgmask, 180, 255, cv2.THRESH_BINARY)
    #     mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    #     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    #     # Remove small blobs
    #     num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    #     out = np.zeros_like(mask)
    #     for i in range(1, num_labels):
    #         if stats[i, cv2.CC_STAT_AREA] >= A_min:
    #             out[labels == i] = 255

    #     cv2.imwrite(os.path.join(out_dir, f"mask_{frame_names[idx]}"), out)
    #     idx += 1

    print(f"KNN background subtraction done for {sequence_dir}")