import cv2
import numpy as np
import os
from davis_loader import load_davis_frames

def baseline_motion_segmentation(sequence_dir, out_dir,
                                 percentile=94,
                                 smooth_kernel=7,
                                 A_min=1500):
    """
    Baseline motion segmentation using optical flow magnitude thresholding.
    Args:
        sequence_dir: Path to DAVIS sequence folder
        out_dir: Output directory for masks
        percentile: Threshold percentile for motion magnitude
        smooth_kernel: Gaussian smoothing for magnitude map
        A_min: Minimum blob area (px)
    """
    os.makedirs(out_dir, exist_ok=True)
    frames, names = load_davis_frames(sequence_dir)
    if len(frames) < 2:
        print("Not enough frames in sequence.")
        return

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)

    for i in range(1, len(frames)):
        gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)

        # ----- Optical flow -----
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mag = cv2.GaussianBlur(mag, (smooth_kernel, smooth_kernel), 0)
        mag_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        mag_norm = np.uint8(np.clip(mag_norm, 0, 255))

        # ----- Threshold -----
        th = np.percentile(mag_norm, percentile)
        mask = (mag_norm > th).astype(np.uint8) * 255

        # ----- Morphology + Blob cleanup -----
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        filtered = np.zeros_like(mask)
        for j in range(1, num_labels):
            if stats[j, cv2.CC_STAT_AREA] >= A_min:
                filtered[labels == j] = 255

        cv2.imwrite(os.path.join(out_dir, f"mask_{names[i]}"), filtered)
        prev_gray = gray

    print(f"✅ Baseline motion segmentation done for {sequence_dir}")

# if __name__ == "__main__":
#     import argparse

#     parser = argparse.ArgumentParser()
#     input_root_default = "datasets/DAVIS/JPEGImages/480p"
#     sequence_default = "bear"
#     parser.add_argument("--sequence_dir", type=str, required=True, help="Path to DAVIS sequence folder", default = sequence_default)
#     # output_root_default = "./outputs/}baseline_motion"
#     parser.add_argument("--out_dir", type=str, required=True, help="Output directory for masks")
#     parser.add_argument("--percentile", type=int, default=94, help="Threshold percentile for motion magnitude")
#     parser.add_argument("--smooth_kernel", type=int, default=7, help="Gaussian smoothing kernel size")
#     parser.add_argument("--A_min", type=int, default=1500, help="Minimum blob area (px)")
    
#     args = parser.parse_args()
#     args.out_dir = os.path.join("./outputs", os.path.basename(args.sequence_dir) + "_baseline_motion")
#     baseline_motion_segmentation(args.sequence_dir, args.out_dir,
#                                  args.percentile, args.smooth_kernel,
#                                  args.A_min)
