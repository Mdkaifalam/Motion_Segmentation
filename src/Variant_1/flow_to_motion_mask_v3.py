import os
import argparse
import numpy as np
import cv2


# ============================================================
# Threshold helpers
# ============================================================
def _th_from_percentile(mag_norm, percentile):
    return float(np.percentile(mag_norm, percentile))


def _th_from_top_p(mag_norm, p):
    arr = mag_norm[mag_norm > 0]
    if arr.size < 50:
        arr = mag_norm.flatten()
    return float(np.percentile(arr, 100.0 * (1.0 - p)))


def _th_from_otsu(mag_norm):
    t, _ = cv2.threshold(mag_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(t)


# ============================================================
# Balanced Camera Motion Removal
# ============================================================
def remove_global_motion(flow,
                         bg_percentile=70.0,
                         sample_ratio=0.05,
                         min_bg_points=2000,
                         ransac_thresh=3.0,
                         global_weight=0.8):

    H, W, _ = flow.shape
    xs, ys = np.meshgrid(np.arange(W), np.arange(H))
    pts1 = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float32)
    pts2 = pts1 + flow.reshape(-1, 2)

    mag = np.linalg.norm(flow.reshape(-1, 2), axis=1)
    thr = np.percentile(mag, bg_percentile)
    bg_idx = np.where(mag < thr)[0]

    if bg_idx.size < min_bg_points:
        bg_idx = np.arange(mag.size)

    N = bg_idx.size
    K = min(max(int(N * sample_ratio), min_bg_points), N)
    sel = np.random.choice(bg_idx, size=K, replace=False)

    pts1_s = pts1[sel]
    pts2_s = pts2[sel]

    M, _ = cv2.estimateAffine2D(
        pts1_s, pts2_s,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_thresh
    )

    if M is None:
        return flow

    pts2_pred = (pts1 @ M[:, :2].T + M[:, 2]).reshape(H, W, 2)
    pts2_pred = cv2.GaussianBlur(pts2_pred, (0, 0), sigmaX=1.5)

    global_flow = pts2_pred - pts1.reshape(H, W, 2)
    return flow - global_weight * global_flow


# ============================================================
# FLOW → CLEAN MASK
# ============================================================
def flow_to_motion_mask(flow,
                        threshold_mode="percentile",
                        percentile=97.0,
                        top_p=0.005,
                        A_min=2000,
                        smooth_kernel=7,
                        bilateral=True,
                        prev_mask=None):

    # -------------------------------------------------------
    # 1. Camera Motion Removal
    # -------------------------------------------------------
    flow = remove_global_motion(flow)

    # -------------------------------------------------------
    # 2. Magnitude
    # -------------------------------------------------------
    mag = np.linalg.norm(flow, axis=2).astype(np.float32)

    # Smoothing
    if bilateral:
        mag = cv2.bilateralFilter(mag, 5, 40, 40)

    if smooth_kernel > 1:
        k = smooth_kernel if smooth_kernel % 2 == 1 else smooth_kernel + 1
        mag = cv2.GaussianBlur(mag, (k, k), 0)

    # -------------------------------------------------------
    # 3. Normalize
    # -------------------------------------------------------
    mag_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # -------------------------------------------------------
    # 4. Adaptive threshold
    # -------------------------------------------------------
    if threshold_mode == "percentile":
        th = _th_from_percentile(mag_norm, percentile)
    elif threshold_mode == "top_p":
        th = _th_from_top_p(mag_norm, top_p)
    else:
        th = _th_from_otsu(mag_norm)

    mask = (mag_norm > th).astype(np.uint8) * 255

    # -------------------------------------------------------
    # 5. Remove border noise
    # -------------------------------------------------------
    h, w = mask.shape
    border = 20
    mask[:border, :] = 0
    mask[-border:, :] = 0
    mask[:, :border] = 0
    mask[:, -border:] = 0

    # -------------------------------------------------------
    # 6. Morphology (controlled)
    # -------------------------------------------------------
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # -------------------------------------------------------
    # 7. Remove small blobs
    # -------------------------------------------------------
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    cleaned = np.zeros_like(mask)

    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= A_min:
            cleaned[labels == i] = 255

    # -------------------------------------------------------
    # 8. Safe Temporal Smoothing (only if object disappears)
    # -------------------------------------------------------
    if prev_mask is not None:
        if cleaned.sum() < 0.3 * prev_mask.sum():   # sudden drop
            cleaned = prev_mask.copy()

    return cleaned


# ============================================================
# Sequence Processing
# ============================================================
def process_sequence(flow_npy_dir, out_dir, **kwargs):

    os.makedirs(out_dir, exist_ok=True)
    files = sorted([f for f in os.listdir(flow_npy_dir) if f.endswith(".npy")])

    prev = None

    for fname in files:
        flow = np.load(os.path.join(flow_npy_dir, fname))

        mask = flow_to_motion_mask(flow, prev_mask=prev, **kwargs)
        prev = mask.copy()

        base = os.path.splitext(fname)[0]
        cv2.imwrite(os.path.join(out_dir, f"{base}_motion.png"), mask)

    print(f"Saved → {out_dir}")

# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Convert RAFT flow → motion masks (FlowI).")

    parser.add_argument("--flow_root", type=str,
                        default="./Variant_1_output/raft_flow")
    parser.add_argument("--out_root", type=str,
                        default="./Variant_1_output/flowI_motion")
    parser.add_argument("--sequence", type=str, default="bear")

    parser.add_argument("--threshold_mode", type=str, default="percentile",
                        choices=["percentile", "top_p", "otsu"])
    parser.add_argument("--percentile", type=float, default=97.0)
    parser.add_argument("--top_p", type=float, default=0.005)
    parser.add_argument("--A_min", type=int, default=2000)
    parser.add_argument("--smooth_kernel", type=int, default=7)
    parser.add_argument("--no_bilateral", action="store_true")

    args = parser.parse_args()
    bilateral = not args.no_bilateral

    # Determine sequences
    if args.sequence.lower() == "all":
        sequences = sorted(os.listdir(args.flow_root))
    elif "," in args.sequence:
        sequences = [s.strip() for s in args.sequence.split(",")]
    else:
        sequences = [args.sequence]

    print(f"Sequences: {sequences}")

    for seq in sequences:
        # Support both directory layouts
        cand1 = os.path.join(args.flow_root, seq, "flow_npy")
        cand2 = os.path.join(args.flow_root, "flow_npy", seq)

        if os.path.isdir(cand1):
            flow_npy_dir = cand1
        elif os.path.isdir(cand2):
            flow_npy_dir = cand2
        else:
            print(f"❌ Missing flow for seq {seq}")
            continue

        out_dir = os.path.join(args.out_root, seq)

        process_sequence(
            flow_npy_dir,
            out_dir,
            threshold_mode=args.threshold_mode,
            percentile=args.percentile,
            top_p=args.top_p,
            A_min=args.A_min,
            smooth_kernel=args.smooth_kernel,
            bilateral=bilateral,
        )

    print("\n✅ FlowI mask generation DONE.")


if __name__ == "__main__":
    main()
