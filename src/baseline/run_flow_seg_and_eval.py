import os
import cv2
import argparse
import numpy as np
from glob import glob
import json
# -----------------------
# Utilities
# -----------------------
def load_davis_frames(sequence_dir):
    names = sorted([f for f in os.listdir(sequence_dir)
                    if f.lower().endswith((".jpg", ".png"))])
    frames = []
    good_names = []
    for n in names:
        img = cv2.imread(os.path.join(sequence_dir, n))
        if img is not None:
            frames.append(img)
            good_names.append(n)
    return frames, good_names


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


# ---------- threshold helpers ----------
def _th_from_percentile(mag_norm: np.ndarray, percentile: float) -> float:
    return float(np.percentile(mag_norm, percentile))

def _th_from_top_p(mag_norm: np.ndarray, p: float) -> float:
    # keep top-p fraction of (nonzero) motion pixels
    arr = mag_norm[mag_norm > 0]
    if arr.size < 50:
        arr = mag_norm.flatten()
    return float(np.percentile(arr, 100.0 * (1.0 - p)))

def _th_from_otsu(mag_norm: np.ndarray) -> float:
    t,_  = cv2.threshold(mag_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # print(t.shape, mag_norm.shape)
    # print("K : ", k)
    return float(t)

# -----------------------
# Segmentation via optical flow + videos
# -----------------------

def _letterbox_to_h(img, target_h):
    h, w = img.shape[:2]
    scale = target_h / float(h)
    new_w = int(round(w * scale))
    return cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)

def _title_bar(panel, title, font_scale=0.9, thickness=2):
    h, w = panel.shape[:2]
    bar_h = max(32, int(0.07*h))
    out = np.zeros((h + bar_h, w, 3), dtype=np.uint8)
    out[bar_h:] = panel
    cv2.putText(out, title, (12, int(bar_h*0.7)),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), thickness, cv2.LINE_AA)
    return out

def farneback_davis(sequence_dir, out_dir,
                    percentile=94, smooth_kernel=7, A_min=1500, fps=15,
                    panel_height=360, gap=24, margin=24,
                    overlay_color=(0, 255, 255),  # cyan
                    overlay_alpha=0.35, edge_color=(0, 0, 255), edge_thickness=2,
                    threshold_mode="percentile", top_p=0.01,
                    verbose=False):
    """
    Compute optical flow on DAVIS sequences with stabilization.

    Saves:
      - triptych.mp4  : [Original | Flow Heatmap | Motion Mask]
      - segmented.mp4 : original with colored overlay + contours
      - mask_*.png    : per-frame binary motion masks (for evaluation)
    """
    os.makedirs(out_dir, exist_ok=True)
    frames, names = load_davis_frames(sequence_dir)
    if len(frames) < 2:
        print("Not enough frames for optical flow.")
        return

    h, w = frames[0].shape[:2]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # init previous gray and write dummy first mask to keep counts aligned with GT
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(out_dir, f"mask_{names[0]}"), np.zeros((h, w), np.uint8))

    triptych_writer = None
    segmented_writer = None
    
    for i in range(1, len(frames)):
        frame = frames[i]
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1) Stabilize previous → current
        stab_prev = compensate_camera_motion(prev_gray, gray)

        # 2) Optical flow
        flow = cv2.calcOpticalFlowFarneback(
            stab_prev, gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )

        # 3) Magnitude + smoothing
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mag = cv2.bilateralFilter(mag.astype(np.float32), 9, 50, 50)
        mag = cv2.GaussianBlur(mag, (smooth_kernel, smooth_kernel), 0)
        mag_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # 4) Threshold + cleanup => segmentation mask
        # th = np.percentile(mag_norm, percentile)
        if threshold_mode == "percentile":
            th = _th_from_percentile(mag_norm, percentile)
        elif threshold_mode == "top_p":
            th = _th_from_top_p(mag_norm, top_p)
        elif threshold_mode == "otsu":
            th = _th_from_otsu(mag_norm)
        else:
            raise ValueError(f"Unknown threshold_mode: {threshold_mode}")
        
        mask = (mag_norm > th).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        filtered = np.zeros_like(mask)
        for j in range(1, num_labels):
            if stats[j, cv2.CC_STAT_AREA] >= A_min:
                filtered[labels == j] = 255

        # save mask
        cv2.imwrite(os.path.join(out_dir, f"mask_{names[i]}"), filtered)

        # ---------- A) build TRIPTYCH frame ----------
        # Original
        panel_orig = _letterbox_to_h(frame, panel_height)
        panel_orig = _title_bar(panel_orig, "Original")

        # Heatmap (use TURBO for nicer gradient; switch to JET if preferred)
        heatmap = cv2.applyColorMap(mag_norm, cv2.COLORMAP_TURBO)
        panel_heat = _letterbox_to_h(heatmap, panel_height)
        panel_heat = _title_bar(panel_heat, "Flow Heatmap")

        # Binary mask (white on black)
        mask_rgb = np.zeros((filtered.shape[0], filtered.shape[1], 3), dtype=np.uint8)
        mask_rgb[filtered == 255] = (255, 255, 255)
        panel_mask = _letterbox_to_h(mask_rgb, panel_height)
        panel_mask = _title_bar(panel_mask, "Motion Mask")

        # make same height
        H = max(panel_orig.shape[0], panel_heat.shape[0], panel_mask.shape[0])
        def pad_h(img, H):
            if img.shape[0] == H: return img
            pad = H - img.shape[0]
            return cv2.copyMakeBorder(img, pad//2, pad - pad//2, 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))
        panel_orig = pad_h(panel_orig, H)
        panel_heat = pad_h(panel_heat, H)
        panel_mask = pad_h(panel_mask, H)

        # black canvas with margins and gaps
        W_total = (margin*2) + panel_orig.shape[1] + gap + panel_heat.shape[1] + gap + panel_mask.shape[1]
        canvas  = np.zeros((H + margin*2, W_total, 3), dtype=np.uint8)
        x = margin
        canvas[margin:margin+panel_orig.shape[0], x:x+panel_orig.shape[1]] = panel_orig; x += panel_orig.shape[1] + gap
        canvas[margin:margin+panel_heat.shape[0], x:x+panel_heat.shape[1]] = panel_heat; x += panel_heat.shape[1] + gap
        canvas[margin:margin+panel_mask.shape[0], x:x+panel_mask.shape[1]] = panel_mask

        # ---------- B) build SEGMENTED overlay frame ----------
        seg_frame = frame.copy().astype(np.float32)
        tint = np.zeros_like(frame, dtype=np.uint8); tint[:] = overlay_color
        mbool = (filtered == 255)
        seg_frame[mbool] = (1.0 - overlay_alpha) * seg_frame[mbool] + overlay_alpha * tint[mbool]
        seg_frame = np.clip(seg_frame, 0, 255).astype(np.uint8)
        # crisp contours
        contours, _ = cv2.findContours(filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(seg_frame, contours, -1, edge_color, edge_thickness, lineType=cv2.LINE_AA)

        # ---------- init writers and write frames ----------
        if triptych_writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            triptych_writer = cv2.VideoWriter(os.path.join(out_dir, "triptych.mp4"),
                                              fourcc, fps, (canvas.shape[1], canvas.shape[0]))
        if segmented_writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            segmented_writer = cv2.VideoWriter(os.path.join(out_dir, "segmented.mp4"),
                                               fourcc, fps, (frame.shape[1], frame.shape[0]))

        triptych_writer.write(canvas)
        segmented_writer.write(seg_frame)

        prev_gray = gray

    if triptych_writer is not None: triptych_writer.release()
    if segmented_writer is not None: segmented_writer.release()

    if verbose :
        print(f"✅ Saved videos:\n  - {os.path.join(out_dir,'triptych.mp4')}\n  - {os.path.join(out_dir,'segmented.mp4')}")
        print(f"🖼️  Per-frame masks: {out_dir}")
    
    print(f"✅ Baseline motion segmentation done for {sequence_dir}")



# -----------------------
# Evaluation (DAVIS)
# -----------------------
def compute_metrics(pred_mask, gt_mask):
    """
    Compute IoU, Dice, F-measure, Precision, Recall, MAE.
    pred_mask: uint8 [0..255] predicted (binary-ish)
    gt_mask:   DAVIS label mask (0=bg, 1..K=fg)
    """
    # Binarize
    pred = (pred_mask > 127).astype(np.uint8)
    gt   = (gt_mask   > 0  ).astype(np.uint8)  # IMPORTANT for DAVIS

    # Align size if needed
    if pred.shape != gt.shape:
        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)

    intersection = np.logical_and(pred, gt).sum()
    union        = np.logical_or(pred,  gt).sum()
    iou = (intersection / union) if union > 0 else 0.0

    dice = (2.0 * intersection) / (pred.sum() + gt.sum() + 1e-8)

    tp = intersection
    fp = pred.sum() - tp
    fn = gt.sum()   - tp
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    fmeasure  = 2 * precision * recall / (precision + recall + 1e-8)

    mae = np.abs(pred.astype(np.float32) - gt.astype(np.float32)).mean()

    return dict(IoU=iou, Dice=dice, F=fmeasure,
                Precision=precision, Recall=recall, MAE=mae)


def _collect_pred_files(pred_dir):
    # Prefer our saved naming; fallbacks just in case
    pats = [
        os.path.join(pred_dir, "mask_*.png"),
        os.path.join(pred_dir, "*.png"),
        os.path.join(pred_dir, "*.jpg"),
    ]
    for p in pats:
        files = sorted(glob(p))
        if files:
            return files
    return []


def evaluate_sequence(pred_dir, gt_dir, verbose=False):
    """
    Evaluate predicted masks against DAVIS GT masks.
    - trims to shortest list when counts mismatch
    - resizes pred to GT size if needed
    """
    pred_files = _collect_pred_files(pred_dir)
    gt_files   = sorted(glob(os.path.join(gt_dir, "*.png")))

    if verbose :
        print(f"pred_files: {len(pred_files)} from {pred_dir}")
        print(f"gt_files:   {len(gt_files)} from {gt_dir}")

    if len(pred_files) == 0 or len(gt_files) == 0:
        print("❌ Missing prediction or GT files.")
        return {}

    if len(pred_files) != len(gt_files):
        print(f"[Info] Frame count mismatch: pred={len(pred_files)}, gt={len(gt_files)}")
        n = min(len(pred_files), len(gt_files))
        pred_files = pred_files[:n]
        gt_files   = gt_files[:n]
        print(f"[Info] Evaluating first {n} frame pairs only.")

    keys = ["IoU", "Dice", "F", "Precision", "Recall", "MAE"]
    agg  = {k: [] for k in keys}

    for idx, (p, g) in enumerate(zip(pred_files, gt_files)):
        pred = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        gt   = cv2.imread(g, cv2.IMREAD_GRAYSCALE)
        if pred is None or gt is None:
            if verbose:
                print(f"[Warn] Skipping unreadable: {p} | {g}")
            continue

        m = compute_metrics(pred, gt)
        for k in keys: agg[k].append(m[k])

        if verbose and (idx % 25 == 0 or idx == len(pred_files) - 1):
            print(f"[dbg] #{idx:03d} IoU={m['IoU']:.3f} Dice={m['Dice']:.3f}")

    summary = {k: float(np.mean(agg[k])) if len(agg[k]) else 0.0 for k in keys}

    if verbose :
        print(f"\n📊 Evaluation Results ({os.path.basename(pred_dir)}):")
        for k in keys:
            print(f"{k}: {summary[k]:.4f}")
        print("✅ Evaluation complete.")
    return summary


# -----------------------
# CLI
# -----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Optical-flow baseline: segment + videos + evaluation (DAVIS)")
    
    input_root_default = "datasets/DAVIS/JPEGImages/480p"
    parser.add_argument("--input_root",  type=str, default=input_root_default,
                        help="DAVIS sequences root (e.g., datasets/DAVIS/JPEGImages/480p)")
    
    sequence_default = "bear"
    parser.add_argument("--sequence",    type=str, default=sequence_default,
                        help="Sequence name (e.g., bear, bmx-bumps)")
    
    output_root_default = "./outputs_baseline"
    parser.add_argument("--output_root", type=str, default=output_root_default,
                        help="Where to save outputs")
    gt_root_default = "datasets/DAVIS/Annotations/480p"
    parser.add_argument("--gt_root",     type=str, default=gt_root_default,
                        help="DAVIS Annotations root (e.g., datasets/DAVIS/Annotations/480p)")
    parser.add_argument("--smooth_kernel", type=int, default=7)
    parser.add_argument("--A_min",       type=int, default=1500)
    parser.add_argument("--fps",         type=int, default=15)
    parser.add_argument("--verbose",     action="store_true")
    parser.add_argument("--threshold_mode", type=str, default="percentile",
                    choices=["percentile", "top_p", "otsu"])
    parser.add_argument("--percentile",  type=float, default=94.0)
    parser.add_argument("--top_p",       type=float, default=0.01)
    args = parser.parse_args()

    if args.sequence != "all" :
        
        seq_dir = os.path.join(args.input_root, args.sequence)
        out_dir = os.path.join(args.output_root,"sample", os.path.join(args.threshold_mode,f"{args.sequence}_baseline_motion"))
        os.makedirs(out_dir, exist_ok=True)

        # Segment + videos
        farneback_davis(
            sequence_dir=seq_dir, out_dir=out_dir,
            percentile=args.percentile, smooth_kernel=args.smooth_kernel,
            A_min=args.A_min, fps=args.fps,
            threshold_mode=args.threshold_mode, top_p=args.top_p
        )

        # Evaluate
        if args.gt_root is not None:
            gt_dir = os.path.join(args.gt_root, args.sequence)
            if os.path.isdir(gt_dir):
                evaluate_sequence(out_dir, gt_dir, verbose=args.verbose)
            else:
                print(f"[Warn] GT dir not found: {gt_dir} (skipping evaluation)")
    else :

        seq_dirs = [d for d in sorted(os.listdir(args.input_root))
                if os.path.isdir(os.path.join(args.input_root, d))]
        all_metrics = []
        for seq in seq_dirs:
            seq_dir = os.path.join(args.input_root, seq)
            out_dir = os.path.join(args.output_root, os.path.join(args.threshold_mode,f"{seq}_baseline_motion"))
            os.makedirs(out_dir, exist_ok=True)

            print(f"\nProcessing sequence: {seq}")
            farneback_davis(
                sequence_dir=seq_dir, out_dir=out_dir,
                percentile=args.percentile, smooth_kernel=args.smooth_kernel,
                A_min=args.A_min, fps=args.fps,
                threshold_mode=args.threshold_mode, top_p=args.top_p
            )

            gt_dir = os.path.join(args.gt_root, seq)
            if os.path.isdir(gt_dir):
                summary = evaluate_sequence(out_dir, gt_dir, verbose=args.verbose)
                if summary:
                    all_metrics.append(summary)
                    json_path = os.path.join(out_dir, "metrics_summary.json")
                    with open(json_path, "w") as f:
                        json.dump(summary, f, indent=4)
            else:
                print(f"[Warn] GT dir not found: {gt_dir} (skipping evaluation)")

        # ---- dataset average ----
        if all_metrics:
            keys = list(all_metrics[0].keys())
            avg = {k: float(np.mean([m[k] for m in all_metrics])) for k in keys}
            print("\n================ DATASET AVERAGE ================")
            for k in keys:
                print(f"{k}: {avg[k]:.4f}")
            
            avg_json_path = os.path.join(args.output_root, args.threshold_mode, "dataset_metrics_average.json")
            avg['Threshold_mode'] = args.threshold_mode
            avg['Percentile'] = args.percentile
            avg['Top_p'] = args.top_p
            avg['Smooth_kernel'] = args.smooth_kernel
            avg['A_min'] = args.A_min
            avg['FPS'] = args.fps
            with open(avg_json_path, "w") as f:
                json.dump(avg, f, indent=4)
        else:
            print("\n[Warn] No sequences evaluated.")
    
    print("✅ All done.")
