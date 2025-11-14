# import cv2
# import numpy as np
# import os
# from glob import glob
# import argparse
# from baseline_segmentation import baseline_motion_segmentation

# def compute_metrics(pred_mask, gt_mask):
#     """Compute IoU, Dice, F-measure, Precision, Recall, MAE."""
#     pred = (pred_mask > 127).astype(np.uint8)
#     gt = (gt_mask > 127).astype(np.uint8)

#     intersection = np.logical_and(pred, gt).sum()
#     union = np.logical_or(pred, gt).sum()
#     iou = intersection / union if union > 0 else 0

#     dice = 2 * intersection / (pred.sum() + gt.sum() + 1e-8)

#     tp = intersection
#     fp = pred.sum() - tp
#     fn = gt.sum() - tp
#     precision = tp / (tp + fp + 1e-8)
#     recall = tp / (tp + fn + 1e-8)
#     fmeasure = 2 * precision * recall / (precision + recall + 1e-8)

#     mae = np.abs(pred - gt).mean()

#     return dict(IoU=iou, Dice=dice, F=fmeasure,
#                 Precision=precision, Recall=recall, MAE=mae)


# def evaluate_sequence(pred_dir, gt_dir):
#     """
#     Evaluate motion segmentation masks against DAVIS ground truth.
#     Automatically handles unequal frame counts between predictions and GT.
#     """
#     pred_files = sorted(glob(pred_dir + "/*.jpg"))
#     gt_files = sorted(glob(os.path.join(gt_dir, "*.png")))

#     print("pred_files:", len(pred_files), "pred_dir:", pred_dir)
#     if len(pred_files) != len(gt_files):
#         print(f"[Info] Frame count mismatch: pred={len(pred_files)}, gt={len(gt_files)}")
#         min_len = min(len(pred_files), len(gt_files))
#         pred_files = pred_files[:min_len]
#         gt_files = gt_files[:min_len]
#         print(f"[Info] Evaluating first {min_len} frame pairs only.")

#     if len(pred_files) == 0:
#         print("❌ No matching prediction or GT files found.")
#         return

#     all_metrics = {k: [] for k in ["IoU", "Dice", "F", "Precision", "Recall", "MAE"]}

#     for p, g in zip(pred_files, gt_files):
#         pred = cv2.imread(p, 0)
#         gt = cv2.imread(g, 0)
#         if pred is None or gt is None:
#             continue
#         metrics = compute_metrics(pred, gt)
#         for k, v in metrics.items():
#             all_metrics[k].append(v)

#     summary = {k: np.mean(v) for k, v in all_metrics.items() if len(v) > 0}

#     print(f"\n📊 Evaluation Results ({os.path.basename(pred_dir)}):")
#     for k, v in summary.items():
#         print(f"{k}: {v:.4f}")

#     return summary


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     sequence_default = "bear"
#     parser.add_argument("--sequence", type=str, default=sequence_default, help="DAVIS sequence name")
#     input_root_default = "datasets/DAVIS/JPEGImages/480p"
#     parser.add_argument("--input_root", type=str, default=input_root_default)
#     output_root_default = "./outputs"
#     parser.add_argument("--output_root", type=str, default=output_root_default)
#     args = parser.parse_args()
#     in_seq = os.path.join(args.input_root, args.sequence)
#     baseline_seq = os.path.join(args.output_root, args.sequence + "_baseline_motion")
#     print("Running baseline motion segmentation...")
#     baseline_motion_segmentation(
#         sequence_dir=in_seq,
#         out_dir=baseline_seq,
#         percentile=90,
#         smooth_kernel=7,
#         A_min=1500
#     )

#     annotations_seq = os.path.join("datasets/DAVIS/Annotations/480p", args.sequence)
#     print("Evaluating segmentation results...")
#     evaluate_sequence(
#         pred_dir=baseline_seq,
#         gt_dir=annotations_seq
#     )
#     print("✅ Evaluation complete.")

import cv2
import numpy as np
import os
from glob import glob
import argparse
from baseline_segmentation import baseline_motion_segmentation

def compute_metrics(pred_mask, gt_mask):
    """Compute IoU, Dice, F-measure, Precision, Recall, MAE."""
    # --- Binarize ---
    pred = (pred_mask > 127).astype(np.uint8)
    # DAVIS annotations are label images (0 bg, 1..K fg) -> binarize with >0
    gt   = (gt_mask   > 0  ).astype(np.uint8)

    # --- Align spatial size if needed ---
    if pred.shape != gt.shape:
        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)

    intersection = np.logical_and(pred, gt).sum()
    union        = np.logical_or(pred,  gt).sum()
    iou = intersection / union if union > 0 else 0.0

    dice = 2 * intersection / (pred.sum() + gt.sum() + 1e-8)

    tp = intersection
    fp = pred.sum() - tp
    fn = gt.sum() - tp
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    fmeasure  = 2 * precision * recall / (precision + recall + 1e-8)

    mae = np.abs(pred.astype(np.float32) - gt.astype(np.float32)).mean()

    return dict(IoU=iou, Dice=dice, F=fmeasure,
                Precision=precision, Recall=recall, MAE=mae)


def evaluate_sequence(pred_dir, gt_dir, verbose=False):
    """
    Evaluate motion segmentation masks against DAVIS ground truth.
    Automatically handles unequal frame counts between predictions and GT.
    """
    # Look for the masks your baseline wrote
    pred_files = sorted(glob(os.path.join(pred_dir, "mask_*.png")))
    # Fallbacks if naming differs
    if len(pred_files) == 0:
        pred_files = sorted(glob(os.path.join(pred_dir, "*.png")))
    if len(pred_files) == 0:
        pred_files = sorted(glob(os.path.join(pred_dir, "*.jpg")))

    gt_files = sorted(glob(os.path.join(gt_dir, "*.png")))

    print("pred_files:", len(pred_files), "pred_dir:", pred_dir)
    print("gt_files:  ", len(gt_files),   "gt_dir:", gt_dir)

    if len(pred_files) == 0 or len(gt_files) == 0:
        print("❌ No matching prediction or GT files found.")
        return {}

    if len(pred_files) != len(gt_files):
        print(f"[Info] Frame count mismatch: pred={len(pred_files)}, gt={len(gt_files)}")
        min_len = min(len(pred_files), len(gt_files))
        pred_files = pred_files[:min_len]
        gt_files   = gt_files[:min_len]
        print(f"[Info] Evaluating first {min_len} frame pairs only.")

    all_metrics = {k: [] for k in ["IoU", "Dice", "F", "Precision", "Recall", "MAE"]}

    for idx, (p, g) in enumerate(zip(pred_files, gt_files)):
        pred = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        gt   = cv2.imread(g, cv2.IMREAD_GRAYSCALE)
        if pred is None or gt is None:
            if verbose:
                print(f"[Warn] Skipping unreadable: {p} | {g}")
            continue

        metrics = compute_metrics(pred, gt)
        for k, v in metrics.items():
            all_metrics[k].append(v)

        if verbose and (idx % 25 == 0 or idx == len(pred_files)-1):
            print(f"[dbg] #{idx:03d} IoU={metrics['IoU']:.3f} Dice={metrics['Dice']:.3f}")

    summary = {k: float(np.mean(v)) for k, v in all_metrics.items() if len(v) > 0}

    print(f"\n📊 Evaluation Results ({os.path.basename(pred_dir)}):")
    for k, v in summary.items():
        print(f"{k}: {v:.4f}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sequence_default = "bear"
    parser.add_argument("--sequence", type=str, default=sequence_default, help="DAVIS sequence name")
    input_root_default = "datasets/DAVIS/JPEGImages/480p"
    parser.add_argument("--input_root", type=str, default=input_root_default)
    output_root_default = "./outputs"
    parser.add_argument("--output_root", type=str, default=output_root_default)
    parser.add_argument("--percentile", type=float, default=90.0)
    parser.add_argument("--smooth_kernel", type=int, default=7)
    parser.add_argument("--A_min", type=int, default=1500)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    in_seq = os.path.join(args.input_root, args.sequence)
    baseline_seq = os.path.join(args.output_root, args.sequence + "_baseline_motion")

    print("Running baseline motion segmentation...")
    baseline_motion_segmentation(
        sequence_dir=in_seq,
        out_dir=baseline_seq,
        percentile=args.percentile,
        smooth_kernel=args.smooth_kernel,
        A_min=args.A_min
    )

    annotations_seq = os.path.join("datasets/DAVIS/Annotations/480p", args.sequence)
    print("Evaluating segmentation results...")
    evaluate_sequence(
        pred_dir=baseline_seq,
        gt_dir=annotations_seq,
        verbose=args.verbose
    )
    print("✅ Evaluation complete.")
