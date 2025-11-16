import os
import cv2
import json
import argparse
import numpy as np
from glob import glob

import torch
from segment_anything import sam_model_registry, SamPredictor

# ------------------------------------------------------------
# DEVICE
# ------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------------
# DAVIS frame loader
# ------------------------------------------------------------
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


# ------------------------------------------------------------
# Small viz helpers (same spirit as baseline)
# ------------------------------------------------------------
def _letterbox_to_h(img, target_h):
    h, w = img.shape[:2]
    scale = target_h / float(h)
    new_w = int(round(w * scale))
    return cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)


def _title_bar(panel, title, font_scale=0.9, thickness=2):
    h, w = panel.shape[:2]
    bar_h = max(32, int(0.07 * h))
    out = np.zeros((h + bar_h, w, 3), dtype=np.uint8)
    out[bar_h:] = panel
    cv2.putText(out, title, (12, int(bar_h * 0.7)),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (255, 255, 255), thickness, cv2.LINE_AA)
    return out


# ------------------------------------------------------------
# FLOwI mask → bounding box helper
# ------------------------------------------------------------
def mask_to_box(mask, min_area=500):
    """
    Convert binary mask (0/255) to a single bounding box.
    Ignores border-touching blobs and tiny blobs. Returns None if no good region.
    Box format: [x1, y1, x2, y2]
    """
    h, w = mask.shape[:2]
    # Binarize
    m = (mask > 127).astype(np.uint8)

    # Remove components that touch the border (strong hint of noise)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m)
    clean = np.zeros_like(m)
    for i in range(1, num_labels):
        x, y, bw, bh, area = stats[i]
        if area < min_area:
            continue
        # Check border touch
        if x <= 0 or y <= 0 or (x + bw) >= (w - 1) or (y + bh) >= (h - 1):
            continue
        clean[labels == i] = 1

    ys, xs = np.where(clean > 0)
    if xs.size == 0:
        return None

    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())

    # Slightly pad box
    pad_x = int(0.03 * (x2 - x1 + 1))
    pad_y = int(0.03 * (y2 - y1 + 1))
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w - 1, x2 + pad_x)
    y2 = min(h - 1, y2 + pad_y)

    return np.array([x1, y1, x2, y2], dtype=np.int32)


# ------------------------------------------------------------
# SAM-based segmentation for one sequence
# ------------------------------------------------------------
def run_flowI_sam_on_sequence(
    rgb_seq_dir,
    flowI_seq_dir,
    out_dir,
    predictor: SamPredictor,
    fps=15,
    panel_height=360,
    gap=24,
    margin=24,
    overlay_color=(0, 255, 0),
    overlay_alpha=0.35,
    edge_color=(0, 0, 255),
    edge_thickness=2,
    verbose=False,
):
    """
    For a DAVIS sequence:
      - Load RGB frames
      - Load FlowI motion masks
      - Convert mask → SAM bounding box
      - Run SAM to get refined segmentation
      - Save:
          mask_XXXX.png             (SAM final masks)
          triptych.mp4              (RGB | FlowI mask | SAM mask)
          sam_segmented.mp4         (RGB + SAM overlay)
    """
    os.makedirs(out_dir, exist_ok=True)

    # --- load frames ---
    frames, names = load_davis_frames(rgb_seq_dir)
    if len(frames) == 0:
        print(f"[Warn] No frames in {rgb_seq_dir}")
        return

    h, w = frames[0].shape[:2]

    # --- map flowI masks by base name ---
    flowI_files = sorted([f for f in os.listdir(flowI_seq_dir)
                          if f.lower().endswith(".png")])
    if not flowI_files:
        print(f"[Warn] No FlowI masks in {flowI_seq_dir}")
        return

    flowI_map = {}
    for f in flowI_files:
        base = os.path.splitext(f)[0]
        # handle possible *_motion suffix
        if base.endswith("_flow_motion"):
            base_key = base.replace("_flow_motion", "")
        else:
            base_key = base
        flowI_map[base_key] = os.path.join(flowI_seq_dir, f)

    if verbose:
        print(f"[dbg] FlowI masks found: {len(flowI_map)}")

    # print(flowI_map)
    triptych_writer = None
    seg_writer = None

    for idx, (frame, name) in enumerate(zip(frames, names)):
        base = os.path.splitext(name)[0]
        # print(f"Processing frame {idx+1}/{len(frames)}: {name} : base={base}")
        # ---------- 1) Load corresponding FlowI mask ----------
        flow_mask_path = flowI_map.get(base, None)
        # print("FLOW MASK PATH:", flow_mask_path)
        if flow_mask_path is None:
            # no mask for this frame → all zeros
            flow_mask = np.zeros((h, w), np.uint8)
        else:
            flow_mask = cv2.imread(flow_mask_path, cv2.IMREAD_GRAYSCALE)
            if flow_mask is None:
                flow_mask = np.zeros((h, w), np.uint8)

        # ---------- 2) Get SAM segmentation ----------
        sam_mask_bin = np.zeros((h, w), np.uint8)

        if np.count_nonzero(flow_mask) > 0:
            # BGR → RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            predictor.set_image(rgb)

            box = mask_to_box(flow_mask)
            # print("BOX:", box)
            if box is not None:
                # SAM expects box in a batch of shape (1, 4)
                boxes = box[None, :]

                # print("bbox:", boxes)
                with torch.no_grad():
                    masks, scores, _ = predictor.predict(
                        box=boxes,
                        multimask_output=True
                    )
                # pick best
                best_id = int(np.argmax(scores))
                sam_mask = masks[best_id]  # (H,W) bool
                sam_mask_bin = (sam_mask.astype(np.uint8) * 255)
            else:
                # no good box → keep zeros
                pass

        # save mask for this frame with baseline-compatible name
        mask_name = f"mask_{name}"
        mask_path = os.path.join(out_dir, mask_name)
        cv2.imwrite(mask_path, sam_mask_bin)

        # ---------- 3) Build triptych frame ----------
        # A) Original
        panel_orig = _letterbox_to_h(frame, panel_height)
        panel_orig = _title_bar(panel_orig, "Original")

        # B) FlowI mask (white on black)
        flow_rgb = np.zeros((h, w, 3), dtype=np.uint8)

        if flow_mask.shape[:2] != flow_rgb.shape[:2]:
            flow_mask_resized = cv2.resize(
                flow_mask, 
                (flow_rgb.shape[1], flow_rgb.shape[0]), 
                interpolation=cv2.INTER_NEAREST
            )
        else:
            flow_mask_resized = flow_mask

        flow_rgb[flow_mask_resized > 127] = (255, 255, 255)

        panel_flow = _letterbox_to_h(flow_rgb, panel_height)
        panel_flow = _title_bar(panel_flow, "FlowI Motion Mask")

        # C) SAM mask (white on black)
        sam_rgb = np.zeros((h, w, 3), dtype=np.uint8)
        sam_rgb[sam_mask_bin > 127] = (255, 255, 255)
        panel_sam = _letterbox_to_h(sam_rgb, panel_height)
        panel_sam = _title_bar(panel_sam, "SAM Segmentation")

        # unify heights
        Hmax = max(panel_orig.shape[0], panel_flow.shape[0], panel_sam.shape[0])

        def pad_h(img, H):
            if img.shape[0] == H:
                return img
            pad = H - img.shape[0]
            return cv2.copyMakeBorder(img, pad // 2, pad - pad // 2,
                                      0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

        panel_orig = pad_h(panel_orig, Hmax)
        panel_flow = pad_h(panel_flow, Hmax)
        panel_sam = pad_h(panel_sam, Hmax)

        W_total = (margin * 2) + panel_orig.shape[1] + gap + panel_flow.shape[1] + gap + panel_sam.shape[1]
        canvas = np.zeros((Hmax + margin * 2, W_total, 3), dtype=np.uint8)

        x = margin
        canvas[margin:margin + panel_orig.shape[0], x:x + panel_orig.shape[1]] = panel_orig
        x += panel_orig.shape[1] + gap
        canvas[margin:margin + panel_flow.shape[0], x:x + panel_flow.shape[1]] = panel_flow
        x += panel_flow.shape[1] + gap
        canvas[margin:margin + panel_sam.shape[0], x:x + panel_sam.shape[1]] = panel_sam

        # ---------- 4) Segmented overlay ----------
        seg_frame = frame.copy().astype(np.float32)
        tint = np.zeros_like(frame, dtype=np.uint8)
        tint[:] = overlay_color

        mbool = (sam_mask_bin > 127)
        seg_frame[mbool] = (1.0 - overlay_alpha) * seg_frame[mbool] + overlay_alpha * tint[mbool]
        seg_frame = np.clip(seg_frame, 0, 255).astype(np.uint8)

        contours, _ = cv2.findContours(sam_mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(seg_frame, contours, -1, edge_color, edge_thickness, lineType=cv2.LINE_AA)

        # ---------- 5) Init writers ----------
        if triptych_writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            triptych_writer = cv2.VideoWriter(
                os.path.join(out_dir, "triptych_flowI_SAM.mp4"),
                fourcc, fps, (canvas.shape[1], canvas.shape[0])
            )

        if seg_writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            seg_writer = cv2.VideoWriter(
                os.path.join(out_dir, "sam_segmented.mp4"),
                fourcc, fps, (frame.shape[1], frame.shape[0])
            )

        triptych_writer.write(canvas)
        seg_writer.write(seg_frame)

        if verbose and (idx % 20 == 0 or idx == len(frames) - 1):
            print(f"[dbg] frame {idx+1}/{len(frames)}")

    if triptych_writer is not None:
        triptych_writer.release()
    if seg_writer is not None:
        seg_writer.release()

    print(f" FlowI-SAM videos + masks saved in {out_dir}")


# ------------------------------------------------------------
# METRICS (copied from baseline style)
# ------------------------------------------------------------
def compute_metrics(pred_mask, gt_mask):
    """
    Compute IoU, Dice, F-measure, Precision, Recall, MAE.
    pred_mask: uint8 [0..255] predicted (binary-ish)
    gt_mask:   DAVIS label mask (0=bg, 1..K=fg)
    """
    pred = (pred_mask > 127).astype(np.uint8)
    gt = (gt_mask > 0).astype(np.uint8)

    # Align size if needed
    if pred.shape != gt.shape:
        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)

    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    iou = (intersection / union) if union > 0 else 0.0

    dice = (2.0 * intersection) / (pred.sum() + gt.sum() + 1e-8)

    tp = intersection
    fp = pred.sum() - tp
    fn = gt.sum() - tp
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    fmeasure = 2 * precision * recall / (precision + recall + 1e-8)

    mae = np.abs(pred.astype(np.float32) - gt.astype(np.float32)).mean()

    return dict(IoU=iou, Dice=dice, F=fmeasure,
                Precision=precision, Recall=recall, MAE=mae)


def _collect_pred_files(pred_dir):
    patterns = [
        os.path.join(pred_dir, "mask_*.png"),
        os.path.join(pred_dir, "*.png"),
        os.path.join(pred_dir, "*.jpg"),
    ]
    for p in patterns:
        files = sorted(glob(p))
        if files:
            return files
    return []


def evaluate_sequence(pred_dir, gt_dir, verbose=False):
    pred_files = _collect_pred_files(pred_dir)
    gt_files = sorted(glob(os.path.join(gt_dir, "*.png")))

    if len(pred_files) == 0 or len(gt_files) == 0:
        print(f" Missing prediction or GT in {pred_dir} / {gt_dir}")
        return {}

    if len(pred_files) != len(gt_files):
        n = min(len(pred_files), len(gt_files))
        if verbose:
            print(f"[Info] Frame-count mismatch: pred={len(pred_files)}, gt={len(gt_files)} → using {n}")
        pred_files = pred_files[:n]
        gt_files = gt_files[:n]

    keys = ["IoU", "Dice", "F", "Precision", "Recall", "MAE"]
    agg = {k: [] for k in keys}

    for idx, (p, g) in enumerate(zip(pred_files, gt_files)):
        pred = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(g, cv2.IMREAD_GRAYSCALE)
        if pred is None or gt is None:
            continue

        m = compute_metrics(pred, gt)
        for k in keys:
            agg[k].append(m[k])

        if verbose and (idx % 25 == 0 or idx == len(pred_files) - 1):
            print(f"[dbg] #{idx:03d} IoU={m['IoU']:.3f} Dice={m['Dice']:.3f}")

    summary = {k: float(np.mean(agg[k])) if len(agg[k]) else 0.0 for k in keys}

    if verbose:
        print(f"\n FlowI-SAM Results ({os.path.basename(pred_dir)}):")
        for k in keys:
            print(f"{k}: {summary[k]:.4f}")
        print("Evaluation complete.")

    return summary


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="FlowI-SAM: use FlowI motion masks as prompts for SAM on DAVIS."
    )

    parser.add_argument("--rgb_root", type=str,
                        default="./datasets/DAVIS/JPEGImages/480p",
                        help="DAVIS RGB root")

    parser.add_argument("--flowI_root", type=str,
                        default="Variant_1_output/flowI_motion",
                        help="Root of FlowI motion masks (per sequence subfolders)")

    parser.add_argument("--gt_root", type=str,
                        default="./datasets/DAVIS/Annotations/480p",
                        help="DAVIS annotation root")

    parser.add_argument("--output_root", type=str,
                        default="./Variant_1_output/flowI_sam_output",
                        help="Where to save SAM masks + videos + metrics")

    parser.add_argument("--sequence", type=str, default="bear",
                        help="Sequence name or 'all'")

    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument("--sam_checkpoint", type=str,
                        default="./third_party/sam_checkpoint/sam_vit_h_4b8939.pth",
                        help="Path to SAM checkpoint")

    parser.add_argument("--sam_model_type", type=str,
                        default="vit_h",
                        choices=["vit_h", "vit_l", "vit_b"],
                        help="SAM model type")

    args = parser.parse_args()

    os.makedirs(args.output_root, exist_ok=True)

    # ---- Load SAM once ----
    print(f"Loading SAM ({args.sam_model_type}) from {args.sam_checkpoint} ...")
    sam = sam_model_registry[args.sam_model_type](checkpoint=args.sam_checkpoint)
    sam.to(DEVICE)
    predictor = SamPredictor(sam)
    print("SAM loaded.")

    # ---- Which sequences? ----
    if args.sequence.lower() == "all":
        sequences = sorted([d for d in os.listdir(args.rgb_root)
                           if os.path.isdir(os.path.join(args.rgb_root, d))])
    elif "," in args.sequence:
        sequences = [s.strip() for s in args.sequence.split(",")]
    else:
        sequences = [args.sequence]

    print(f"Sequences: {sequences}")

    all_metrics = []

    for seq in sequences:
        rgb_seq_dir = os.path.join(args.rgb_root, seq)
        flowI_seq_dir = os.path.join(args.flowI_root, seq)
        out_dir = os.path.join(args.output_root, seq)

        if not os.path.isdir(rgb_seq_dir):
            print(f"[Warn] RGB sequence dir missing: {rgb_seq_dir}")
            continue
        if not os.path.isdir(flowI_seq_dir):
            print(f"[Warn] FlowI motion dir missing: {flowI_seq_dir}")
            continue

        print(f"\n=== Processing sequence: {seq} ===")
        run_flowI_sam_on_sequence(
            rgb_seq_dir=rgb_seq_dir,
            flowI_seq_dir=flowI_seq_dir,
            out_dir=out_dir,
            predictor=predictor,
            fps=args.fps,
            verbose=args.verbose,
        )

        # ---- Evaluate ----
        gt_dir = os.path.join(args.gt_root, seq)
        if os.path.isdir(gt_dir):
            summary = evaluate_sequence(out_dir, gt_dir, verbose=args.verbose)
            if summary:
                all_metrics.append(summary)
                # write per-sequence metrics
                json_path = os.path.join(out_dir, "metrics_summary.json")
                with open(json_path, "w") as f:
                    json.dump(summary, f, indent=4)
        else:
            print(f"[Warn] GT dir not found: {gt_dir} (skipping evaluation)")

    # ---- Global metrics_summary ----
    if all_metrics:
        keys = list(all_metrics[0].keys())
        avg = {k: float(np.mean([m[k] for m in all_metrics])) for k in keys}
        print("\n================ FLOWI-SAM DATASET AVERAGE ================")
        for k in keys:
            print(f"{k}: {avg[k]:.4f}")

        # add meta info
        avg["sam_model_type"] = args.sam_model_type
        avg["sam_checkpoint"] = args.sam_checkpoint
        avg_json_path = os.path.join(args.output_root, "metrics_summary.json")
        with open(avg_json_path, "w") as f:
            json.dump(avg, f, indent=4)
    else:
        print("\n[Warn] No sequences evaluated.")

    print("\n FlowI-SAM pipeline complete.")


if __name__ == "__main__":
    main()
