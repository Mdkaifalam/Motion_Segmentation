"""
raft_check_1.py  (MULTI-GAP RAFT FLOW GENERATOR for DAVIS)
----------------------------------------------------------
Computes RAFT optical flow for entire DAVIS sequences, using
multi-gap flow + balanced camera-motion removal to reduce
noise while keeping object motion for FlowI-SAM.

Key features:
- Uses RAFT (pretrained raft-things.pth)
- For each frame index t, averages flows:
      t -> t+1, t -> t-1, t -> t+2, t -> t-2   (valid ones)
- Removes *global* camera motion using RANSAC on mostly-static pixels
- Temporal smoothing over time so objects don't "blink"
- Mixed precision for speed

Saves:
    - flow_npy/  : H x W x 2 raw flow (camera-stabilized + temporally smoothed)
    - flow_rgb/  : color-coded flow (for inspection)
    - <seq>_flow_video.mp4 : visualization

Default paths:
- DAVIS root : datasets/DAVIS/JPEGImages/480p
- RAFT ckpt  : third_party/RAFT/models/raft-things.pth
- Output     : ./Variant_1_output/raft_flow_outputs_1
"""

import os
import argparse
import sys

import cv2
import numpy as np
import torch
from argparse import Namespace
from PIL import Image

# Workaround for Intel OMP duplicate load on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# --------------------------------------------------------
# FIX RAFT IMPORT FOR UPDATED REPO STRUCTURE
# --------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))      # /src/Variant_1
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
RAFT_PATH = os.path.join(PROJECT_ROOT, "third_party", "RAFT", "core")

print("Using RAFT from:", RAFT_PATH)
sys.path.insert(0, RAFT_PATH)

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------
# Helper functions
# --------------------------------------------------------

def load_rgb_image_tensor(path: str) -> torch.Tensor:
    """Loads a DAVIS frame into a PyTorch tensor in RGB format."""
    img = np.array(Image.open(path)).astype(np.uint8)
    tensor = torch.from_numpy(img).permute(2, 0, 1).float()[None].to(DEVICE) / 255.0
    return tensor


def remove_module_prefix(state_dict):
    """Remove 'module.' prefix added by DataParallel checkpoints."""
    new_state = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state[k.replace("module.", "")] = v
        else:
            new_state[k] = v
    return new_state


@torch.no_grad()
def run_raft(model, img1, img2, iters=32):
    """Computes dense flow between two frames."""
    padder = InputPadder(img1.shape)
    img1, img2 = padder.pad(img1, img2)
    flow_low, flow_up = model(img1, img2, iters=iters, test_mode=True)
    flow = flow_up[0].permute(1, 2, 0).cpu().numpy()  # H x W x 2
    return flow


def remove_global_motion(
    flow: np.ndarray,
    bg_percentile: float = 70.0,
    sample_ratio: float = 0.05,
    min_bg_points: int = 2000,
    ransac_thresh: float = 3.0,
    global_weight: float = 0.8,
) -> np.ndarray:
    """
    Balanced global camera motion removal.

    Steps:
    1. Compute flow magnitude and treat the *lowest* bg_percentile as "background"
    2. Run RANSAC affine (fallback homography) on only those background points
    3. Estimate a smooth global flow field and subtract (global_weight * global_flow)

    This preserves object motion while greatly reducing camera-induced background motion.
    """
    H, W, _ = flow.shape

    # pixel coordinate grid (x, y)
    xs, ys = np.meshgrid(np.arange(W), np.arange(H))
    pts1 = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float32)  # (N, 2)
    pts2 = pts1 + flow.reshape(-1, 2)                                     # (N, 2)

    # Flow magnitude
    mag = np.linalg.norm(flow.reshape(-1, 2), axis=1)                     # (N,)

    # Select mostly-static pixels as background candidates
    thr = np.percentile(mag, bg_percentile)
    bg_mask = mag < thr
    bg_idx = np.where(bg_mask)[0]

    if bg_idx.size < min_bg_points:
        # Not enough clear background; use all pixels as fallback
        bg_idx = np.arange(mag.shape[0])

    # Random subsample for speed + robustness
    N_bg = bg_idx.size
    sample_size = min(max(int(N_bg * sample_ratio), 2000), N_bg)
    sel = np.random.choice(bg_idx, size=sample_size, replace=False)

    pts1_s = pts1[sel]
    pts2_s = pts2[sel]

    # ---- First try affine RANSAC ----
    M, inliers = cv2.estimateAffine2D(
        pts1_s, pts2_s,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_thresh
    )

    if M is not None:
        pts2_pred = (pts1 @ M[:, :2].T + M[:, 2]).reshape(H, W, 2)
    else:
        # ---- Fallback: try homography ----
        Hmat, inliers = cv2.findHomography(
            pts1_s, pts2_s,
            method=cv2.RANSAC,
            ransacReprojThreshold=ransac_thresh
        )

        if Hmat is None:
            # Give up: keep original flow
            return flow

        pts1_reshaped = pts1.reshape(1, -1, 2)
        pts2_pred = cv2.perspectiveTransform(pts1_reshaped, Hmat).reshape(H, W, 2)

    # Smooth predicted global motion field to avoid blocky artifacts
    pts2_pred = cv2.GaussianBlur(pts2_pred, (0, 0), sigmaX=1.5)

    # Global flow = camera motion
    global_flow = pts2_pred - pts1.reshape(H, W, 2)

    # Subtract majority of global flow, keep some to avoid over-correcting
    residual = flow - global_weight * global_flow

    return residual


@torch.no_grad()
def compute_multi_gap_flow(model, rgb_frames, t, iters=32):
    """
    Computes multi-gap RAFT flow by averaging:
        t -> t+1, t -> t-1, t -> t+2, t -> t-2
    Only valid gaps (within sequence bounds) are used.
    Falls back to t -> t+1 if necessary.
    """
    H, W, _ = rgb_frames[t].shape
    gaps = [1, -1, 2, -2]
    flows = []

    for g in gaps:
        t2 = t + g
        if t2 < 0 or t2 >= len(rgb_frames):
            continue

        img1 = torch.from_numpy(rgb_frames[t]).permute(2, 0, 1).float()[None].to(DEVICE) / 255.0
        img2 = torch.from_numpy(rgb_frames[t2]).permute(2, 0, 1).float()[None].to(DEVICE) / 255.0

        flow = run_raft(model, img1, img2, iters=iters)
        flows.append(flow)

    # Fallback: at least compute t -> t+1
    if len(flows) == 0:
        t2 = min(t + 1, len(rgb_frames) - 1)
        img1 = torch.from_numpy(rgb_frames[t]).permute(2, 0, 1).float()[None].to(DEVICE) / 255.0
        img2 = torch.from_numpy(rgb_frames[t2]).permute(2, 0, 1).float()[None].to(DEVICE) / 255.0
        return run_raft(model, img1, img2, iters=iters)

    flows = np.stack(flows, axis=0)  # (num_gaps, H, W, 2)
    return np.mean(flows, axis=0)


# Optional sharpening for visualization
def sharpen_flow_image(image):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)


# --------------------------------------------------------
# Main
# --------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(description="Compute multi-gap RAFT optical flow for DAVIS sequences")

    parser.add_argument("--davis_root", type=str,
                        default="datasets/DAVIS/JPEGImages/480p")
    parser.add_argument("--sequence", type=str, default="bear",
                        help="Sequence name OR comma-separated list OR 'all'")
    parser.add_argument("--model_path", type=str,
                        default="third_party/RAFT/models/raft-things.pth")
    parser.add_argument("--out_dir", type=str,
                        default="./Variant_1_output/raft_flow_outputs_1")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--iters", type=int, default=32,
                        help="RAFT iterations (default: 32 for clearer flow)")

    args = parser.parse_args()

    # --------------------------------------------------------
    # Load RAFT model
    # --------------------------------------------------------
    print("\nLoading RAFT model...")

    args_raft = Namespace(
        small=False,              # use full model for cleanest flow
        mixed_precision=True,     # smoother gradients
        alternate_corr=False      # must be False on Windows (no alt_cuda_corr)
    )

    model = RAFT(args_raft).to(DEVICE).eval()

    ckpt = torch.load(args.model_path, map_location=DEVICE)
    ckpt = remove_module_prefix(ckpt)
    model.load_state_dict(ckpt, strict=True)

    print("RAFT loaded successfully with multi-gap + balanced camera-motion settings.\n")

    # --------------------------------------------------------
    # Determine sequences to process
    # --------------------------------------------------------
    if args.sequence.lower() == "all":
        sequences = sorted(os.listdir(args.davis_root))
        print(f"Running RAFT on ALL sequences: {len(sequences)} found.\n")

    elif "," in args.sequence:
        sequences = [s.strip() for s in args.sequence.split(",")]
        print(f"Running RAFT on MULTIPLE sequences: {sequences}\n")

    else:
        sequences = [args.sequence]
        print(f"Running RAFT on sequence: {args.sequence}\n")

    # --------------------------------------------------------
    # Process each sequence
    # --------------------------------------------------------
    for seq in sequences:
        seq_dir = os.path.join(args.davis_root, seq)

        if not os.path.isdir(seq_dir):
            print(f"❌ Sequence not found: {seq_dir}")
            continue

        frames = sorted([f for f in os.listdir(seq_dir) if f.endswith(".jpg")])
        if len(frames) < 2:
            print(f"⚠️  Not enough frames in '{seq}', skipping.\n")
            continue

        print(f"📂 Sequence '{seq}' — {len(frames)} frames")

        # Load all frames into memory (RGB) for multi-gap flow
        rgb_frames = []
        for f in frames:
            bgr = cv2.imread(os.path.join(seq_dir, f), cv2.IMREAD_COLOR)
            if bgr is None:
                raise FileNotFoundError(f"Could not read frame: {os.path.join(seq_dir, f)}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb_frames.append(rgb)

        # Output folders
        out_rgb = os.path.join(args.out_dir, seq, "flow_rgb")
        out_npy = os.path.join(args.out_dir, seq, "flow_npy")
        os.makedirs(out_rgb, exist_ok=True)
        os.makedirs(out_npy, exist_ok=True)

        # Video writer
        video_path = os.path.join(args.out_dir, seq, f"{seq}_flow_video.mp4")
        video_writer = None

        # Temporal smoothing buffer
        prev_flow = None
        temporal_alpha = 0.7   # weight for current flow vs previous

        # Run RAFT through sequence (per-index flow, using multi-gap)
        for i in range(len(frames) - 1):
            f1_name = frames[i]
            print(f"   RAFT (multi-gap + cam-removal): {f1_name} → neighbors (+/-1,+/-2)")

            # Multi-gap RAFT flow centered at frame i
            flow_raw = compute_multi_gap_flow(model, rgb_frames, i, iters=args.iters)

            # Balanced camera-motion removal
            flow_cam_removed = remove_global_motion(flow_raw)

            # Temporal smoothing to prevent blinking
            if prev_flow is None:
                flow_final = flow_cam_removed
            else:
                flow_final = temporal_alpha * flow_cam_removed + (1.0 - temporal_alpha) * prev_flow

            prev_flow = flow_final

            # Save NPY (float32)
            npy_path = os.path.join(out_npy, f1_name.replace(".jpg", "_flow.npy"))
            np.save(npy_path, flow_final.astype(np.float32))

            # Flow-color image for visualization
            flow_smooth = cv2.GaussianBlur(flow_final, (0, 0), sigmaX=1.0)
            flow_color = flow_viz.flow_to_image(flow_smooth)

            # Optional sharpening (purely for visualization)
            flow_color = sharpen_flow_image(flow_color)

            png_path = os.path.join(out_rgb, f1_name.replace(".jpg", "_flow.png"))
            cv2.imwrite(png_path, cv2.cvtColor(flow_color, cv2.COLOR_RGB2BGR))

            # Init video writer
            if video_writer is None:
                h, w, _ = flow_color.shape
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(video_path, fourcc, args.fps, (w, h))

            video_writer.write(cv2.cvtColor(flow_color, cv2.COLOR_RGB2BGR))

        if video_writer:
            video_writer.release()

        print(f"   ✔ Saved NPY to:   {out_npy}")
        print(f"   ✔ Saved PNG to:   {out_rgb}")
        print(f"   ✔ Saved Video to: {video_path}\n")

    print("\n============================================")
    print("   🎉 ALL MULTI-GAP RAFT SEQUENCES DONE     ")
    print("============================================\n")


if __name__ == "__main__":
    main()
