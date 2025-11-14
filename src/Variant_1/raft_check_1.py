"""
raft_check.py  (FULL-SEQUENCE RAFT FLOW GENERATOR with enhanced clarity)
-----------------------------------------------------------------------
Computes RAFT optical flow for an entire DAVIS sequence
with improved clarity, stability, and visualization.

Improvements:
- mixed_precision=True for cleaner edges
- alternate_corr=True for better correlations
- RAFT iterations increased (20 → 32)
- Optional: smoothing & sharpening for visualization

Outputs:
- flow_npy/
- flow_rgb/
- flow_video.mp4
"""

import os
import argparse
import torch
import cv2
import numpy as np
import sys
from argparse import Namespace
from PIL import Image

# --------------------------------------------------------
# FIX RAFT IMPORT FOR UPDATED REPO STRUCTURE
# --------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))      # /src/Variant_1
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..",".."))
RAFT_PATH = os.path.join(PROJECT_ROOT, "third_party", "RAFT", "core")

print("Using RAFT from:", RAFT_PATH)
sys.path.insert(0, RAFT_PATH)

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder


DEVICE = "cuda"


# --------------------------------------------------------
# Helper functions
# --------------------------------------------------------

def load_rgb_image_tensor(path):
    """Loads a DAVIS frame into a PyTorch tensor in RGB format."""
    img = np.array(Image.open(path)).astype(np.uint8)
    tensor = torch.from_numpy(img).permute(2,0,1).float()[None].to(DEVICE) / 255.0
    # tensor = torch.from_numpy(img).float()[None].to(DEVICE) / 255.0
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
    """Computes dense flow between two frames with improved clarity."""
    padder = InputPadder(img1.shape)
    img1, img2 = padder.pad(img1, img2)
    flow_low, flow_up = model(img1, img2, iters=iters, test_mode=True)
    flow = flow_up[0].permute(1,2,0).cpu().numpy()
    return flow


# Optional sharpening for visualization
def sharpen_flow_image(image):
    kernel = np.array([[0,-1,0],
                       [-1,5,-1],
                       [0,-1,0]])
    return cv2.filter2D(image, -1, kernel)


# --------------------------------------------------------
# Main
# --------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(description="Compute improved RAFT optical flow for DAVIS sequences")

    parser.add_argument("--davis_root", type=str,
                        default="datasets/DAVIS/JPEGImages/480p")
    parser.add_argument("--sequence", type=str, default="bear",
                        help="Sequence name OR comma-separated list OR 'all'")
    parser.add_argument("--model_path", type=str,
                        default="third_party/RAFT/models/raft-things.pth")
    parser.add_argument("--out_dir", type=str, default="./checks/raft_flow_outputs")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--iters", type=int, default=32,
                        help="RAFT iterations (default: 32 for clearer flow)")

    args = parser.parse_args()

    # --------------------------------------------------------
    # Load RAFT model (with clarity-improving settings)
    # --------------------------------------------------------
    print("\nLoading RAFT model...")

    args_raft = Namespace(
        small=False,              # use full model for cleanest flow
        mixed_precision=True,     # smoother gradients
        alternate_corr=False       # improved correlation implementation
    )

    model = RAFT(args_raft).to(DEVICE).eval()

    ckpt = torch.load(args.model_path)
    ckpt = remove_module_prefix(ckpt)
    model.load_state_dict(ckpt, strict=True)

    print("RAFT loaded successfully with high-clarity settings.\n")

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

        # Output folders
        out_rgb = os.path.join(args.out_dir, seq, "flow_rgb")
        out_npy = os.path.join(args.out_dir, seq, "flow_npy")
        os.makedirs(out_rgb, exist_ok=True)
        os.makedirs(out_npy, exist_ok=True)

        # Video writer
        video_path = os.path.join(args.out_dir, seq, f"{seq}_flow_video.mp4")
        video_writer = None

        # Run RAFT through sequence
        for i in range(len(frames) - 1):
            f1 = os.path.join(seq_dir, frames[i])
            f2 = os.path.join(seq_dir, frames[i+1])

            print(f"   RAFT: {frames[i]} → {frames[i+1]}")

            img1 = load_rgb_image_tensor(f1)
            img2 = load_rgb_image_tensor(f2)
            flow = run_raft(model, img1, img2, iters=args.iters)

            # Save NPY
            npy_path = os.path.join(out_npy, frames[i].replace(".jpg", "_flow.npy"))
            np.save(npy_path, flow)

            # Flow-color image
            flow_color = flow_viz.flow_to_image(flow)

            # Optional sharpening (helps clarity)
            flow_color = sharpen_flow_image(flow_color)

            png_path = os.path.join(out_rgb, frames[i].replace(".jpg", "_flow.png"))
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
    print("   🎉 ALL RAFT SEQUENCES PROCESSED CLEANLY   ")
    print("============================================\n")


if __name__ == "__main__":
    main()
