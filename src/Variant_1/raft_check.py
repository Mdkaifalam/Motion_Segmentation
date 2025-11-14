"""
raft_check.py  (FULL-SEQUENCE RAFT FLOW GENERATOR)
---------------------------------------------------
Computes RAFT optical flow for an entire DAVIS sequence.

Outputs:
- flow_npy/      : raw flow HxWx2 .npy files
- flow_rgb/      : color flow visualization .png
- flow_video.mp4 : video visualization of optical flow

Defaults:
- DAVIS root  : datasets/DAVIS2017/JPEGImages/480p
- Sequence    : bear
- Model path  : third_party/RAFT/models/raft-things.pth

Usage:
python src/Variant_1/raft_check.py
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
    """Loads a DAVIS frame into a PyTorch tensor."""
    img = np.array(Image.open(path)).astype(np.uint8)
    tensor = torch.from_numpy(img).permute(2,0,1).float()[None].to(DEVICE) / 255.0
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
def run_raft(model, img1, img2):
    """Computes dense flow between two frames."""
    padder = InputPadder(img1.shape)
    img1, img2 = padder.pad(img1, img2)
    flow_low, flow_up = model(img1, img2, iters=20, test_mode=True)
    flow = flow_up[0].permute(1,2,0).cpu().numpy()
    return flow


# --------------------------------------------------------
# Main
# --------------------------------------------------------

# def main():

#     parser = argparse.ArgumentParser(description="Compute RAFT flow for full DAVIS sequence")

#     parser.add_argument("--davis_root", type=str,
#                         default="datasets/DAVIS/JPEGImages/480p")
#     parser.add_argument("--sequence", type=str, default="bear")
#     parser.add_argument("--model_path", type=str,
#                         default="third_party/RAFT/models/raft-things.pth")
#     parser.add_argument("--out_dir", type=str, default="./checks/raft_flow_outputs")
#     parser.add_argument("--fps", type=int, default=10)

#     args = parser.parse_args()

#     # seq_dir = args.davis_root + "/" + args.sequence
#     seq_dir = os.path.join(args.davis_root, args.sequence)
#     # print(os.listdir(seq_dir))
#     frames = sorted([f for f in os.listdir(seq_dir) if f.endswith(".jpg")])

#     print(f"\nFound {len(frames)} frames in sequence: {args.sequence}")

#     # Create output structure
#     out_rgb = os.path.join(args.out_dir, args.sequence, "flow_rgb")
#     out_npy = os.path.join(args.out_dir, args.sequence, "flow_npy")
#     os.makedirs(out_rgb, exist_ok=True)
#     os.makedirs(out_npy, exist_ok=True)

#     # --------------------------------------------------------
#     # Load RAFT
#     # --------------------------------------------------------
#     print("\nLoading RAFT model...")

#     args_raft = Namespace(small=False, mixed_precision=False, alternate_corr=False)
#     model = RAFT(args_raft).to(DEVICE).eval()

#     ckpt = torch.load(args.model_path)
#     ckpt = remove_module_prefix(ckpt)
#     model.load_state_dict(ckpt, strict=True)

#     # --------------------------------------------------------
#     # Prepare video writer (set after reading first flow map)
#     # --------------------------------------------------------
#     video_path = os.path.join(args.out_dir, f"{args.sequence}_flow_video.mp4")
#     video_writer = None

#     # --------------------------------------------------------
#     # Process entire sequence
#     # --------------------------------------------------------
#     for i in range(len(frames) - 1):
#         f1 = os.path.join(seq_dir, frames[i])
#         f2 = os.path.join(seq_dir, frames[i+1])

#         print(f"Computing RAFT: {frames[i]} -> {frames[i+1]}")

#         img1 = load_rgb_image_tensor(f1)
#         img2 = load_rgb_image_tensor(f2)

#         flow = run_raft(model, img1, img2)

#         # Save raw flow
#         npy_name = frames[i].replace(".jpg", "_flow.npy")
#         np.save(os.path.join(out_npy, npy_name), flow)

#         # Save colored flow
#         flow_color = flow_viz.flow_to_image(flow)
#         png_name = frames[i].replace(".jpg", "_flow.png")
#         cv2.imwrite(os.path.join(out_rgb, png_name),
#                     cv2.cvtColor(flow_color, cv2.COLOR_RGB2BGR))

#         # Initialize video writer if needed
#         if video_writer is None:
#             h, w, _ = flow_color.shape
#             fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#             video_writer = cv2.VideoWriter(video_path, fourcc, args.fps, (w, h))

#         # Add frame to video
#         video_writer.write(cv2.cvtColor(flow_color, cv2.COLOR_RGB2BGR))

#     if video_writer:
#         video_writer.release()

#     print("\n=====================================")
#     print("   🎉 RAFT FULL SEQUENCE COMPLETED   ")
#     print("=====================================")
#     print(f"Flow NPY files saved to : {out_npy}")
#     print(f"Flow PNG images saved to: {out_rgb}")
#     print(f"Flow video saved to     : {video_path}\n")

def main():

    parser = argparse.ArgumentParser(description="Compute RAFT flow for full DAVIS sequence(s)")

    parser.add_argument("--davis_root", type=str,
                        default="datasets/DAVIS/JPEGImages/480p")
    parser.add_argument("--sequence", type=str, default="bear",
                        help="Sequence name OR comma-separated list OR 'all'")
    parser.add_argument("--model_path", type=str,
                        default="third_party/RAFT/models/raft-things.pth")
    parser.add_argument("--out_dir", type=str, default="./checks/raft_flow_outputs")
    parser.add_argument("--fps", type=int, default=10)

    args = parser.parse_args()

    # --------------------------------------------------------
    # Load RAFT model (once!)
    # --------------------------------------------------------
    print("\nLoading RAFT model...")

    args_raft = Namespace(small=False, mixed_precision=False, alternate_corr=False)
    model = RAFT(args_raft).to(DEVICE).eval()

    ckpt = torch.load(args.model_path)
    ckpt = remove_module_prefix(ckpt)
    model.load_state_dict(ckpt, strict=True)

    print("RAFT loaded successfully.\n")

    # --------------------------------------------------------
    # Determine which sequences to run
    # --------------------------------------------------------
    if args.sequence.lower() == "all":
        sequences = sorted(os.listdir(args.davis_root))
        print(f"🔍 Running RAFT on ALL sequences: {len(sequences)} found.\n")

    elif "," in args.sequence:
        sequences = [s.strip() for s in args.sequence.split(",")]
        print(f"🔍 Running RAFT on MULTIPLE sequences: {sequences}\n")

    else:
        sequences = [args.sequence]
        print(f"🔍 Running RAFT on SINGLE sequence: {args.sequence}\n")

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

        # --------------------------------------------------------
        # Run RAFT on the entire sequence
        # --------------------------------------------------------
        for i in range(len(frames) - 1):
            f1 = os.path.join(seq_dir, frames[i])
            f2 = os.path.join(seq_dir, frames[i+1])

            print(f"   RAFT: {frames[i]} → {frames[i+1]}")

            img1 = load_rgb_image_tensor(f1)
            img2 = load_rgb_image_tensor(f2)
            flow = run_raft(model, img1, img2)

            # Save NPY
            npy_path = os.path.join(out_npy, frames[i].replace(".jpg", "_flow.npy"))
            np.save(npy_path, flow)

            # Save flow-color
            flow_color = flow_viz.flow_to_image(flow)
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

    print("\n=====================================")
    print("   🎉 ALL RAFT SEQUENCES COMPLETED   ")
    print("=====================================\n")


if __name__ == "__main__":
    main()
