# ===============================
# baseline_motion.py
# ===============================

import argparse
import os
from frame_difference import frame_diff_davis
from background_subtraction import knn_background_davis, mog2_background_davis
from optical_flow_baseline import farneback_davis

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True, choices=["frame_diff", "mog2", "knn", "flow"])
    # parser.add_argument("--input_root", type=str, required=True)
    # parser.add_argument("--sequence", type=str, required=True)
    # parser.add_argument("--output_root", type=str, default="./outputs")
    input_root_default = "datasets/DAVIS/JPEGImages/480p"
    parser.add_argument("--input_root", type=str, default=input_root_default)

    sequence_default = "bear"
    parser.add_argument("--sequence", type=str, default=sequence_default)

    output_root_default = "./outputs"
    parser.add_argument("--output_root", type=str, default=output_root_default)
    args = parser.parse_args()

    seq_path = os.path.join(args.input_root, args.sequence)
    out_path = os.path.join(args.output_root, args.sequence + "_" + args.method)

    if args.method == "frame_diff":
        frame_diff_davis(seq_path, out_path, T = 5, kernel_size=7, A_min=100)
    
    elif args.method == 'knn' :
        knn_background_davis(seq_path, out_path, A_min=500, kernel_size= 7, dist2Threshold=3000)
    elif args.method == "mog2":
        mog2_background_davis(seq_path, out_path, method=args.method.upper(), A_min=500, kernel_size=7)
    elif args.method == "flow":
        farneback_davis(seq_path, out_path, percentile=90)
    else:
        raise ValueError("Unknown method")

    print("✅ Baseline finished successfully.")
