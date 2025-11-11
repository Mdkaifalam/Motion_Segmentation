# ===============================
# davis_loader.py
# ===============================

import os
import cv2

def load_davis_frames(sequence_dir):
    """
    Loads and sorts all frames in a DAVIS sequence folder.
    
    Args:
        sequence_dir (str): Path to the sequence (e.g. /DAVIS/JPEGImages/480p/bear)
    Returns:
        frames (List[np.ndarray]): list of RGB frames
        frame_names (List[str]): corresponding filenames
    """
    frame_names = sorted([
        f for f in os.listdir(sequence_dir)
        if f.endswith(('.jpg', '.png'))
    ])
    frames = [cv2.imread(os.path.join(sequence_dir, f)) for f in frame_names]
    frames = [f for f in frames if f is not None]
    return frames, frame_names


