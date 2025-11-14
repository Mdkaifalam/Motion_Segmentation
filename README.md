# Motion Segmentation

This project implements motion segmentation pipelines including:
- Baseline model (camera stabilization + Farnebäck optical flow)
- RAFT-based Flow-SAM motion segmentation
- Evaluation on DAVIS and FBMS datasets

### Dataset :
```
https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip
```



### Structure
- `src/` – source code for baseline and Flow-SAM variants  
- `datasets/` – data (not uploaded)  
- `outputs/` – generated masks and overlays  (not uploaded)
 
### 🧩 Baseline Model – Motion Segmentation

#### **Overview**
The baseline model performs **motion segmentation** on the **DAVIS-2017 dataset** using classical computer vision methods.  
It follows these key steps:

- **Camera Stabilization:** Homography-based alignment between consecutive frames.  
- **Motion Estimation:** Dense **Farnebäck Optical Flow** to estimate pixel-wise motion.  
- **Heatmap Generation:** Convert optical flow magnitudes into normalized motion heatmaps.  
- **Adaptive Mask Thresholding:** Generate motion masks using **percentile**, **top-p**, or **Otsu** thresholds.  
- **Post-processing:** Morphological operations and optional **temporal smoothing** for noise removal.  
- **Visualization:** Produces both **Triptych (Original | Heatmap | Mask)** and **Segmented Overlay** videos.  
- **Evaluation:** Quantitative evaluation using *IoU, Dice, Precision, Recall, F1*, and *MAE* metrics against DAVIS ground truth.  

---

#### **Run Command (Single Sequence)**
```bash
python Code/baseline/run_flow_seg_and_eval.py \
  --input_root datasets/DAVIS/JPEGImages/480p \
  --gt_root datasets/DAVIS/Annotations/480p \
  --sequence bear \
  --output_root Code/outputs \
  --threshold_mode percentile \
  --percentile 90 \
  --smooth_kernel 7 \
  --A_min 1500 \
  --fps 15 \
  --verbose
### Tech Stack
Python, OpenCV, PyTorch, NumPy, HuggingFace, Segment Anything, RAFT
