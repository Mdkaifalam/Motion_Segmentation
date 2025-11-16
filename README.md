# Motion Segmentation

This project implements motion segmentation pipelines including:
- Baseline model (camera stabilization + Farnebäck optical flow)
- RAFT-based Flow-SAM motion segmentation
- Evaluation on DAVIS and FBMS datasets

### Dataset :
```
wget https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip
```
then unzip it and store in the `datasets/`


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
```

### Tech Stack
Python, OpenCV, PyTorch, NumPy, HuggingFace, Segment Anything, RAFT


### Variant 2

- Step 1 : Download RAFT to get the optical flow of the images

```
mkdir third_party
cd third_party
git clone https://github.com/princeton-vl/RAFT.git
cd RAFT
bash download_models.sh
```
#### RAFT Directory Structure
third_party/
└── RAFT/
    ├── alt_cuda_corr/
    ├── core/
    ├── demo-frames/
    ├── models/
    │   ├── raft-things.pth
    │   └── raft-sintel.pth
    ├── __pycache__/
    ├── .gitignore
    ├── chairs_split.txt
    ├── demo.py
    ├── download_models.sh
    ├── evaluate.py
    ├── LICENSE
    ├── RAFT.png
    ├── README.md
    ├── train.py
    ├── train_mixed.sh
    ├── train_standard.sh
    └── utils/


- Step 2 : Install SAM ViT-h checkpoint

```
mkdir -p sam_checkpoint
cd sam_checkpoint

wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

- Step 3 : Run the following code to get the optical flow from RAFT.

```
python .\src\Variant_1\compute_flow_raft.py --davis_root datasets/DAVIS/JPEGImages/480p -- model_path third_party/RAFT/models/raft-things.pth 
--out_dir ./Variant_1_output/raft_flow --sequence all 
```
- Step 4 : Run the following code to get the mask of all the optical flow.
```
python .\src\Variant_1\flow_to_motion_mask_v3.py --flow_root ./Variant_1_output/raft_flow --out_root ./Variant_1_output/flowI_motion --threshold_model top_p --percentile 95 --top_p 0.05 --A_min 500 --smooth_kernel 7 --no_bilateral --sequence all
```
- Step 5 : Run the flowI-SAM model to get the video along with the mask.

```
python .\src\Variant_1\flowI_sam.py --sequence all --verbose --flowI_root ./Variant_1_output/flowI_motion --output_root ./Variant_1_output/flowI_sam_output
```
