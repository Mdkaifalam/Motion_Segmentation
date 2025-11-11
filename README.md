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
- `outputs/` – generated masks and overlays  
- `RS_Papers/` – reference papers  
- `DIP_Project_Proposal.pdf` – project proposal  

### Tech Stack
Python, OpenCV, PyTorch, NumPy, HuggingFace, Segment Anything, RAFT
