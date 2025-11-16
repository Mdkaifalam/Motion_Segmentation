#!/bin/bash
echo "=== Creating conda env: flowi-sam ==="
conda env create -f environment.yml

echo "=== Activating environment ==="
source ~/.bashrc
conda activate flowi-sam

echo "=== Installing PyTorch (CUDA 11.8) ==="
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo "=== Cloning RAFT ==="
mkdir -p third_party
cd third_party
git clone https://github.com/princeton-vl/RAFT.git
cd RAFT
bash ./download_models.sh

echo "=== Cloning SAM ==="
pip install git+https://github.com/facebookresearch/segment-anything.git

echo "=== Installation complete! ==="
