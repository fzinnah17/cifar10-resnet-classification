# CIFAR-10 ResNet Classification

## Team Members
- [Team Member 1]
- [Team Member 2]
- [Team Member 3]

## Project Overview
This project implements an optimized ResNet architecture for image classification on the CIFAR-10 dataset. We designed a custom ResNet variant that achieves 81% accuracy on the test set while maintaining a parameter count under 5 million. The model features Squeeze-and-Excitation (SE) blocks, an asymmetric [3,5,3] block distribution, and a 40-80-160 channel progression.

## Setup Instructions

### Requirements
- Python 3.8+
- PyTorch 1.8+
- NumPy
- Pandas
- tqdm
- matplotlib (for visualization)

### Installation
```bash
# Clone the repository
git clone https://github.com/[username]/cifar10-resnet-classification.git
cd cifar10-resnet-classification

# Install required packages
pip install torch torchvision numpy pandas tqdm matplotlib
```

### Dataset
The CIFAR-10 dataset should be organized as follows:
```
cifar10-resnet-classification/
└── data/
    └── cifar-10-python/
        ├── cifar-10-batches-py/
        │   ├── batches.meta
        │   ├── data_batch_1
        │   ├── data_batch_2
        │   ├── data_batch_3
        │   ├── data_batch_4
        │   ├── data_batch_5
        │   └── test_batch
        └── cifar_test_nolabel.pkl
```

## Usage

### Training
Run the main script to train the model:
```bash
python script.py
```

## Running on NYU HPC

To run this project on NYU's High-Performance Computing (HPC) cluster:

### 1. Connect to HPC
```bash
# Install Cisco AnyConnect VPN for your OS
# Connect to NYU VPN, then SSH to Greene
ssh netid@greene.hpc.nyu.edu
# Enter your password
# If you get a warning about host identification, run:
# ssh-keygen -R greene.hpc.nyu.edu

# Connect to the compute node
ssh burst
```

### 2. Request GPU Resources
Choose one of the following options:
```bash
# For V100 GPU:
srun --account=ece_gy_7123-2025sp --partition=n1s8-v100-1 --gres=gpu:v100:1 --time=04:00:00 --pty /bin/bash

# For A100 GPU:
srun --account=ece_gy_7123-2025sp --partition=c12m85-a100-1 --gres=gpu --time=04:00:00 --pty /bin/bash
```

### 3. Setup Container Environment
```bash
# Start Singularity container
singularity exec --bind /scratch --nv --overlay /scratch/kh3134/overlay-25GB-500K.ext3:rw /scratch/kh3134/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif /bin/bash

# Inside the container
Singularity> source /ext3/env.sh
Singularity> conda activate base
(base) Singularity> cd /scratch/netid/path-to-project
```

### 4. Run the Training
```bash
(base) Singularity> python script.py
```

### First-time Setup (Only if needed)
If you need to set up the environment for the first time:
```bash
# Copy overlay and singularity image
scp greene-dtn:/scratch/work/public/overlay-fs-ext3/overlay-25GB-500K.ext3.gz .
gunzip -vvv ./overlay-25GB-500K.ext3.gz
scp -rp greene-dtn:/scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif .

# Install Miniconda in the overlay
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash ./Miniconda3-latest-Linux-x86_64.sh -b -p /ext3/miniconda3
```

## Results
- **Test Accuracy**: 81%
- **Model Parameters**: 4,788,170
- **Architecture**: ResNet variant with SE blocks and block distribution [3,5,3]

## Key Features
1. **Optimized Channel Distribution**: 40-80-160 progression for efficient parameter usage
2. **Attention Mechanisms**: SE blocks with reduction ratio 8 
3. **Advanced Training**: Adaptive augmentation, focal label smoothing, model averaging
4. **Robust Inference**: Model ensemble (Best, EMA, SWA) with test-time augmentation

## Project Structure
```
cifar10-resnet-classification/
├── data/                # Dataset directory
├── script.py            # Main training and evaluation script
├── best_model.pth       # Best validation model checkpoint
├── best_ema_model.pth   # EMA model checkpoint
├── swa_model.pth        # SWA model checkpoint
└── README.md            # Project documentation
```
