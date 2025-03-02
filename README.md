# CIFAR-10 ResNet Classification

This project is a deep learning project, where we implement a custom ResNet architecture to classify CIFAR-10 images.

## Project Structure
- `data/` → Contains CIFAR-10 dataset and test set.
- `notebooks/` → Contains Jupyter notebooks for initial exploration and experiments.
- `src/models/` → Contains the ResNet model implementation.
- `src/training/` → Training scripts and hyperparameter tuning.
- `src/inference/` → Code to generate Kaggle submissions.
- `src/utils/` → Helper functions for data preprocessing, metrics, etc.
- `reports/` → Final project report and presentations.

## Setup Instructions
- Install Python 3.6.x.
- Install PyTorch 1.0.0 and torchvision 0.2.1.
- Install other dependencies using `pip install -r requirements.txt`.
- Download CIFAR-10 dataset from [here](https://www.cs.toronto.edu/~kriz/cifar.html) and extract it to `data/` directory.
- Download ResNet weights from [here](https://download.pytorch.org/models/resnet101-5d3b4d8f.pth) and extract it to `src/models/` directory.

## Training
To train the model, run `python src/training/train.py`.

## Evaluation
To evaluate the model, run `python src/inference/evaluate.py`.

## More will be added soon.
