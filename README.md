# BEV Environment Mapping for Autonomous Vehicles

## Overview
Developed as a solution for a machine learning contest on environment mapping for autonomous vehicles, this project implements a Lift-Splat-Shoot (LSS) based deep learning model to predict Bird's-Eye View (BEV) occupancy grids entirely from multi-camera setups. The model ingests images from six surround-view cameras and projects them into a unified 3D ego-centric space to generate an accurate top-down map of the vehicle's surroundings. 

This repository contains the evaluation pipeline, model architecture definition, and validation results on the nuScenes dataset.

## Model Architecture
The network is designed around the LSS paradigm, effectively translating 2D image features into a 3D dimensional space before flattening them into a BEV grid.

* **Image Backbone:** Utilizes `efficientnet_b0` (via `timm`) to extract dense feature maps from the 6 input cameras. The input images are resized to 224x480 resolution.
* **Depth Head (ASPP):** An Atrous Spatial Pyramid Pooling module is used to predict categorical depth distributions over 24 discrete depth bins (from 1.0m to 49.0m).
* **LSS View Transformer:** Projects the 2D image features into 3D frustums using the predicted depth probabilities and known camera intrinsics/extrinsics. These features are then voxel-pooled into a 2D BEV grid.
* **BEV Decoder:** A convolutional block that refines the aggregated BEV features and outputs the final binary occupancy probability map.
* **Grid Specifications:** The final output is a 200x200 BEV grid representing a 100m x 100m area (-50.0m to 50.0m on X and Y axes) at a resolution of 0.5 meters per cell.

## Evaluation Metrics & Results
The model was evaluated on the `v1.0-mini` validation split of the nuScenes dataset, consisting of 162 samples.

### Distance-Weighted Error
Standard Intersection over Union (IoU) treats all pixels equally. To better suit the autonomous driving context, we implemented a custom Distance-Weighted Error metric. This applies a $1/r$ penalty weight map to the loss, prioritizing the accuracy of occupancy predictions closer to the ego-vehicle where immediate navigation decisions are critical.

### Quantitative Results
A comprehensive threshold search was conducted to find the optimal binarization threshold for the predicted probability maps. 

At the optimal threshold of **0.60**, the model achieved the following performance:
* **Occupancy IoU:** 32.44%
* **F1 Score:** 48.99%
* **Precision:** 53.47%
* **Recall:** 45.21%
* **Distance-Weighted Error:** 0.1097

The relationship between the classification threshold and our core metrics is visualized in the generated plots (`metrics_vs_threshold.png`), demonstrating the tradeoff between precision and recall, peaking in F1 and IoU at the 0.60 mark.

### Qualitative Visualizations
The evaluation script automatically generates side-by-side comparisons of the model's output against the ground truth LiDAR-derived occupancy. Examples (`bev_sample_000.jpg` through `bev_sample_007.jpg`) demonstrate the model's ability to successfully infer spatial geometry and obstacle boundaries strictly from 2D camera feeds.

## Installation & Setup
To run the evaluation notebook, ensure you have the required dependencies installed. The environment requires PyTorch and the nuScenes devkit.

```bash
pip install numpy==1.26.4 scipy==1.11.4 torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2
pip install nuscenes-devkit pyquaternion timm matplotlib pillow
```

## Usage
1.  **Dataset Preparation:** Download the nuScenes dataset (`v1.0-mini` is supported by default) and update the `DATAROOT` variable in the notebook to point to your local dataset path.
2.  **Model Weights:** Ensure the pre-trained weights (`bev_model_final2.pth`) and configuration dictionary (`bev_cfg.pkl`) are placed in your model directory.
3.  **Run Evaluation:** Execute the cells in `evaluate_bev_colab.ipynb`. The script will build the PyTorch DataLoader, instantiate the EfficientNet-LSS architecture, and compute predictions over the validation split.
4.  **Output:** The script will output the optimal threshold metrics to the console, save a comprehensive JSON summary (`eval_results.json`), plot the metric curves, and generate image visualizations of the BEV predictions.
