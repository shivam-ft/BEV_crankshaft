# BEV Environment Mapping for Autonomous Vehicles

## Project Overview
This repository contains the complete solution developed for a machine learning hackathon focused on autonomous vehicle perception. The primary objective is to generate accurate Bird's-Eye View (BEV) occupancy grids using only 2D camera data. By leveraging a Lift-Splat-Shoot (LSS) deep learning architecture, the system ingests feeds from six surround-view cameras and projects them into a unified, ego-centric 3D space to predict environmental occupancy. 

This approach eliminates the reliance on expensive LiDAR sensors while maintaining robust spatial awareness for autonomous navigation.

## Repository Contents
* **Model Training:** The complete training pipeline, dataset preprocessing, and model optimization steps are contained within the primary project notebook.
* **Evaluation Pipeline:** The validation scripts, metric calculations, and threshold optimizations are handled in the evaluation notebook.
* **Visual Data:** The repository includes training/architecture visual documentation, performance metric graphs, and direct BEV prediction comparisons against ground truth data.
* **Results Summary:** Quantitative performance data across all tested thresholds is stored in the evaluation output file.

---

## Technical Architecture
The core model is designed to translate standard 2D image features into a top-down 3D representation. The architecture is broken down into four primary components:

* **Camera Feature Extraction:** The model processes inputs from six cameras (Front, Front Right, Front Left, Back, Back Right, Back Left). It uses an `efficientnet_b0` backbone via the `timm` library to extract dense features from images resized to a 224x480 resolution.
* **Depth Estimation (ASPP):** An Atrous Spatial Pyramid Pooling (ASPP) head is utilized to predict depth probabilities. The depth is categorized into 24 discrete bins ranging from 1.0m to 49.0m, with a step size of 2.0m.
* **LSS View Transformer:** This module projects the 2D features into 3D frustums by combining the predicted depth distributions with the known camera intrinsic and extrinsic matrices. The 3D points are then voxel-pooled to aggregate features into a flat 2D grid.
* **BEV Occupancy Decoder:** A custom convolutional block refines the aggregated spatial features into a final binary occupancy map. The output is a 200x200 grid representing a 100m x 100m physical area (-50.0m to 50.0m on the X and Y axes) with a spatial resolution of 0.5 meters per cell.

---

## Training Pipeline
The model's end-to-end training procedure is documented in the main project file. This includes the instantiation of the nuScenes dataset loaders, the EfficientNet-LSS architecture initialization, and the iterative optimization over the training splits. Supporting diagrams and architectural visuals generated during the development phase are also provided to illustrate the workflow.

---

## Evaluation Methodology
The model was validated on the `v1.0-mini` split of the nuScenes dataset, processing a total of 162 validation samples. 

### Custom Distance-Weighted Error
Standard intersection metrics treat all misclassifications equally. To adapt the evaluation for autonomous driving, we introduced a custom Distance-Weighted Error metric. This applies a $1/r$ penalty weight map to the loss function, heavily penalizing occupancy prediction errors that occur closer to the ego-vehicle, where accuracy is critical for collision avoidance.

---

## Quantitative Results
A thorough threshold sweep (from 0.30 to 0.80) was conducted to determine the optimal binarization point for the predicted probability maps. The trade-offs between Precision, Recall, and F1 score are fully visualized in the metric plots.

The optimal threshold was identified as **0.60**, yielding the following performance metrics:
* **Occupancy IoU:** 32.444%
* **F1 Score:** 48.993%
* **Precision:** 53.467%
* **Recall:** 45.21%
* **Distance-Weighted Error:** 0.109755

---

## Qualitative Results
To visually verify the model's spatial understanding, the evaluation pipeline generates side-by-side plots containing the Predicted Probability map, the Binarized Output (at the 0.60 threshold), and the Ground Truth LiDAR map. 

The sample outputs successfully demonstrate the model's capacity to infer road boundaries, vehicle positions, and general environment geometry strictly from camera features without depth sensor inputs.

---

## Installation and Setup

### Dependencies
The environment requires PyTorch, the nuScenes development kit, and various image processing libraries. Install the required packages using the following commands:

```bash
pip install numpy==1.26.4 scipy==1.11.4 torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2
pip install nuscenes-devkit pyquaternion timm matplotlib pillow
```

### Execution
1. Download the nuScenes dataset (the codebase defaults to `v1.0-mini`) and configure the `DATAROOT` path in the notebooks.
2. Execute the cells in the main training notebook to train the model from scratch or load checkpointed weights.
3. For validation, ensure the pre-trained weights (`bev_model_final2.pth`) and configuration file (`bev_cfg.pkl`) are accessible.
4. Run the evaluation notebook to generate the JSON results, metric plots, and qualitative BEV sample comparisons.

**Note**:Download all the files provided in the drive link to my drive folder of your drive.
