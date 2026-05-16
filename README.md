# RGB-Thermal Fusion for Drone-Based Person Detection and Geo-Localization

> Progressive Mid-Stage RGB-Thermal Fusion for Lightweight Drone-Based Search-and-Rescue Person Detection using YOLOv8n

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green)
![ONNX](https://img.shields.io/badge/ONNX-Runtime-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

## Overview

This project presents a lightweight RGB-Thermal fusion framework for drone-based human detection and geo-localization in Search-and-Rescue (SAR) scenarios. The system combines RGB and thermal modalities using multiple fusion strategies built on top of YOLOv8n, with a focus on real-time deployment on edge hardware.

The project systematically compares:

* Early-stage fusion
* Mid-stage fusion
* Late-stage fusion (Weighted Boxes Fusion)

Among all strategies, the proposed **Progressive Mid-Stage Fusion** achieved the best balance between precision and recall while maintaining real-time performance.

The final ONNX deployment pipeline runs at approximately **45 FPS on CPU** and includes:

* Real-time drone video inference
* GPS geo-localization
* Satellite map visualization
* ONNX Runtime deployment

Based on the experimental results, the proposed method achieved:

* **59.04% mAP@0.5** on the RGBTDronePerson benchmark
* **85.07% mAP@0.5** after custom domain fine-tuning



---

# Key Features

* Lightweight dual-backbone YOLOv8n architecture
* RGB-Thermal feature fusion
* Progressive freeze-unfreeze training curriculum
* Weighted Boxes Fusion (WBF)
* ONNX export and real-time inference
* GPS geo-localization from drone imagery
* Leaflet satellite map visualization
* Real-time SAR-oriented detection pipeline

---

# Architecture

## Mid-Stage RGB-T Fusion Architecture

The proposed architecture uses:

* Two YOLOv8n backbones
* RGB stream
* Thermal stream
* Multi-scale feature extraction (P3/P4/P5)
* Feature concatenation fusion
* FPN + PAN neck
* Single-class detection head

The fusion process combines intermediate RGB and thermal feature maps before passing them into the detection neck.



---

# Fusion Strategies

## 1. Early-Stage Fusion

* Single YOLOv8n model
* RGB and thermal samples trained together
* Shared weights across modalities
* Simplest baseline approach

## 2. Mid-Stage Fusion (Proposed)

* Dual YOLOv8n backbones
* Intermediate feature fusion
* Concatenation-based feature merging
* Best overall performance

## 3. Late-Stage Fusion

* Independent RGB and thermal detectors
* Prediction merging using Weighted Boxes Fusion (WBF)



---

# Progressive Training Strategy

The key contribution of this project is a three-stage progressive freeze-unfreeze curriculum:

### Stage 1

* Freeze pretrained backbones
* Train fusion layers only

### Stage 2

* Unfreeze upper backbone layers

### Stage 3

* Fully fine-tune all parameters

This training strategy improved:

* Mid-stage baseline from **54.26% → 59.04% mAP@0.5**

without changing the network architecture.



---

# Datasets

## Primary Benchmark

### RGBTDronePerson

* 6,125 aligned RGB-Thermal image pairs
* DJI Matrice 300 RTK
* 20–80m altitude
* Tiny object detection benchmark

## Backbone Pretraining

### LLVIP

* Low-light RGB-Thermal pedestrian dataset

### SeaDronesSee

* Maritime aerial SAR dataset

## Custom Fine-Tuning Dataset

* DJI Mavic 3 footage
* NTUT AIoT Lab dataset
* Pseudo-thermal generation using OpenCV HOT colormap



---

# Experimental Results

| Method                    | mAP@0.5   | mAP@0.5:0.95 |
| ------------------------- | --------- | ------------ |
| Early Stage               | 25.48     | 8.35         |
| Mid Baseline S1           | 54.26     | 17.63        |
| Progressive S2 (Best)     | **59.04** | **20.04**    |
| Fine-Tuned Custom Dataset | **85.07** | **41.32**    |



---

# State-of-the-Art Comparison

The proposed model outperformed previous methods on RGBTDronePerson while remaining lightweight and real-time deployable.

| Method | Params | CPU FPS | mAP@0.5   |
| ------ | ------ | ------- | --------- |
| QFDet* | ~60M   | ~3 FPS  | 46.72     |
| Ours   | ~7.5M  | ~45 FPS | **59.04** |



---

# Geo-Localization System

The system estimates GPS coordinates by:

1. Extracting bounding box bottom-center
2. Computing camera rays
3. Applying drone yaw/pitch/roll transformation
4. Intersecting with ground plane
5. Converting offsets into latitude/longitude

The final visualization is rendered on a Leaflet satellite map.



---

# Project Structure

```bash
.
├── Early-stage/
├── Mid-stage/
├── Late-stage/
├── LLVIP_Backbones/
├── SDS_Backbones/
├── Tune/
├── Eda/
├── Demo/
├── datasets/
├── weights/
├── runs/
└── README.md
```

---

Features:

* Live drone stream
* Real-time detection
* GPS projection
* Satellite map visualization
* ONNX Runtime inference

---

# Technologies Used

* Python
* PyTorch
* Ultralytics YOLOv8
* ONNX Runtime
* OpenCV
* FastAPI
* Leaflet.js
* Roboflow
* Optuna

---

# Future Improvements

* Real thermal camera deployment
* Attention-based fusion modules
* Jetson Orin Nano optimization
* Temporal tracking integration
* TensorRT deployment



---

# Citation

```bibtex
@article{rgbt_drone_sar_2026,
  title={Progressive Mid-Stage RGB-Thermal Fusion for Lightweight Drone-Based Person Detection},
  author={Le, Tu Quoc Huy and Vu, Hoang Minh Khanh and others},
  year={2026}
}
```

---

# Authors

* Tu Quoc Huy Le
* Hoang Minh Khanh Vu
* Quang Khai Mai
* Trong Hieu Nguyen

FPT University — Department of Artificial Intelligence

