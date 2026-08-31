---
title: OilSight SAR Oil Spill Detector
emoji: 🛢️
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: mit
app_port: 7860
---

# OilSight — SAR Oil Spill Detector

**ResNet-18 + Swin-Tiny hybrid model** for detecting oil spills in Sentinel-1 SAR imagery.

## Features
- Drag-and-drop batch image upload
- Real-time oil spill classification with confidence scores
- Object bounding-box coordinates from the annotated Sentinel-1 SAR dataset
- Professional clean dashboard UI (HTML/CSS/JS + FastAPI)

## Model Architecture
- **CNN branch**: ResNet-18 → 512-d features
- **Transformer branch**: Swin-Tiny (timm) → 768-d features
- **Fusion MLP**: 1280 → 512 → 128 → 2 classes

## Dataset
Sentinel-1 SAR Oil Spill dataset with `JEETHU_BHAI.csv` containing annotated
object bounding-box coordinates (obj_ul, obj_ur, obj_br, obj_bl lon/lat).
