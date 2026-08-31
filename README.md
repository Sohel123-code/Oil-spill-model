---
title: OilSight SAR Oil Spill Detector
emoji: 🛢️
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.5.1
app_file: app.py
pinned: false
license: mit
---

# OilSight — SAR Oil Spill Detector

**ResNet-18 + Swin-Tiny hybrid deep learning model** for detecting oil spills in Sentinel-1 SAR imagery.

## Features
- Upload Sentinel-1 SAR images for binary oil spill classification
- Automatic bounding box annotation using ground-truth pixel locations
- Geo-coordinate lookup (UL/BR corners) from the annotated database
- Confidence scores and inference statistics
- ZeroGPU accelerated inference via `@spaces.GPU`
