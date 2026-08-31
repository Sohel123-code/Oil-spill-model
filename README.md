---
title: OilSight SAR Oil Spill Detector
emoji: 🛢️
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# OilSight — SAR Oil Spill Detector

**ResNet-18 + Swin-Tiny hybrid deep learning model** for detecting oil spills in Sentinel-1 SAR imagery.

## Features
- Drag-and-drop batch SAR image upload
- Automatic bounding box drawing using annotated pixel locations
- Side-by-side detected card with full coordinates table (Pixel & Geo)
- High-resolution Lightbox viewer with Raw vs Annotated toggle
- Single-click detected image download
