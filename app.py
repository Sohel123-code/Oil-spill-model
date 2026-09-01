# -*- coding: utf-8 -*-
"""
app.py - OilSight SAR Oil Spill Detector
Hugging Face ZeroGPU-compatible Gradio application.
Model: cnn_swin_v2_best.pth (CNNSwinHybrid)

Exposes:
  - Gradio UI  (existing, unchanged)
  - POST /api/predict  (new, for React frontend)

IMPORTANT: import spaces and @spaces.GPU MUST be at top level
for HF ZeroGPU runtime to detect them during startup AST scan.
"""

# ZeroGPU: MUST be imported unconditionally at top level
import spaces

import io
import os
import base64
import time
import torch
import torch.nn.functional as F
import gradio as gr
from PIL import Image, ImageDraw, ImageFont
from fastapi import File, Form, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from model import CNNSwinHybrid, TRANSFORM, DEVICE, CLASS_NAMES, load_model
from geo_db import load_geo_db

print('===== OilSight Startup =====')
_model = load_model()
_model.eval()
_geo_db = load_geo_db()
print(f'Geo DB ready - {len(_geo_db)} images indexed.')


# ---------------------------------------------------------------------------
# Annotation helper (used by Gradio UI only)
# ---------------------------------------------------------------------------
def _annotate(pil_img, pred_class, confidence, objects):
    annotated = pil_img.convert('RGBA')
    w, h = annotated.size
    overlay = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    try:
        font_large = ImageFont.truetype('arial.ttf', max(14, int(w * 0.024)))
        font_small = ImageFont.truetype('arial.ttf', max(11, int(w * 0.018)))
    except Exception:
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()
    is_oil = (pred_class == 1)
    for i, obj in enumerate(objects, 1):
        pw = obj.get('patch_width', w) or w
        ph = obj.get('patch_height', h) or h
        sx, sy = w / pw, h / ph
        xmin = int(obj['xmin'] * sx); ymin = int(obj['ymin'] * sy)
        xmax = int(obj['xmax'] * sx); ymax = int(obj['ymax'] * sy)
        xmin, xmax = max(0, min(xmin, xmax)), min(w, max(xmin, xmax))
        ymin, ymax = max(0, min(ymin, ymax)), min(h, max(ymin, ymax))
        draw.rectangle([xmin, ymin, xmax, ymax], fill=(239, 68, 68, 55), outline=(239, 68, 68, 240), width=3)
        cl = max(8, int(min(xmax - xmin, ymax - ymin) * 0.2))
        for pts in [[(xmin, ymin), (xmin + cl, ymin)], [(xmin, ymin), (xmin, ymin + cl)],
                    [(xmax, ymax), (xmax - cl, ymax)], [(xmax, ymax), (xmax, ymax - cl)]]:
            draw.line(pts, fill=(255, 255, 255, 255), width=3)
        label = f"Spill #{i} ({obj['xmin']},{obj['ymin']})"
        lbl_w = max(90, int(len(label) * 7.5))
        ly0 = max(0, ymin - 24)
        draw.rectangle([xmin, ly0, min(w, xmin + lbl_w), ly0 + 22], fill=(220, 38, 38, 230))
        draw.text((xmin + 6, ly0 + 3), label, fill=(255, 255, 255, 255), font=font_small)
    banner_h = 34
    banner_bg = (185, 28, 28, 220) if is_oil else (22, 101, 52, 220)
    status_str = (f'OIL SPILL DETECTED | Conf: {confidence*100:.1f}% | Objects: {len(objects)}'
                  if is_oil else f'NO OIL SPILL | Conf: {confidence*100:.1f}%')
    draw.rectangle([0, 0, w, banner_h], fill=banner_bg)
    draw.text((12, 8), f'OilSight SAR | {status_str}', fill=(255, 255, 255, 255), font=font_large)
    return Image.alpha_composite(annotated, overlay).convert('RGB')


# ---------------------------------------------------------------------------
# Geo lookup: image filename -> list of all spill objects from metadata
# ---------------------------------------------------------------------------
def _resolve_objects(filename: str) -> list:
    if not filename:
        return []
    basename = os.path.basename(filename).strip()
    # Exact match
    if basename in _geo_db:
        return _geo_db[basename]
    # Case-insensitive match
    lower = basename.lower()
    for key in _geo_db:
        if key.lower() == lower:
            return _geo_db[key]
    # Partial match (e.g. prefixed uploads)
    for key in _geo_db:
        if key.lower() in lower or lower in key.lower():
            return _geo_db[key]
    return []


# ---------------------------------------------------------------------------
# Core GPU inference (ZeroGPU - top-level required)
# ---------------------------------------------------------------------------
@spaces.GPU(duration=60)
def _run_model(tensor: torch.Tensor) -> 'list[float]':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _model.to(device)
    tensor = tensor.to(device)
    with torch.no_grad():
        logits = _model(tensor)
        probs = F.softmax(logits, dim=1)[0].cpu().tolist()
    return probs  # [prob_no_oil, prob_oil]


# ---------------------------------------------------------------------------
# predict_with_location
# The single reusable function for ALL consumers (Gradio UI + React API).
#
# Returns a dict matching the React-ready JSON contract:
# {
#   image_id, prediction, oil_detected, confidence, spill_count, spills
# }
#
# spills is always a list - 0, 1, or N entries.
# Each spill entry uses the upper-left corner (obj_ul_lon / obj_ul_lat)
# as the canonical representative coordinate for map placement.
# All four corner coordinates are also included for full fidelity.
# ---------------------------------------------------------------------------
def predict_with_location(
    pil_img: Image.Image,
    image_id: str,
    threshold: float = 0.5,
) -> dict:
    """
    Run CNN+Swin inference on pil_img, look up ALL matching spill objects
    from the metadata CSV by image_id, and return a structured dict.

    Args:
        pil_img   : PIL Image (any mode).
        image_id  : Original filename, e.g. 'ow-0011.jpg'.
        threshold : Confidence threshold for oil classification (default 0.5).

    Returns dict with keys:
        image_id, prediction, oil_detected, confidence,
        spill_count, spills
    """
    # 1. Preprocess
    img_gray = pil_img.convert('L')
    tensor = TRANSFORM(img_gray).unsqueeze(0)

    # 2. GPU inference
    probs = _run_model(tensor)  # [prob_no_oil, prob_oil]
    prob_oil = probs[1]
    prob_no_oil = probs[0]
    oil_detected = prob_oil >= threshold
    pred_label = 'oil' if oil_detected else 'no_oil'
    confidence = round(prob_oil if oil_detected else prob_no_oil, 6)

    # 3. Metadata lookup - ALL spill objects for this image
    raw_objects = _resolve_objects(image_id)

    # 4. Build spills list
    #    - If oil is detected AND metadata has objects -> return all
    #    - If oil is detected but no metadata match -> empty spills
    #      (model saw oil but we have no coordinates - honest response)
    #    - If no oil detected -> always empty spills
    if oil_detected and raw_objects:
        spills = [
            {
                'spill_id':  idx,
                'latitude':  obj['obj_ul_lat'],
                'longitude': obj['obj_ul_lon'],
                'bbox': {
                    'xmin': obj['xmin'],
                    'ymin': obj['ymin'],
                    'xmax': obj['xmax'],
                    'ymax': obj['ymax'],
                },
                'corners': {
                    'ul': {'lat': obj['obj_ul_lat'], 'lon': obj['obj_ul_lon']},
                    'ur': {'lat': obj['obj_ur_lat'], 'lon': obj['obj_ur_lon']},
                    'br': {'lat': obj['obj_br_lat'], 'lon': obj['obj_br_lon']},
                    'bl': {'lat': obj['obj_bl_lat'], 'lon': obj['obj_bl_lon']},
                },
            }
            for idx, obj in enumerate(raw_objects, start=1)
        ]
    else:
        spills = []

    return {
        'image_id':    image_id,
        'prediction':  pred_label,
        'oil_detected': oil_detected,
        'confidence':  confidence,
        'spill_count': len(spills),
        'spills':      spills,
    }


# ---------------------------------------------------------------------------
# Gradio UI function (wraps predict_with_location + annotation)
# ---------------------------------------------------------------------------
@spaces.GPU(duration=60)
def predict_oil(image, filepath, threshold):
    if image is None:
        return None, '<p>Please upload a SAR image.</p>', '', ''
    image_id = os.path.basename(filepath or 'upload.jpg')
    result = predict_with_location(image, image_id, threshold)
    pred_class = 1 if result['oil_detected'] else 0
    confidence = result['confidence']
    raw_objects = _resolve_objects(image_id)
    annotated_img = _annotate(image, pred_class, confidence, raw_objects)
    if result['oil_detected']:
        status_html = (
            '<div style="background:#b91c1c;color:white;padding:16px;border-radius:10px;'
            'font-size:1.2rem;font-weight:700;text-align:center;">'
            f'Oil Spill DETECTED &nbsp;|&nbsp; Confidence: {confidence*100:.1f}% &nbsp;|&nbsp;'
            f' Spills: {result["spill_count"]}'
            '</div>'
        )
    else:
        status_html = (
            '<div style="background:#166534;color:white;padding:16px;border-radius:10px;'
            'font-size:1.2rem;font-weight:700;text-align:center;">'
            f'NO OIL SPILL &nbsp;|&nbsp; Confidence: {confidence*100:.1f}%'
            '</div>'
        )
    stats_md = (
        '| Metric | Value |\n'
        '|--------|-------|\n'
        f'| **Prediction** | {result["prediction"]} |\n'
        f'| **Confidence** | **{confidence*100:.2f}%** |\n'
        f'| **Oil Probability** | {result["spills"] and "see spills" or "-"} |\n'
        f'| **Spill Count** | {result["spill_count"]} |\n'
        f'| **Resolution** | {image.width}x{image.height} |\n'
        f'| **Image ID** | `{image_id}` |\n'
    )
    if result['spills']:
        rows = ''
        for sp in result['spills']:
            c = sp['corners']
            rows += (
                '<tr>'
                f'<td style="padding:8px;border:1px solid #334155;text-align:center;font-weight:700;">{sp["spill_id"]}</td>'
                f'<td style="padding:8px;border:1px solid #334155;">{sp["bbox"]["xmin"]}, {sp["bbox"]["ymin"]}</td>'
                f'<td style="padding:8px;border:1px solid #334155;">{sp["bbox"]["xmax"]}, {sp["bbox"]["ymax"]}</td>'
                f'<td style="padding:8px;border:1px solid #334155;font-family:monospace;">{c["ul"]["lon"]:.6f}, {c["ul"]["lat"]:.6f}</td>'
                f'<td style="padding:8px;border:1px solid #334155;font-family:monospace;">{c["br"]["lon"]:.6f}, {c["br"]["lat"]:.6f}</td>'
                '</tr>'
            )
        coords_html = (
            '<div style="overflow-x:auto;margin-top:12px;">'
            '<table style="width:100%;border-collapse:collapse;font-size:0.88rem;color:#e2e8f0;background:#0f172a;">'
            '<thead><tr style="background:#1e3a5f;color:#93c5fd;">'
            '<th style="padding:10px;border:1px solid #334155;">#</th>'
            '<th style="padding:10px;border:1px solid #334155;">Pixel (xmin,ymin)</th>'
            '<th style="padding:10px;border:1px solid #334155;">Pixel (xmax,ymax)</th>'
            '<th style="padding:10px;border:1px solid #334155;">UL (lon,lat)</th>'
            '<th style="padding:10px;border:1px solid #334155;">BR (lon,lat)</th>'
            '</tr></thead>'
            f'<tbody>{rows}</tbody>'
            '</table></div>'
        )
    else:
        coords_html = "<p style='color:#94a3b8;font-style:italic;'>No geo-coordinate data found for this filename.</p>"
    return annotated_img, status_html, stats_md, coords_html


# ---------------------------------------------------------------------------
# Gradio UI (unchanged layout)
# ---------------------------------------------------------------------------
custom_css = '#title { text-align: center; }'

with gr.Blocks(
    title='OilSight - SAR Oil Spill Detector',
    theme=gr.themes.Ocean(),
    css=custom_css,
) as demo:
    gr.Markdown(
        '# OilSight - SAR Oil Spill Detector\n'
        '**ResNet-18 + Swin-Tiny Hybrid (v2)** - Sentinel-1 SAR Oil Spill Classification\n'
        '**Model: cnn_swin_v2_best.pth**  |  '
        '**React API:** `POST /api/predict`',
        elem_id='title',
    )
    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(
                label='Upload SAR Image (preserves filename for geo lookup)',
                file_types=['image'],
                type='filepath',
            )
            image_input = gr.Image(type='pil', label='Image Preview', height=260)
            threshold_slider = gr.Slider(
                minimum=0.1, maximum=0.9, value=0.5, step=0.05,
                label='Detection Threshold',
                info='Lower = more sensitive to oil spills',
            )
            analyze_btn = gr.Button('Analyze Image', variant='primary', size='lg')
        with gr.Column(scale=1):
            annotated_output = gr.Image(label='Detected Image (with bounding boxes)', height=300)
            status_output = gr.HTML(label='Detection Status')
    with gr.Row():
        with gr.Column(scale=1):
            stats_output = gr.Markdown(label='Analysis Statistics')
        with gr.Column(scale=1):
            coords_output = gr.HTML(label='Geo-Coordinate Table')

    def _load_image(filepath):
        if filepath is None:
            return None
        return Image.open(filepath)

    file_input.change(fn=_load_image, inputs=file_input, outputs=image_input)
    analyze_btn.click(
        fn=predict_oil,
        inputs=[image_input, file_input, threshold_slider],
        outputs=[annotated_output, status_output, stats_output, coords_output],
    )
    gr.Markdown(
        '---\n'
        '*Model: CNNSwinHybrid v2 (ResNet-18 + Swin-Tiny) - '
        'Dataset: Sentinel-1 SAR - Classes: No Oil (0) / Oil Spill (1)*'
    )


# ---------------------------------------------------------------------------
# FastAPI: POST /api/predict  - React frontend endpoint
#
# multipart/form-data fields:
#   file      : image file (required)
#   threshold : float 0.0-1.0 (optional, default 0.5)
# ---------------------------------------------------------------------------
demo.queue()
app = demo.app

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['POST', 'GET', 'OPTIONS'],
    allow_headers=['*'],
)


@app.post('/api/predict')
async def api_predict(
    file: UploadFile = File(...),
    threshold: float = Form(default=0.5),
):
    """
    React-ready inference endpoint.

    Request : multipart/form-data
        file      - image file (jpg/png/tif)
        threshold - float (optional, default 0.5)

    Response: application/json
    {
        image_id    : str,
        prediction  : 'oil' | 'no_oil',
        oil_detected: bool,
        confidence  : float,
        spill_count : int,
        spills      : [
            {
                spill_id  : int,
                latitude  : float,
                longitude : float,
                bbox      : {xmin, ymin, xmax, ymax},
                corners   : {ul, ur, br, bl} each {lat, lon}
            },
            ...
        ]
    }
    """
    raw = await file.read()
    try:
        pil_img = Image.open(io.BytesIO(raw))
    except Exception:
        return JSONResponse(
            status_code=400,
            content={'error': 'Invalid image file. Must be JPG, PNG, or TIFF.'},
        )
    image_id = file.filename or 'upload.jpg'
    result = predict_with_location(pil_img, image_id, threshold)
    return JSONResponse(content=result)


@app.get('/api/health')
async def api_health():
    return JSONResponse(content={
        'status': 'ok',
        'model': 'cnn_swin_v2_best.pth',
        'geo_db_size': len(_geo_db),
    })


if __name__ == '__main__':
    demo.launch()
