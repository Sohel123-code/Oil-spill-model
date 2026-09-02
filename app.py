# -*- coding: utf-8 -*-
"""
app.py - OilSight SAR Oil Spill Detector
Hugging Face ZeroGPU-compatible Gradio + FastAPI application.
Model: cnn_swin_v2_best.pth (CNNSwinHybrid)
"""

# ZeroGPU: MUST be imported unconditionally at top level
import spaces

import io
import os
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
# Annotation helper (used by Gradio UI)
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
        if not is_oil:
            break
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
    if basename in _geo_db:
        return _geo_db[basename]
    lower = basename.lower()
    for key in _geo_db:
        if key.lower() == lower:
            return _geo_db[key]
    for key in _geo_db:
        if key.lower() in lower or lower in key.lower():
            return _geo_db[key]
    return []


# ---------------------------------------------------------------------------
# Single @spaces.GPU function - runs model forward pass on GPU
# ---------------------------------------------------------------------------
@spaces.GPU
def _run_model(tensor):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _model.to(device)
    tensor = tensor.to(device)
    with torch.no_grad():
        logits = _model(tensor)
        probs = F.softmax(logits, dim=1)[0].cpu().tolist()
    return probs


# ---------------------------------------------------------------------------
# Reusable inference + metadata lookup function
# ---------------------------------------------------------------------------
def predict_with_location(pil_img, image_id, threshold=0.5):
    img_gray = pil_img.convert('L')
    tensor = TRANSFORM(img_gray).unsqueeze(0)
    probs = _run_model(tensor)
    prob_oil = probs[1]
    prob_no_oil = probs[0]
    oil_detected = prob_oil >= threshold
    pred_label = 'oil' if oil_detected else 'no_oil'
    confidence = round(prob_oil if oil_detected else prob_no_oil, 6)
    raw_objects = _resolve_objects(image_id)

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


# -# ---------------------------------------------------------------------------
# Batch Gradio UI Function — processes multiple files
# ---------------------------------------------------------------------------
def predict_oil_batch(filepaths, threshold=0.5):
    """Process multiple uploaded images and return summary + per-image results."""
    if not filepaths:
        return '<p>Please upload SAR images.</p>', '', ''

    # Accept list of filepaths from gr.File(file_count="multiple")
    if not isinstance(filepaths, list):
        filepaths = [filepaths]

    all_results = []
    gallery_images = []

    for fpath in filepaths:
        try:
            image = Image.open(fpath)
        except Exception:
            continue

        image_id = os.path.basename(fpath)
        result = predict_with_location(image, image_id, threshold)
        pred_class = 1 if result['oil_detected'] else 0
        confidence = result['confidence']
        raw_objects = _resolve_objects(image_id)
        annotated_img = _annotate(image, pred_class, confidence, raw_objects)

        all_results.append({
            'image_id': image_id,
            'result': result,
            'pred_class': pred_class,
            'confidence': confidence,
            'raw_objects': raw_objects,
            'annotated_img': annotated_img,
            'width': image.width,
            'height': image.height,
        })
        gallery_images.append((annotated_img, image_id))

    if not all_results:
        return '<p>No valid images found.</p>', '', []

    # ── Summary Statistics ──
    n_total = len(all_results)
    n_oil = sum(1 for r in all_results if r['pred_class'] == 1)
    n_clean = n_total - n_oil
    avg_conf = sum(r['confidence'] for r in all_results) / n_total * 100

    summary_html = (
        '<div style="display:flex;gap:16px;flex-wrap:wrap;margin-bottom:8px;">'
        # Total
        '<div style="flex:1;min-width:120px;background:#eff6ff;border:2px solid #93c5fd;'
        'border-radius:12px;padding:16px;text-align:center;">'
        f'<div style="font-size:2rem;font-weight:900;color:#0c1a3a;font-family:monospace;">{n_total}</div>'
        '<div style="font-size:.75rem;font-weight:700;color:#475569;text-transform:uppercase;'
        'letter-spacing:.5px;">Total Analysed</div>'
        '</div>'
        # Oil
        '<div style="flex:1;min-width:120px;background:#fef2f2;border:2px solid #fca5a5;'
        'border-radius:12px;padding:16px;text-align:center;">'
        f'<div style="font-size:2rem;font-weight:900;color:#b91c1c;font-family:monospace;">{n_oil}</div>'
        '<div style="font-size:.75rem;font-weight:700;color:#475569;text-transform:uppercase;'
        'letter-spacing:.5px;">🛢️ Oil Spill Detected</div>'
        '</div>'
        # Clean
        '<div style="flex:1;min-width:120px;background:#f0fdf4;border:2px solid #86efac;'
        'border-radius:12px;padding:16px;text-align:center;">'
        f'<div style="font-size:2rem;font-weight:900;color:#15803d;font-family:monospace;">{n_clean}</div>'
        '<div style="font-size:.75rem;font-weight:700;color:#475569;text-transform:uppercase;'
        'letter-spacing:.5px;">✅ Clean / No Spill</div>'
        '</div>'
        # Avg Confidence
        '<div style="flex:1;min-width:120px;background:#f6f8fb;border:2px solid #dbe3ed;'
        'border-radius:12px;padding:16px;text-align:center;">'
        f'<div style="font-size:2rem;font-weight:900;color:#0c1a3a;font-family:monospace;">{avg_conf:.1f}%</div>'
        '<div style="font-size:.75rem;font-weight:700;color:#475569;text-transform:uppercase;'
        'letter-spacing:.5px;">Avg Confidence</div>'
        '</div>'
        '</div>'
    )

    # ── Per-image detail cards ──
    detail_cards = ''
    for i, r in enumerate(all_results, 1):
        res = r['result']
        is_oil = r['pred_class'] == 1
        conf = r['confidence']
        image_id = r['image_id']

        # Status badge
        if is_oil:
            badge = (
                f'<span style="background:#b91c1c;color:white;padding:6px 16px;border-radius:99px;'
                f'font-weight:800;font-size:.85rem;">🛢️ Oil Spill Detected — {conf*100:.1f}%</span>'
            )
        else:
            badge = (
                f'<span style="background:#166534;color:white;padding:6px 16px;border-radius:99px;'
                f'font-weight:800;font-size:.85rem;">✅ No Oil Spill — {conf*100:.1f}%</span>'
            )

        # Spill coordinates table
        coords = ''
        if res['spills']:
            rows = ''
            for sp in res['spills']:
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
            coords = (
                '<div style="overflow-x:auto;margin-top:8px;">'
                '<table style="width:100%;border-collapse:collapse;font-size:0.85rem;color:#e2e8f0;background:#0f172a;">'
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
            coords = '<p style="color:#94a3b8;font-style:italic;font-size:.85rem;">No geo-coordinate data for this filename.</p>'

        border_color = '#fca5a5' if is_oil else '#86efac'
        left_border = '#b91c1c' if is_oil else '#166534'
        detail_cards += (
            f'<div style="background:white;border:1.5px solid {border_color};border-left:5px solid {left_border};'
            f'border-radius:12px;padding:16px;margin-bottom:12px;">'
            f'<div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px;margin-bottom:10px;">'
            f'<span style="font-weight:900;font-size:1rem;color:#0c1a3a;font-family:monospace;">#{i} — {image_id}</span>'
            f'{badge}'
            '</div>'
            f'<div style="font-size:.82rem;color:#475569;margin-bottom:6px;">'
            f'Resolution: <strong>{r["width"]}x{r["height"]}</strong> · '
            f'Spill objects: <strong>{res["spill_count"]}</strong>'
            '</div>'
            f'{coords}'
            '</div>'
        )

    details_html = (
        '<div style="margin-top:12px;">'
        '<h3 style="font-size:1rem;font-weight:800;color:#0c1a3a;margin-bottom:10px;">'
        '📋 Per-Image Results</h3>'
        f'{detail_cards}'
        '</div>'
    )

    return summary_html, details_html, gallery_images


# ---------------------------------------------------------------------------
# Build Gradio Blocks UI — Multi-Image Batch Upload
# ---------------------------------------------------------------------------
with gr.Blocks(title='OilSight - SAR Oil Spill Detector', theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        '# 🛢️ OilSight - SAR Oil Spill Detector\n'
        '**ResNet-18 + Swin-Tiny Hybrid (v2)** — Sentinel-1 SAR Oil Spill Classification\n\n'
        'Upload **multiple** SAR images to batch-analyse them. The model classifies each image '
        'and retrieves spill bounding-box coordinates from the annotated dataset.',
        elem_id='title',
    )

    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(
                label='Upload SAR Images (select multiple files)',
                file_types=['image'],
                file_count='multiple',
                type='filepath',
            )
            threshold_slider = gr.Slider(
                minimum=0.1, maximum=0.9, value=0.5, step=0.05,
                label='Detection Threshold',
                info='Lower = more sensitive to oil spills',
            )
            analyze_btn = gr.Button('🔍 Analyse All Images', variant='primary', size='lg')

    # ── Summary Section ──
    gr.Markdown('---')
    summary_output = gr.HTML(label='Summary Statistics')

    # ── Gallery of annotated images ──
    gallery_output = gr.Gallery(
        label='Annotated Results (click to enlarge)',
        columns=4,
        height='auto',
        object_fit='contain',
    )

    # ── Per-image detail cards ──
    details_output = gr.HTML(label='Per-Image Details')

    # ── Wire up ──
    analyze_btn.click(
        fn=predict_oil_batch,
        inputs=[file_input, threshold_slider],
        outputs=[summary_output, details_output, gallery_output],
    )

    gr.Markdown(
        '---\n'
        '*Model: CNNSwinHybrid v2 (ResNet-18 + Swin-Tiny) — '
        'Dataset: Sentinel-1 SAR — Classes: No Oil (0) / Oil Spill (1)*'
    )

# ---------------------------------------------------------------------------
# Attach REST endpoints to demo.app
# ---------------------------------------------------------------------------
demo.app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

@demo.app.post('/api/predict')
@demo.app.post('/predict')
def api_predict(
    file: UploadFile = File(...),
    threshold: float = Form(default=0.5),
):
    raw = file.file.read()
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


@demo.app.get('/api/health')
@demo.app.get('/health')
def api_health():
    return JSONResponse(content={
        'status': 'ok',
        'model': 'cnn_swin_v2_best.pth',
        'geo_db_size': len(_geo_db),
    })

demo.launch()
