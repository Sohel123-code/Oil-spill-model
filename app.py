# -*- coding: utf-8 -*-
"""
app.py - OilSight SAR Oil Spill Detector
Hugging Face ZeroGPU-compatible Gradio application.
Model: cnn_swin_v2_best.pth (CNNSwinHybrid)

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

from model import CNNSwinHybrid, TRANSFORM, DEVICE, CLASS_NAMES, load_model
from geo_db import load_geo_db

print('===== OilSight Startup =====')
_model = load_model()
_model.eval()
_geo_db = load_geo_db()
print(f'Geo DB ready - {len(_geo_db)} images indexed.')


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
    status_str = (f'OIL SPILL DETECTED  |  Conf: {confidence*100:.1f}%  |  Objects: {len(objects)}'
                  if is_oil else f'NO OIL SPILL  |  Conf: {confidence*100:.1f}%')
    draw.rectangle([0, 0, w, banner_h], fill=banner_bg)
    draw.text((12, 8), f'OilSight SAR  |  {status_str}', fill=(255, 255, 255, 255), font=font_large)
    return Image.alpha_composite(annotated, overlay).convert('RGB')


def _resolve_geo_key(filepath):
    if not filepath:
        return '', []
    basename = os.path.basename(filepath).strip()
    if basename in _geo_db:
        return basename, _geo_db[basename]
    basename_lower = basename.lower()
    for key in _geo_db:
        if key.lower() == basename_lower:
            return key, _geo_db[key]
    for key in _geo_db:
        if key.lower() in basename_lower or basename_lower in key.lower():
            return key, _geo_db[key]
    return basename, []


@spaces.GPU(duration=60)
def predict_oil(image, filepath, threshold):
    if image is None:
        return None, '<p>Please upload a SAR image.</p>', '', ''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _model.to(device)
    img_gray = image.convert('L')
    tensor = TRANSFORM(img_gray).unsqueeze(0).to(device)
    t0 = time.time()
    with torch.no_grad():
        logits = _model(tensor)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()
    elapsed_ms = (time.time() - t0) * 1000
    pred_class = 1 if float(probs[1]) >= threshold else 0
    confidence = float(probs[pred_class])
    prob_oil = float(probs[1])
    prob_clean = float(probs[0])
    matched_key, objects = _resolve_geo_key(filepath or '')
    annotated_img = _annotate(image, pred_class, confidence, objects)
    if pred_class == 1:
        status_html = (
            '<div style="background:#b91c1c;color:white;padding:16px;border-radius:10px;'
            'font-size:1.2rem;font-weight:700;text-align:center;">'
            f'Oil Spill DETECTED &nbsp;|&nbsp; Confidence: {confidence*100:.1f}%'
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
        f'| **Prediction** | {CLASS_NAMES[pred_class]} |\n'
        f'| **Confidence** | **{confidence*100:.2f}%** |\n'
        f'| **Oil Probability** | {prob_oil*100:.2f}% |\n'
        f'| **Clean Probability** | {prob_clean*100:.2f}% |\n'
        f'| **Inference Time** | {elapsed_ms:.1f} ms |\n'
        f'| **Objects Detected** | {len(objects)} |\n'
        f'| **Resolution** | {image.width}x{image.height} |\n'
        f'| **Filename** | `{matched_key or "none"}` |\n'
    )
    if objects:
        rows = ''
        for i, obj in enumerate(objects, 1):
            ul_lon = obj.get('obj_ul_lon', '-')
            ul_lat = obj.get('obj_ul_lat', '-')
            br_lon = obj.get('obj_br_lon', '-')
            br_lat = obj.get('obj_br_lat', '-')
            lon_s = f'{ul_lon:.6f}' if isinstance(ul_lon, float) else str(ul_lon)
            lat_s = f'{ul_lat:.6f}' if isinstance(ul_lat, float) else str(ul_lat)
            blon_s = f'{br_lon:.6f}' if isinstance(br_lon, float) else str(br_lon)
            blat_s = f'{br_lat:.6f}' if isinstance(br_lat, float) else str(br_lat)
            rows += (
                '<tr>'
                f'<td style="padding:8px;border:1px solid #334155;text-align:center;font-weight:700;">{i}</td>'
                f'<td style="padding:8px;border:1px solid #334155;">{obj.get("xmin","")}, {obj.get("ymin","")}</td>'
                f'<td style="padding:8px;border:1px solid #334155;">{obj.get("xmax","")}, {obj.get("ymax","")}</td>'
                f'<td style="padding:8px;border:1px solid #334155;font-family:monospace;">{lon_s}, {lat_s}</td>'
                f'<td style="padding:8px;border:1px solid #334155;font-family:monospace;">{blon_s}, {blat_s}</td>'
                '</tr>'
            )
        coords_html = (
            '<div style="overflow-x:auto;margin-top:12px;">'
            '<table style="width:100%;border-collapse:collapse;font-size:0.88rem;color:#e2e8f0;background:#0f172a;">'
            '<thead><tr style="background:#1e3a5f;color:#93c5fd;">'
            '<th style="padding:10px;border:1px solid #334155;">#</th>'
            '<th style="padding:10px;border:1px solid #334155;">Pixel (xmin, ymin)</th>'
            '<th style="padding:10px;border:1px solid #334155;">Pixel (xmax, ymax)</th>'
            '<th style="padding:10px;border:1px solid #334155;">UL (lon, lat)</th>'
            '<th style="padding:10px;border:1px solid #334155;">BR (lon, lat)</th>'
            '</tr></thead>'
            f'<tbody>{rows}</tbody>'
            '</table></div>'
        )
    else:
        coords_html = "<p style='color:#94a3b8;font-style:italic;'>No geo-coordinate data found for this filename.</p>"
    return annotated_img, status_html, stats_md, coords_html


custom_css = '#title { text-align: center; }'

with gr.Blocks(
    title='OilSight - SAR Oil Spill Detector',
    theme=gr.themes.Ocean(),
    css=custom_css,
) as demo:
    gr.Markdown(
        '# OilSight - SAR Oil Spill Detector\n'
        '**ResNet-18 + Swin-Tiny Hybrid (v2)** - Sentinel-1 SAR Oil Spill Classification\n'
        '**Model: cnn_swin_v2_best.pth**',
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


if __name__ == '__main__':
    demo.launch()
