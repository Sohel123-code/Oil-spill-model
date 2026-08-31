"""
app.py — OilSight SAR Oil Spill Detector
Hugging Face ZeroGPU-compatible Gradio application.

IMPORTANT: `import spaces` and `@spaces.GPU` MUST be at the top level
for HF ZeroGPU runtime to detect them during startup AST scan.
"""

# ── ZeroGPU: MUST be imported unconditionally at top level ────────────────────
import spaces

# ── Standard imports ──────────────────────────────────────────────────────────
import io
import base64
import time
import torch
import torch.nn.functional as F
import gradio as gr
from PIL import Image, ImageDraw, ImageFont

from model import CNNSwinHybrid, TRANSFORM, DEVICE, CLASS_NAMES, load_model
from geo_db import load_geo_db

# ── Load model & geo-db once at startup (outside GPU function) ────────────────
print("===== OilSight Startup =====")
_model = load_model()
_model.eval()
_geo_db = load_geo_db()
print(f"Geo DB ready — {len(_geo_db)} images indexed.")


# ── Annotation helper (CPU-only, no GPU needed) ───────────────────────────────
def _annotate(pil_img: Image.Image, pred_class: int, confidence: float, objects: list) -> Image.Image:
    """Draw bounding boxes and status banner onto image. Returns annotated PIL Image."""
    annotated = pil_img.convert("RGBA")
    w, h = annotated.size
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    try:
        font_large = ImageFont.truetype("arial.ttf", max(14, int(w * 0.024)))
        font_small = ImageFont.truetype("arial.ttf", max(11, int(w * 0.018)))
    except Exception:
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()

    is_oil = (pred_class == 1)

    # Draw bounding boxes for each detected object
    for i, obj in enumerate(objects, 1):
        pw = obj.get("patch_width", w) or w
        ph = obj.get("patch_height", h) or h
        sx, sy = w / pw, h / ph

        xmin = int(obj["xmin"] * sx); ymin = int(obj["ymin"] * sy)
        xmax = int(obj["xmax"] * sx); ymax = int(obj["ymax"] * sy)
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

    # Status banner
    banner_h = 34
    banner_bg = (185, 28, 28, 220) if is_oil else (22, 101, 52, 220)
    status_str = (f"OIL SPILL DETECTED  •  Conf: {confidence*100:.1f}%  •  Objects: {len(objects)}"
                  if is_oil else f"NO OIL SPILL  •  Conf: {confidence*100:.1f}%")
    draw.rectangle([0, 0, w, banner_h], fill=banner_bg)
    draw.text((12, 8), f"OilSight SAR  |  {status_str}", fill=(255, 255, 255, 255), font=font_large)

    return Image.alpha_composite(annotated, overlay).convert("RGB")


# ── ZeroGPU inference function: MUST be top-level with @spaces.GPU ───────────
@spaces.GPU(duration=60)
def predict_oil(image: Image.Image, filename: str, threshold: float):
    """
    GPU-accelerated inference. Called directly by Gradio event.
    @spaces.GPU decorator is at TOP LEVEL — required for HF ZeroGPU detection.
    """
    if image is None:
        return None, "⚠️ Please upload a SAR image.", "", ""

    # Move model to correct device (ZeroGPU provides GPU on demand)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _model.to(device)

    # Preprocess — must match training pipeline exactly
    img_gray = image.convert("L")
    tensor = TRANSFORM(img_gray).unsqueeze(0).to(device)

    # Inference
    t0 = time.time()
    with torch.no_grad():
        logits = _model(tensor)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()
    elapsed_ms = (time.time() - t0) * 1000

    pred_class = 1 if float(probs[1]) >= threshold else 0
    confidence = float(probs[pred_class])
    prob_oil   = float(probs[1])
    prob_clean = float(probs[0])

    # Geo-coordinate lookup by filename
    fname_key = filename.strip() if filename else ""
    objects = _geo_db.get(fname_key, [])

    # Annotate image with bounding boxes
    annotated_img = _annotate(image, pred_class, confidence, objects)

    # ── Status text ──────────────────────────────────────────────────────────
    if pred_class == 1:
        status_html = f"""
        <div style="background:#b91c1c;color:white;padding:16px;border-radius:10px;font-size:1.2rem;font-weight:700;text-align:center;">
          🛢️ OIL SPILL DETECTED &nbsp;|&nbsp; Confidence: {confidence*100:.1f}%
        </div>"""
    else:
        status_html = f"""
        <div style="background:#166534;color:white;padding:16px;border-radius:10px;font-size:1.2rem;font-weight:700;text-align:center;">
          ✅ NO OIL SPILL &nbsp;|&nbsp; Confidence: {confidence*100:.1f}%
        </div>"""

    # ── Stats markdown ───────────────────────────────────────────────────────
    stats_md = f"""
| Metric | Value |
|--------|-------|
| **Prediction** | {CLASS_NAMES[pred_class]} |
| **Confidence** | **{confidence*100:.2f}%** |
| **Oil Probability** | {prob_oil*100:.2f}% |
| **Clean Probability** | {prob_clean*100:.2f}% |
| **Inference Time** | {elapsed_ms:.1f} ms |
| **Objects Detected** | {len(objects)} |
| **Resolution** | {image.width}×{image.height} |
"""

    # ── Geo-coordinates table ────────────────────────────────────────────────
    if objects:
        rows = ""
        for i, obj in enumerate(objects, 1):
            ul_lon = obj.get("obj_ul_lon", "—")
            ul_lat = obj.get("obj_ul_lat", "—")
            br_lon = obj.get("obj_br_lon", "—")
            br_lat = obj.get("obj_br_lat", "—")
            lon_str = f"{ul_lon:.6f}" if isinstance(ul_lon, float) else str(ul_lon)
            lat_str = f"{ul_lat:.6f}" if isinstance(ul_lat, float) else str(ul_lat)
            br_lon_str = f"{br_lon:.6f}" if isinstance(br_lon, float) else str(br_lon)
            br_lat_str = f"{br_lat:.6f}" if isinstance(br_lat, float) else str(br_lat)
            rows += f"""
            <tr>
              <td style="padding:8px;border:1px solid #334155;text-align:center;font-weight:700;">{i}</td>
              <td style="padding:8px;border:1px solid #334155;">{obj.get('xmin','')}, {obj.get('ymin','')}</td>
              <td style="padding:8px;border:1px solid #334155;">{obj.get('xmax','')}, {obj.get('ymax','')}</td>
              <td style="padding:8px;border:1px solid #334155;font-family:monospace;">{lon_str}, {lat_str}</td>
              <td style="padding:8px;border:1px solid #334155;font-family:monospace;">{br_lon_str}, {br_lat_str}</td>
            </tr>"""
        coords_html = f"""
        <div style="overflow-x:auto;margin-top:12px;">
          <table style="width:100%;border-collapse:collapse;font-size:0.88rem;color:#e2e8f0;background:#0f172a;">
            <thead>
              <tr style="background:#1e3a5f;color:#93c5fd;">
                <th style="padding:10px;border:1px solid #334155;">#</th>
                <th style="padding:10px;border:1px solid #334155;">Pixel (xmin, ymin)</th>
                <th style="padding:10px;border:1px solid #334155;">Pixel (xmax, ymax)</th>
                <th style="padding:10px;border:1px solid #334155;">UL (lon, lat)</th>
                <th style="padding:10px;border:1px solid #334155;">BR (lon, lat)</th>
              </tr>
            </thead>
            <tbody>{rows}</tbody>
          </table>
        </div>"""
    else:
        coords_html = "<p style='color:#94a3b8;font-style:italic;'>No geo-coordinate data found for this filename in the database.</p>"

    return annotated_img, status_html, stats_md, coords_html


# ── Gradio UI ─────────────────────────────────────────────────────────────────
custom_css = """
#title { text-align: center; }
.result-card { border-radius: 12px; }
"""

with gr.Blocks(
    title="OilSight — SAR Oil Spill Detector",
    theme=gr.themes.Ocean(),
    css=custom_css
) as demo:

    gr.Markdown("""
# 🛢️ OilSight — SAR Oil Spill Detector
**ResNet-18 + Swin-Tiny Hybrid** · Sentinel-1 SAR Oil Spill Classification
""", elem_id="title")

    with gr.Row():
        # ── Left column: inputs ──────────────────────────────────────────────
        with gr.Column(scale=1):
            image_input = gr.Image(
                type="pil",
                label="📡 Upload SAR Image",
                height=300,
            )
            filename_input = gr.Textbox(
                label="Image Filename (for geo-coordinate lookup)",
                placeholder="e.g. ow-0001.jpg",
                info="Enter the exact filename to look up bounding box geo-coordinates from the database."
            )
            threshold_slider = gr.Slider(
                minimum=0.1, maximum=0.9, value=0.5, step=0.05,
                label="🎯 Detection Threshold",
                info="Lower = more sensitive to oil spills"
            )
            analyze_btn = gr.Button("🔍 Analyze Image", variant="primary", size="lg")

        # ── Right column: outputs ────────────────────────────────────────────
        with gr.Column(scale=1):
            annotated_output = gr.Image(
                label="🖼️ Detected Image (with bounding boxes)",
                height=300,
            )
            status_output = gr.HTML(label="Detection Status")

    with gr.Row():
        with gr.Column(scale=1):
            stats_output = gr.Markdown(label="📊 Analysis Statistics")
        with gr.Column(scale=1):
            coords_output = gr.HTML(label="📍 Geo-Coordinate Table")

    # ── Event binding: Gradio button calls @spaces.GPU function directly ─────
    analyze_btn.click(
        fn=predict_oil,
        inputs=[image_input, filename_input, threshold_slider],
        outputs=[annotated_output, status_output, stats_output, coords_output],
    )

    gr.Markdown("""
---
*Model: CNNSwinHybrid (ResNet-18 + Swin-Tiny) · Dataset: Sentinel-1 SAR · Classes: No Oil (0) / Oil Spill (1)*
""")


# ── Launch ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch()
