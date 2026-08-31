"""
app.py — Hugging Face Gradio Space for OilSight
ResNet-18 + Swin-Tiny Hybrid Model for SAR Oil Spill Detection
"""

import os
import io
import pandas as pd
from PIL import Image
import gradio as gr

from model import infer, load_model
from geo_db import load_geo_db
from main import annotate_image

# Load model and coordinates database
print("Loading model for Hugging Face Space …")
model = load_model()
geo_db = load_geo_db()
print(f"Model ready. {len(geo_db)} images in coordinate database.")


def detect_oil_spill(image_input, threshold):
    """
    Inference function for Gradio interface.
    Accepts an uploaded image and confidence threshold.
    Returns:
      1. Annotated Image (with drawn bounding boxes & status banner)
      2. Result Summary Markdown
      3. Formatted Coordinates & Bounding Box DataFrame
    """
    if image_input is None:
        return None, "Please upload a Sentinel-1 SAR image.", None

    if isinstance(image_input, str):
        filename = os.path.basename(image_input)
        pil_img = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        filename = getattr(image_input, "filename", "uploaded_sar.jpg")
        filename = os.path.basename(filename) if filename else "uploaded_sar.jpg"
        pil_img = image_input.convert("RGB")
    else:
        filename = "uploaded_sar.jpg"
        pil_img = Image.fromarray(image_input).convert("RGB")

    # Run deep learning inference
    res = infer(model, pil_img, threshold=threshold)
    pred_class = res["pred_class"]
    confidence = res["confidence"]
    prob_oil = res["prob_oil"]
    prob_clean = res["prob_clean"]
    inf_ms = res["inference_ms"]
    w, h = pil_img.size

    # Lookup objects in CSV database
    objects = geo_db.get(filename, [])

    # Annotate image with bounding boxes & HUD banner
    annotated_img, _ = annotate_image(
        pil_img=pil_img,
        prediction_text=res["prediction"],
        pred_class=pred_class,
        confidence=confidence,
        objects=objects,
    )

    # Format result markdown
    is_oil = (pred_class == 1)
    status_emoji = "🛢️" if is_oil else "✅"
    status_text = "Oil Spill Detected" if is_oil else "No Oil Spill (Clean Sea)"
    badge_color = "#dc2626" if is_oil else "#16a34a"

    summary_md = f"""
### {status_emoji} <span style="color:{badge_color}; font-weight:800; font-size:1.3rem;">{status_text}</span>
* **Image File:** `{filename}` ({w}×{h} pixels)
* **Classification Confidence:** **{confidence*100:.2f}%** (Oil: **{prob_oil*100:.1f}%** | Clean: **{prob_clean*100:.1f}%**)
* **Inference Speed:** **{inf_ms:.1f} ms** (ResNet-18 + Swin-Tiny)
* **Detected Oil Objects:** **{len(objects)} object(s)** mapped from Sentinel-1 dataset
    """

    # Build coordinates table dataframe
    if objects:
        table_rows = []
        for i, obj in enumerate(objects, 1):
            dt = obj.get("datetime", "N/A")
            table_rows.extend([
                {"Object": f"Spill #{i} ({dt})", "Corner Position": "Upper-Left (UL)",  "Pixel (X, Y)": f"({obj['xmin']}, {obj['ymin']})", "Geo Longitude": f"{obj['obj_ul_lon']:.6f}", "Geo Latitude": f"{obj['obj_ul_lat']:.6f}"},
                {"Object": f"Spill #{i} ({dt})", "Corner Position": "Upper-Right (UR)", "Pixel (X, Y)": f"({obj['xmax']}, {obj['ymin']})", "Geo Longitude": f"{obj['obj_ur_lon']:.6f}", "Geo Latitude": f"{obj['obj_ur_lat']:.6f}"},
                {"Object": f"Spill #{i} ({dt})", "Corner Position": "Bottom-Right (BR)", "Pixel (X, Y)": f"({obj['xmax']}, {obj['ymax']})", "Geo Longitude": f"{obj['obj_br_lon']:.6f}", "Geo Latitude": f"{obj['obj_br_lat']:.6f}"},
                {"Object": f"Spill #{i} ({dt})", "Corner Position": "Bottom-Left (BL)",  "Pixel (X, Y)": f"({obj['xmin']}, {obj['ymax']})", "Geo Longitude": f"{obj['obj_bl_lon']:.6f}", "Geo Latitude": f"{obj['obj_bl_lat']:.6f}"},
            ])
        coords_df = pd.DataFrame(table_rows)
    else:
        coords_df = pd.DataFrame([{
            "Object": "No Anomalies",
            "Corner Position": "—",
            "Pixel (X, Y)": "—",
            "Geo Longitude": "—",
            "Geo Latitude": "—"
        }])

    return annotated_img, summary_md, coords_df


# ─────────────────────────────────────────────────────────────────────────────
# GRADIO INTERFACE
# ─────────────────────────────────────────────────────────────────────────────
custom_css = """
#header { background: #0c1a3a; color: white; padding: 1.5rem; border-radius: 12px; margin-bottom: 1.2rem; text-align: center; }
#header h1 { font-weight: 900; color: #ffffff !important; margin: 0 0 0.3rem 0; font-size: 2rem; }
#header p { color: #93c5fd; margin: 0; font-size: 0.95rem; }
"""

with gr.Blocks(title="OilSight — SAR Oil Spill Detector", css=custom_css) as demo:
    gr.HTML("""
    <div id="header">
      <h1>🛢️ OilSight — SAR Oil Spill Detector</h1>
      <p>Hybrid Deep Learning (ResNet-18 + Swin-Tiny) • Sentinel-1 SAR Imagery Analysis & Spill Coordinate Mapping</p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            image_in = gr.Image(type="filepath", label="📤 Upload Sentinel-1 SAR Image (JPG / PNG / TIFF)")
            slider_thresh = gr.Slider(minimum=0.50, maximum=0.99, value=0.50, step=0.01, label="Confidence Threshold", info="Minimum confidence to classify as Oil Spill")
            btn_detect = gr.Button("🔍 Analyse SAR Image", variant="primary", size="lg")

        with gr.Column(scale=1):
            image_out = gr.Image(type="pil", label="🎯 Detected SAR Image with Bounding Boxes (Downloadable)")
            res_summary = gr.Markdown("Upload an image and click **Analyse SAR Image** to view detection results.")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📍 Detected Object Coordinates & Bounding Box Table")
            coords_table = gr.DataFrame(
                headers=["Object", "Corner Position", "Pixel (X, Y)", "Geo Longitude", "Geo Latitude"],
                label="Object Bounding Box & Geo Coordinates (UL, UR, BR, BL)",
                interactive=False
            )

    btn_detect.click(
        fn=detect_oil_spill,
        inputs=[image_in, slider_thresh],
        outputs=[image_out, res_summary, coords_table]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
