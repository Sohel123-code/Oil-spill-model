"""
app.py — OilSight SAR Oil Spill Detector (Hugging Face Spaces)

Strategy:
  - Gradio is used as the runtime host (required for HF Spaces / ZeroGPU).
  - The full custom HTML/CSS/JS frontend (static/) is served via FastAPI routes
    mounted on top of the Gradio app using app.mount and custom APIRoutes.
  - /predict/batch and /predict endpoints mirror main.py exactly.
  - @spaces.GPU wraps the model inference so ZeroGPU allocates a GPU on demand.

IMPORTANT: `import spaces` and `@spaces.GPU` MUST be at the top level
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
from pathlib import Path
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageDraw, ImageFont

from model import CNNSwinHybrid, TRANSFORM, DEVICE, CLASS_NAMES, load_model
from geo_db import load_geo_db

STATIC_DIR = Path(__file__).parent / "static"

print("===== OilSight Startup =====")
_model = load_model()
_model.eval()
_geo_db = load_geo_db()
print(f"Geo DB ready — {len(_geo_db)} images indexed.")


def _annotate_b64(pil_img, pred_class, confidence, objects):
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
    banner_h = 34
    banner_bg = (185, 28, 28, 220) if is_oil else (22, 101, 52, 220)
    status_str = (f"OIL SPILL DETECTED  •  Conf: {confidence*100:.1f}%  •  Objects: {len(objects)}"
                  if is_oil else f"NO OIL SPILL  •  Conf: {confidence*100:.1f}%")
    draw.rectangle([0, 0, w, banner_h], fill=banner_bg)
    draw.text((12, 8), f"OilSight SAR  |  {status_str}", fill=(255, 255, 255, 255), font=font_large)
    final_img = Image.alpha_composite(annotated, overlay).convert("RGB")
    buf = io.BytesIO()
    final_img.save(buf, format="JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


@spaces.GPU(duration=60)
def _run_inference(tensor):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _model.to(device)
    tensor = tensor.to(device)
    with torch.no_grad():
        logits = _model(tensor)
        probs = torch.nn.functional.softmax(logits, dim=1)[0].cpu().numpy()
    return probs


def _infer(pil_img, filename, threshold):
    w, h = pil_img.size
    img_gray = pil_img.convert("L")
    tensor = TRANSFORM(img_gray).unsqueeze(0)
    t0 = time.time()
    probs = _run_inference(tensor)
    elapsed_ms = (time.time() - t0) * 1000
    pred_class = 1 if float(probs[1]) >= threshold else 0
    confidence = float(probs[pred_class])
    objects = _geo_db.get(filename, [])
    if not objects:
        fn_lower = filename.lower()
        for key in _geo_db:
            if key.lower() == fn_lower:
                objects = _geo_db[key]
                break
    annotated_b64 = _annotate_b64(pil_img, pred_class, confidence, objects)
    return {
        "filename":        filename,
        "pred_class":      pred_class,
        "prediction":      CLASS_NAMES[pred_class],
        "confidence":      round(confidence, 6),
        "prob_oil":        round(float(probs[1]), 6),
        "prob_clean":      round(float(probs[0]), 6),
        "inference_ms":    round(elapsed_ms, 1),
        "image_size":      f"{w}x{h}",
        "objects":         objects,
        "annotated_image": annotated_b64,
    }


# Minimal Gradio UI required for HF Spaces to detect app
with gr.Blocks(title="OilSight — SAR Oil Spill Detector") as demo:
    gr.HTML("""
    <script>
      // Redirect to the custom frontend served at /
      if (window.location.pathname.startsWith('/--/')) {
        window.location.replace('/');
      }
    </script>
    <div style="text-align:center;padding:2rem;color:#93c5fd;font-family:sans-serif;">
      <h2>OilSight SAR Oil Spill Detector</h2>
      <p>Loading custom interface... <a href="/" style="color:#60a5fa;">Click here if not redirected</a></p>
    </div>
    """)

demo.queue()
app = demo.app  # FastAPI instance

# Serve static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static_files")


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/predict")
async def predict_single(
    file: UploadFile = File(...),
    threshold: float = Form(default=0.5),
):
    raw = await file.read()
    try:
        img = Image.open(io.BytesIO(raw))
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid image file."})
    return JSONResponse(content=_infer(img, file.filename or "upload.jpg", threshold))


@app.post("/predict/batch")
async def predict_batch(
    files: list[UploadFile] = File(...),
    threshold: float = Form(default=0.5),
):
    results = []
    for f in files:
        raw = await f.read()
        try:
            img = Image.open(io.BytesIO(raw))
        except Exception:
            results.append({"filename": f.filename, "error": "Invalid image file — skipped."})
            continue
        results.append(_infer(img, f.filename or "upload.jpg", threshold))
    return JSONResponse(content=results)


if __name__ == "__main__":
    demo.launch()
