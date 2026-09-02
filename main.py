"""
main.py — FastAPI backend for OilSight SAR Oil Spill Detector.

Endpoints:
  GET  /            → serves index.html (frontend)
  GET  /health      → health check
  POST /predict     → single image inference
  POST /predict/batch → multiple images inference
"""

import base64
import io
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageDraw, ImageFont

from model import load_model, infer
from geo_db import load_geo_db

# ─────────────────────────────────────────────────────────────────────────────
# GLOBALS  (loaded once at startup)
# ─────────────────────────────────────────────────────────────────────────────
_model  = None
_geo_db: dict = {}

STATIC_DIR = Path(__file__).parent / "static"


# ─────────────────────────────────────────────────────────────────────────────
# LIFESPAN  (startup / shutdown)
# ─────────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _geo_db
    print("Loading model …")
    _model  = load_model()
    print("Model ready.")
    print("Loading geo database …")
    _geo_db = load_geo_db()
    print(f"Geo DB ready — {len(_geo_db)} unique images indexed.")
    yield


# ─────────────────────────────────────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="OilSight API",
    description="SAR Oil Spill Detection — ResNet-18 + Swin-Tiny hybrid",
    version="2.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE ANNOTATION HELPER
# ─────────────────────────────────────────────────────────────────────────────
def annotate_image(
    pil_img: Image.Image,
    prediction_text: str,
    pred_class: int,
    confidence: float,
    objects: list,
) -> str:
    """
    Draws bounding boxes using CSV pixel locations and stamps status banner.
    Returns base64 data URL string (data:image/jpeg;base64,...).
    """
    annotated = pil_img.convert("RGBA")
    w, h = annotated.size
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Try default fonts
    try:
        font_large = ImageFont.truetype("arial.ttf", max(14, int(w * 0.024)))
        font_small = ImageFont.truetype("arial.ttf", max(11, int(w * 0.018)))
    except Exception:
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()

    is_oil = (pred_class == 1)

    # 1. Draw object bounding boxes if available and oil detected
    if objects and is_oil:
        for i, obj in enumerate(objects, 1):
            pw = obj.get("patch_width", w) or w
            ph = obj.get("patch_height", h) or h
            sx = w / pw
            sy = h / ph

            xmin = int(obj["xmin"] * sx)
            ymin = int(obj["ymin"] * sy)
            xmax = int(obj["xmax"] * sx)
            ymax = int(obj["ymax"] * sy)

            # Ensure valid bounds
            xmin, xmax = max(0, min(xmin, xmax)), min(w, max(xmin, xmax))
            ymin, ymax = max(0, min(ymin, ymax)), min(h, max(ymin, ymax))

            # Semi-transparent box fill
            draw.rectangle([xmin, ymin, xmax, ymax], fill=(239, 68, 68, 55), outline=(239, 68, 68, 240), width=3)

            # Corner brackets accent
            corner_len = max(8, int(min(xmax - xmin, ymax - ymin) * 0.2))
            draw.line([(xmin, ymin), (xmin + corner_len, ymin)], fill=(255, 255, 255, 255), width=3)
            draw.line([(xmin, ymin), (xmin, ymin + corner_len)], fill=(255, 255, 255, 255), width=3)
            draw.line([(xmax, ymax), (xmax - corner_len, ymax)], fill=(255, 255, 255, 255), width=3)
            draw.line([(xmax, ymax), (xmax, ymax - corner_len)], fill=(255, 255, 255, 255), width=3)

            # Label pill
            label_txt = f"Spill #{i} ({obj['xmin']}, {obj['ymin']})"
            lbl_w = max(90, int(len(label_txt) * 7.5))
            lbl_h = 22
            lbl_y0 = max(0, ymin - lbl_h - 2)
            lbl_y1 = lbl_y0 + lbl_h
            lbl_x1 = min(w, xmin + lbl_w)

            draw.rectangle([xmin, lbl_y0, lbl_x1, lbl_y1], fill=(220, 38, 38, 230))
            draw.text((xmin + 6, lbl_y0 + 3), label_txt, fill=(255, 255, 255, 255), font=font_small)

    # 2. Stamp status header banner at top
    banner_h = 32
    if is_oil:
        banner_bg = (185, 28, 28, 225)   # Red
        status_str = f"OIL SPILL DETECTED  •  Conf: {confidence*100:.1f}%"
        if objects:
            status_str += f"  •  Objects: {len(objects)}"
    else:
        banner_bg = (22, 101, 52, 225)   # Green
        status_str = f"NO OIL SPILL DETECTED  •  Conf: {confidence*100:.1f}%"

    draw.rectangle([0, 0, w, banner_h], fill=banner_bg)
    draw.text((12, 6), f"OilSight SAR  |  {status_str}", fill=(255, 255, 255, 255), font=font_large)

    # Composite image and save to base64 jpeg
    final_img = Image.alpha_composite(annotated, overlay).convert("RGB")
    buf = io.BytesIO()
    final_img.save(buf, format="JPEG", quality=92)
    b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64_str}"


# ─────────────────────────────────────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────────────────────────────────────
def _build_result(filename: str, pil_img: Image.Image, threshold: float) -> dict:
    result = infer(_model, pil_img, threshold=threshold)
    result["filename"] = filename
    objects = _geo_db.get(filename, [])
    result["objects"] = objects

    # Generate annotated image with bounding boxes & HUD
    annotated_b64 = annotate_image(
        pil_img=pil_img,
        prediction_text=result["prediction"],
        pred_class=result["pred_class"],
        confidence=result["confidence"],
        objects=objects,
    )
    result["annotated_image"] = annotated_b64
    return result


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/predict")
async def predict_single(
    file:      UploadFile = File(...),
    threshold: float      = Form(default=0.5),
):
    """Single image inference. Returns one result object with annotated image."""
    raw = await file.read()
    try:
        img = Image.open(io.BytesIO(raw))
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid image file."})

    result = _build_result(file.filename or "upload.jpg", img, threshold)
    return JSONResponse(content=result)


@app.post("/predict/batch")
async def predict_batch(
    files:     list[UploadFile] = File(...),
    threshold: float            = Form(default=0.5),
):
    """Batch image inference. Returns a list of result objects with annotated images."""
    results = []
    for f in files:
        raw = await f.read()
        try:
            img = Image.open(io.BytesIO(raw))
        except Exception:
            results.append({
                "filename": f.filename,
                "error":    "Invalid image file — skipped.",
            })
            continue
        results.append(_build_result(f.filename or "upload.jpg", img, threshold))
    return JSONResponse(content=results)


# ─────────────────────────────────────────────────────────────────────────────
# STATIC FILES  (frontend — must be mounted LAST)
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return FileResponse(STATIC_DIR / "index.html")

app.mount("/assets", StaticFiles(directory=str(STATIC_DIR)), name="static")
