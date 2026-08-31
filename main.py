"""
main.py — FastAPI backend for OilSight SAR Oil Spill Detector.
"""

import base64
import io
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
# LOAD MODEL & DATABASE ON IMPORT
# ─────────────────────────────────────────────────────────────────────────────
print("Loading model …")
_model = load_model()
print("Model ready.")
print("Loading geo database …")
_geo_db = load_geo_db()
print(f"Geo DB ready — {len(_geo_db)} unique images indexed.")

STATIC_DIR = Path(__file__).parent / "static"


# ─────────────────────────────────────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="OilSight API",
    description="SAR Oil Spill Detection — ResNet-18 + Swin-Tiny hybrid",
    version="2.2.0",
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
) -> tuple[Image.Image, str]:
    """
    Draws bounding boxes using CSV pixel locations and stamps status banner.
    Returns (annotated_pil_image, base64_data_url).
    """
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

    # 1. Draw object bounding boxes if available
    if objects:
        for i, obj in enumerate(objects, 1):
            pw = obj.get("patch_width", w) or w
            ph = obj.get("patch_height", h) or h
            sx = w / pw
            sy = h / ph

            xmin = int(obj["xmin"] * sx)
            ymin = int(obj["ymin"] * sy)
            xmax = int(obj["xmax"] * sx)
            ymax = int(obj["ymax"] * sy)

            xmin, xmax = max(0, min(xmin, xmax)), min(w, max(xmin, xmax))
            ymin, ymax = max(0, min(ymin, ymax)), min(h, max(ymin, ymax))

            draw.rectangle([xmin, ymin, xmax, ymax], fill=(239, 68, 68, 55), outline=(239, 68, 68, 240), width=3)

            corner_len = max(8, int(min(xmax - xmin, ymax - ymin) * 0.2))
            draw.line([(xmin, ymin), (xmin + corner_len, ymin)], fill=(255, 255, 255, 255), width=3)
            draw.line([(xmin, ymin), (xmin, ymin + corner_len)], fill=(255, 255, 255, 255), width=3)
            draw.line([(xmax, ymax), (xmax - corner_len, ymax)], fill=(255, 255, 255, 255), width=3)
            draw.line([(xmax, ymax), (xmax, ymax - corner_len)], fill=(255, 255, 255, 255), width=3)

            label_txt = f"Spill #{i} ({obj['xmin']}, {obj['ymin']})"
            lbl_w = max(90, int(len(label_txt) * 7.5))
            lbl_h = 22
            lbl_y0 = max(0, ymin - lbl_h - 2)
            lbl_y1 = lbl_y0 + lbl_h
            lbl_x1 = min(w, xmin + lbl_w)

            draw.rectangle([xmin, lbl_y0, lbl_x1, lbl_y1], fill=(220, 38, 38, 230))
            draw.text((xmin + 6, lbl_y0 + 3), label_txt, fill=(255, 255, 255, 255), font=font_small)

    banner_h = 32
    if is_oil:
        banner_bg = (185, 28, 28, 225)
        status_str = f"OIL SPILL DETECTED  •  Conf: {confidence*100:.1f}%"
        if objects:
            status_str += f"  •  Objects: {len(objects)}"
    else:
        banner_bg = (22, 101, 52, 225)
        status_str = f"NO OIL SPILL DETECTED  •  Conf: {confidence*100:.1f}%"

    draw.rectangle([0, 0, w, banner_h], fill=banner_bg)
    draw.text((12, 6), f"OilSight SAR  |  {status_str}", fill=(255, 255, 255, 255), font=font_large)

    final_img = Image.alpha_composite(annotated, overlay).convert("RGB")
    buf = io.BytesIO()
    final_img.save(buf, format="JPEG", quality=92)
    b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
    return final_img, f"data:image/jpeg;base64,{b64_str}"


# ─────────────────────────────────────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────────────────────────────────────
def _build_result(filename: str, pil_img: Image.Image, threshold: float) -> dict:
    result = infer(_model, pil_img, threshold=threshold)
    result["filename"] = filename
    objects = _geo_db.get(filename, [])
    result["objects"] = objects

    _, annotated_b64 = annotate_image(
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
# STATIC FILES
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/custom-ui")
@app.get("/app")
async def custom_ui():
    return FileResponse(STATIC_DIR / "index.html")

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
