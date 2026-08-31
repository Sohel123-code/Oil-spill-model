"""
app.py — Hugging Face Gradio Space Entrypoint for OilSight
Serves our custom FastAPI + HTML/CSS/JS frontend.
Uses spaces.GPU decorator to satisfy HF ZeroGPU requirement.
"""

import os
import sys

# ── Handle HF ZeroGPU / spaces package ───────────────────────────────────────
# HF Gradio SDK requires at least one @spaces.GPU decorated function when
# ZeroGPU is active. We satisfy this with a minimal wrapper around inference.
try:
    import spaces

    # Import inference so we can wrap it
    from model import CNNSwinHybrid
    import torch
    import torch.nn.functional as F

    @spaces.GPU(duration=60)
    def _gpu_infer(mdl, tensor):
        """GPU-accelerated inference wrapper for HF ZeroGPU."""
        with torch.no_grad():
            logits = mdl(tensor)
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()
        return probs

    # Monkey-patch model.py's infer to use ZeroGPU when available
    import model as _model_module
    import time

    _orig_infer = _model_module.infer

    def _patched_infer(mdl, pil_img, threshold=0.5):
        from model import preprocess, CLASS_NAMES
        w, h = pil_img.size
        tensor = preprocess(pil_img)
        t0 = time.time()
        probs = _gpu_infer(mdl, tensor)
        elapsed_ms = (time.time() - t0) * 1000
        pred_class = 1 if float(probs[1]) >= threshold else 0
        confidence = float(probs[pred_class])
        return {
            "pred_class":   pred_class,
            "prediction":   CLASS_NAMES[pred_class],
            "confidence":   round(confidence, 6),
            "prob_oil":     round(float(probs[1]), 6),
            "prob_clean":   round(float(probs[0]), 6),
            "inference_ms": round(elapsed_ms, 1),
            "image_size":   f"{w}x{h}",
        }

    _model_module.infer = _patched_infer
    print("[OilSight] ZeroGPU (spaces.GPU) mode activated.")

except ImportError:
    # Running locally or on CPU — spaces not available, use normal inference
    print("[OilSight] Running in CPU mode (spaces not available).")
except Exception as e:
    print(f"[OilSight] spaces.GPU setup warning: {e} — falling back to CPU.")

# ── Launch FastAPI on port 7860 ───────────────────────────────────────────────
import uvicorn
from main import app

uvicorn.run(app, host="0.0.0.0", port=7860)
