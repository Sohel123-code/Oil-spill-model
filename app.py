"""
app.py — Hugging Face Space Entrypoint for OilSight
Launches FastAPI + custom HTML/CSS/JS frontend on port 7860.
Hugging Face Gradio SDK runs this file with `python app.py`.
"""

import uvicorn
from main import app  # FastAPI app with all routes

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
else:
    # When imported by HF runtime, also start via uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
