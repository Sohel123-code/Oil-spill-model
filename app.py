"""
app.py — Hugging Face Gradio Space Entrypoint for OilSight
Serves our custom FastAPI frontend (HTML/CSS/JS) inside a Free Gradio Space.

Hugging Face Gradio SDK runs this file and expects a Gradio app OR
a uvicorn-compatible ASGI app exported as `app`. We use gr.mount_gradio_app
to wrap our FastAPI app so HF can discover and serve it on port 7860.
"""

import gradio as gr
from main import app as fastapi_app

# Create a minimal Gradio Blocks interface (required by HF Gradio SDK)
with gr.Blocks(title="OilSight — SAR Oil Spill Detector") as demo:
    gr.HTML("""
    <div style="text-align:center; padding: 20px;">
        <h2 style="color:#0c1a3a;">🛢️ OilSight — SAR Oil Spill Detector</h2>
        <p style="color:#64748b;">Loading the custom interface... If it doesn't redirect automatically,
        <a href="/" style="color:#1e40af;">click here to open the full UI</a>.</p>
    </div>
    """)

# Mount our FastAPI app at root "/" so all routes (/predict/batch, /health, /static) work
app = gr.mount_gradio_app(fastapi_app, demo, path="/gradio")
