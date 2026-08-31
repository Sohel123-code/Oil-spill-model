"""
app.py — Hugging Face Gradio Space Entrypoint for OilSight
Serves our custom FastAPI frontend (HTML/CSS/JS) on the 100% Free Gradio Space Tier.
"""

import gradio as gr
import uvicorn
from main import app

# Mount a minimal Gradio block if needed, while keeping our custom HTML/CSS/JS frontend on /
demo = gr.Blocks(title="OilSight — SAR Oil Spill Detector")
with demo:
    gr.Markdown("## OilSight SAR Oil Spill Detection System")

app = gr.mount_gradio_app(app, demo, path="/gradio")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
