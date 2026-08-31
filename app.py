"""
OilSight – SAR Oil Spill Detector
Model: CNN (ResNet-18) + Swin-Tiny hybrid
Checkpoint: cnn_swin_best.pth
Preprocessing matches training exactly:
  Grayscale(3-ch) → Resize(224) → ToTensor → Normalize(ImageNet stats)
"""

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from torchvision import transforms, models
import timm

# Path to external dataset CSV with geo-coordinates
CSV_PATH = os.path.join(os.path.dirname(__file__), "external", "JEETHU_BHAI.csv")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="OilSight – SAR Oil Spill Detector",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS  (dark glassy theme)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1528 40%, #0f1e35 70%, #071018 100%);
    min-height: 100vh;
}

section[data-testid="stSidebar"] {
    background: rgba(10, 20, 40, 0.85);
    border-right: 1px solid rgba(0, 170, 255, 0.15);
    backdrop-filter: blur(20px);
}

/* ── Hero ── */
.hero {
    background: linear-gradient(135deg, rgba(0,170,255,0.12) 0%, rgba(0,80,160,0.08) 100%);
    border: 1px solid rgba(0,170,255,0.2);
    border-radius: 20px;
    padding: 2.5rem 2.5rem 2rem;
    margin-bottom: 2rem;
    backdrop-filter: blur(10px);
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: "";
    position: absolute;
    top: -60px; right: -60px;
    width: 200px; height: 200px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(0,170,255,0.18) 0%, transparent 70%);
}
.hero h1 {
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(90deg, #00aaff, #00e5ff, #7b61ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.4rem;
}
.hero p { color: rgba(180,200,230,0.85); font-size: 1.05rem; margin: 0; }

/* ── Cards ── */
.glass-card {
    background: rgba(15, 30, 60, 0.6);
    border: 1px solid rgba(0,170,255,0.18);
    border-radius: 16px;
    padding: 1.8rem;
    backdrop-filter: blur(12px);
    margin-bottom: 1.2rem;
    transition: border-color 0.3s;
}
.glass-card:hover { border-color: rgba(0,170,255,0.4); }

/* ── Result cards ── */
.result-oil {
    background: linear-gradient(135deg, rgba(255,60,60,0.18), rgba(200,0,0,0.10));
    border: 2px solid rgba(255,80,80,0.55);
    border-radius: 18px;
    padding: 2rem;
    text-align: center;
    animation: pulse-red 2s infinite;
}
.result-clean {
    background: linear-gradient(135deg, rgba(0,220,100,0.16), rgba(0,150,60,0.10));
    border: 2px solid rgba(0,220,100,0.5);
    border-radius: 18px;
    padding: 2rem;
    text-align: center;
    animation: pulse-green 2s infinite;
}
@keyframes pulse-red  { 0%,100%{box-shadow:0 0 0 0 rgba(255,80,80,0.3)}  50%{box-shadow:0 0 20px 6px rgba(255,80,80,0.15)} }
@keyframes pulse-green{ 0%,100%{box-shadow:0 0 0 0 rgba(0,220,100,0.3)} 50%{box-shadow:0 0 20px 6px rgba(0,220,100,0.15)} }

.result-label { font-size: 1.8rem; font-weight: 800; letter-spacing: 1px; margin: 0.5rem 0; }
.result-emoji { font-size: 3.5rem; }
.conf-text    { font-size: 1rem; color: rgba(200,220,255,0.75); margin-top: 0.4rem; }

/* ── Confidence bars ── */
.conf-bar-wrap { margin: 1.2rem 0 0.4rem; }
.conf-bar-label { font-size: 0.85rem; color: rgba(180,200,230,0.7); margin-bottom: 4px; display:flex; justify-content:space-between; }
.conf-bar-bg { background: rgba(255,255,255,0.08); border-radius: 99px; height: 10px; overflow: hidden; }
.conf-bar-fill-oil   { height:100%; border-radius:99px; background:linear-gradient(90deg,#ff4444,#ff9944); }
.conf-bar-fill-clean { height:100%; border-radius:99px; background:linear-gradient(90deg,#00dc64,#00c8aa); }

/* ── Info pills ── */
.info-pill { display:inline-block; padding:4px 12px; border-radius:99px; font-size:0.78rem; font-weight:600; letter-spacing:0.5px; margin:2px 4px; }
.pill-blue  { background:rgba(0,170,255,0.18);  color:#00aaff; border:1px solid rgba(0,170,255,0.35); }
.pill-purple{ background:rgba(123,97,255,0.18); color:#a78bff; border:1px solid rgba(123,97,255,0.35); }
.pill-green { background:rgba(0,220,100,0.15);  color:#00dc64; border:1px solid rgba(0,220,100,0.3); }

/* ── Upload zone ── */
[data-testid="stFileUploader"] {
    border: 2px dashed rgba(0,170,255,0.35) !important;
    border-radius: 14px !important;
    background: rgba(0,170,255,0.04) !important;
    padding: 1rem !important;
    transition: border-color 0.3s !important;
}
[data-testid="stFileUploader"]:hover { border-color: rgba(0,170,255,0.65) !important; }

/* ── Metric boxes ── */
.metric-box { background:rgba(15,30,60,0.7); border:1px solid rgba(0,170,255,0.2); border-radius:12px; padding:1rem 1.2rem; text-align:center; }
.metric-val { font-size:1.5rem; font-weight:700; color:#00aaff; }
.metric-lbl { font-size:0.75rem; color:rgba(180,200,230,0.6); text-transform:uppercase; letter-spacing:1px; margin-top:2px; }

hr { border-color: rgba(0,170,255,0.12) !important; }
::-webkit-scrollbar { width:6px; }
::-webkit-scrollbar-track { background:transparent; }
::-webkit-scrollbar-thumb { background:rgba(0,170,255,0.3); border-radius:3px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL DEFINITION  — mirrors the training script exactly
# ─────────────────────────────────────────────────────────────────────────────
class CNNSwinHybrid(nn.Module):
    """
    CNN branch : ResNet-18 (fc replaced with Identity) -> 512-d
    Swin branch: timm swin_tiny_patch4_window7_224, num_classes=0 -> 768-d
    Fusion     : Linear(1280->512) -> BN -> ReLU -> Drop(0.3)
                 -> Linear(512->128) -> ReLU -> Drop(0.2) -> Linear(128->2)
    """
    def __init__(self):
        super().__init__()

        # CNN backbone
        cnn = models.resnet18(weights=None)
        cnn.fc = nn.Identity()
        self.cnn = cnn                           # output: (B, 512)

        # Swin-Tiny backbone (via timm)
        self.swin = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=False,
            num_classes=0,                       # output: (B, 768)
        )

        # Fusion MLP — index positions match saved state_dict keys
        self.fusion = nn.Sequential(
            nn.Linear(512 + 768, 512),   # index 0: fusion.0.*
            nn.BatchNorm1d(512),         # index 1: fusion.1.*
            nn.ReLU(),                   # index 2 (no params)
            nn.Dropout(0.3),             # index 3 (no params)
            nn.Linear(512, 128),         # index 4: fusion.4.*
            nn.ReLU(),                   # index 5 (no params)
            nn.Dropout(0.2),             # index 6 (no params)
            nn.Linear(128, 2),           # index 7: fusion.7.*
        )

    def forward(self, x):
        cnn_feat  = self.cnn(x)                          # (B, 512)
        swin_feat = self.swin(x)                         # (B, 768)
        combined  = torch.cat([cnn_feat, swin_feat], dim=1)  # (B, 1280)
        return self.fusion(combined)


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "cnn_swin_best.pth")
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ["No Oil Spill", "Oil Spill Detected"]
CLASS_EMOJI = ["✅", "🛢️"]
CLASS_CSS   = ["result-clean", "result-oil"]


# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING — matches val_test_transform in the training script exactly
#   Training opened images as: Image.open(path).convert("L")
#   Then transform: Grayscale(3) -> Resize(224) -> ToTensor -> Normalize
# ─────────────────────────────────────────────────────────────────────────────
TRANSFORM = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # SAR grayscale -> 3-channel
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def preprocess(pil_img: Image.Image) -> torch.Tensor:
    """Convert any PIL image to a model-ready (1, 3, 224, 224) batch."""
    # Match training: convert to L (grayscale) first, then Grayscale(3) repeats it
    if pil_img.mode != "L":
        pil_img = pil_img.convert("L")
    return TRANSFORM(pil_img).unsqueeze(0).to(DEVICE)


# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODEL  (cached — 150 MB file loaded once per session)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    mdl   = CNNSwinHybrid()
    state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    # Training used: torch.save(model.state_dict(), path)  — bare state_dict
    # Robustly handle checkpoint dicts too
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    mdl.load_state_dict(state, strict=True)
    mdl.to(DEVICE)
    mdl.eval()
    return mdl


@torch.no_grad()
def predict(mdl, tensor: torch.Tensor):
    logits = mdl(tensor)                  # (1, 2)
    probs  = F.softmax(logits, dim=1)[0]  # (2,)
    return probs.cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# GEO DATABASE  — load JEETHU_BHAI.csv, build filename → [object rows] lookup
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_geo_db() -> dict:
    """
    Reads JEETHU_BHAI.csv (skiprows=3 to skip the 3 header rows).
    Columns (0-indexed in the raw CSV):
      1  = IMAGE       jpg filename  (e.g. ow-0001.jpg)
      5  = start_time  datetime string
     18  = obj_ul_lon
     19  = obj_ul_lat
     20  = obj_ur_lon
     21  = obj_ur_lat
     22  = obj_br_lon
     23  = obj_br_lat
     24  = obj_bl_lon
     25  = obj_bl_lat
    Returns dict: {filename: [ {datetime, obj_ul_lon, obj_ul_lat, ...}, ... ]}
    Each image can have multiple detected object rows in the CSV.
    """
    if not os.path.exists(CSV_PATH):
        return {}
    try:
        df = pd.read_csv(
            CSV_PATH,
            skiprows=3,
            header=None,
            usecols=[1, 5, 18, 19, 20, 21, 22, 23, 24, 25],
            dtype=str,
        )
        df.columns = [
            "IMAGE", "datetime",
            "obj_ul_lon", "obj_ul_lat",
            "obj_ur_lon", "obj_ur_lat",
            "obj_br_lon", "obj_br_lat",
            "obj_bl_lon", "obj_bl_lat",
        ]
        coord_cols = [
            "obj_ul_lon", "obj_ul_lat",
            "obj_ur_lon", "obj_ur_lat",
            "obj_br_lon", "obj_br_lat",
            "obj_bl_lon", "obj_bl_lat",
        ]
        for col in coord_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["IMAGE"] + coord_cols)
        # Build lookup: one list of object dicts per filename
        geo_db: dict = {}
        for _, row in df.iterrows():
            fname = str(row["IMAGE"]).strip()
            entry = {
                "datetime":   str(row["datetime"]).strip(),
                "obj_ul_lon": float(row["obj_ul_lon"]),
                "obj_ul_lat": float(row["obj_ul_lat"]),
                "obj_ur_lon": float(row["obj_ur_lon"]),
                "obj_ur_lat": float(row["obj_ur_lat"]),
                "obj_br_lon": float(row["obj_br_lon"]),
                "obj_br_lat": float(row["obj_br_lat"]),
                "obj_bl_lon": float(row["obj_bl_lon"]),
                "obj_bl_lat": float(row["obj_bl_lat"]),
            }
            geo_db.setdefault(fname, []).append(entry)
        return geo_db
    except Exception:
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# HTML HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def conf_bar_html(label: str, value: float, is_oil: bool) -> str:
    fill = "conf-bar-fill-oil" if is_oil else "conf-bar-fill-clean"
    pct  = value * 100
    return (
        f'<div class="conf-bar-wrap">'
        f'<div class="conf-bar-label"><span>{label}</span><span>{pct:.1f}%</span></div>'
        f'<div class="conf-bar-bg"><div class="{fill}" style="width:{pct:.1f}%"></div></div>'
        f'</div>'
    )


def metric_box(val: str, lbl: str) -> str:
    return (
        f'<div class="metric-box">'
        f'<div class="metric-val">{val}</div>'
        f'<div class="metric-lbl">{lbl}</div>'
        f'</div>'
    )


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;margin-bottom:1.5rem;">
      <div style="font-size:3rem;">🛢️</div>
      <div style="font-size:1.2rem;font-weight:700;color:#00aaff;">OilSight</div>
      <div style="font-size:0.75rem;color:rgba(180,200,230,0.6);margin-top:2px;">
        SAR Oil Spill Detection
      </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Settings")

    threshold = st.slider(
        "Oil-spill confidence threshold",
        min_value=0.50, max_value=0.99, value=0.50, step=0.01,
        help="Raise this to only flag high-confidence detections.",
    )
    show_probs = st.checkbox("Show probability bars", value=True)
    show_gray  = st.checkbox("Show grayscale preview (model input)", value=False)

    st.markdown("---")
    st.markdown("### Model Info")
    st.markdown("""
    <span class="info-pill pill-blue">ResNet-18</span>
    <span class="info-pill pill-purple">Swin-Tiny</span>
    <span class="info-pill pill-green">2-Class</span>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top:1rem;font-size:0.8rem;color:rgba(180,200,230,0.55);">
    <b>Architecture</b><br>
    &nbsp;CNN: ResNet-18 → 512-d<br>
    &nbsp;Transformer: Swin-Tiny (timm) → 768-d<br>
    &nbsp;Fusion MLP: 1280 → 512 → 128 → 2<br><br>
    <b>Preprocessing</b><br>
    &nbsp;SAR → Grayscale → 3-ch repeat<br>
    &nbsp;Resize 224 × 224<br>
    &nbsp;ImageNet normalisation
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        f'<div style="font-size:0.75rem;color:rgba(180,200,230,0.4);text-align:center;">'
        f'Weights: <code>cnn_swin_best.pth</code><br>'
        f'Device: <code>{str(DEVICE).upper()}</code>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PAGE
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>Oil Spill Detector</h1>
  <p>Upload <strong>Sentinel-1 SAR</strong> imagery to instantly detect oil spill
     contamination using a <strong>ResNet-18 + Swin-Tiny hybrid model</strong>
     trained on the Sentinel-1 SAR Oil Spill dataset.</p>
</div>
""", unsafe_allow_html=True)

# ── Load model ─────────────────────────────────────────────────────────────
with st.spinner("Loading cnn_swin_best.pth ..."):
    try:
        model = load_model()
        st.success(
            f"Model loaded on **{str(DEVICE).upper()}** — ResNet-18 + Swin-Tiny ready.",
            icon="✅",
        )
    except Exception as exc:
        st.error(f"Failed to load model:\n\n```\n{exc}\n```")
        st.stop()

# ── Load geo database ────────────────────────────────────────────────────────
geo_db = load_geo_db()

st.markdown("---")

# ── File uploader ───────────────────────────────────────────────────────────
st.markdown("### Upload SAR Image(s)")
uploaded_files = st.file_uploader(
    "Drop SAR images here",
    type=["jpg", "jpeg", "png", "tif", "tiff", "bmp"],
    accept_multiple_files=True,
    label_visibility="collapsed",
)

if not uploaded_files:
    st.markdown("""
    <div class="glass-card" style="text-align:center;padding:3rem;">
      <div style="font-size:3rem;margin-bottom:1rem;">🛰️</div>
      <div style="font-size:1.1rem;font-weight:600;color:rgba(180,200,230,0.85);">
        No images uploaded yet
      </div>
      <div style="font-size:0.9rem;color:rgba(180,200,230,0.5);margin-top:0.5rem;">
        Upload one or more SAR images above to begin detection.
        Supports JPG, PNG, TIFF, BMP.
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"### Analysis Results — {len(uploaded_files)} image(s)")

results_all = []

for uploaded_file in uploaded_files:
    img    = Image.open(uploaded_file)
    tensor = preprocess(img)

    t0         = time.time()
    probs      = predict(model, tensor)
    elapsed_ms = (time.time() - t0) * 1000

    # Apply user-defined threshold
    pred_class = 1 if float(probs[1]) >= threshold else 0
    confidence = float(probs[pred_class])

    # Geo-lookup: find all object rows for this filename in the CSV
    geo_objects = geo_db.get(uploaded_file.name, None)  # list of obj dicts or None

    results_all.append({
        "name":        uploaded_file.name,
        "pred":        pred_class,
        "conf":        confidence,
        "probs":       probs,
        "time_ms":     elapsed_ms,
        "geo_objects": geo_objects,
    })

    title = (
        f"{'🛢️' if pred_class == 1 else '✅'}  {uploaded_file.name}  —  "
        f"**{CLASS_NAMES[pred_class]}** ({confidence*100:.1f}%)"
    )

    with st.expander(title, expanded=True):
        col_img, col_res = st.columns([1, 1], gap="large")

        with col_img:
            display = img.copy()
            if display.mode not in ("RGB", "L", "RGBA"):
                display = display.convert("RGB")
            st.image(display, caption=f"Original: {uploaded_file.name}",
                     width='stretch')
            if show_gray:
                gray = img.convert("L").convert("RGB")
                st.image(gray, caption="As seen by model (grayscale -> 3-ch)",
                         width='stretch')

        with col_res:
            color_hex = "#ff5555" if pred_class == 1 else "#00dc64"
            st.markdown(
                f'<div class="{CLASS_CSS[pred_class]}">'
                f'<div class="result-emoji">{CLASS_EMOJI[pred_class]}</div>'
                f'<div class="result-label" style="color:{color_hex};">'
                f'{CLASS_NAMES[pred_class].upper()}</div>'
                f'<div class="conf-text">Confidence: <b>{confidence*100:.2f}%</b></div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            if show_probs:
                st.markdown(
                    conf_bar_html("No Oil Spill",        float(probs[0]), False) +
                    conf_bar_html("Oil Spill Detected",  float(probs[1]), True),
                    unsafe_allow_html=True,
                )

            st.markdown("<br>", unsafe_allow_html=True)
            m1, m2, m3 = st.columns(3)
            orig_w, orig_h = img.size
            with m1:
                st.markdown(metric_box(f"{elapsed_ms:.0f} ms", "Inference Time"),
                            unsafe_allow_html=True)
            with m2:
                st.markdown(metric_box(f"{orig_w}x{orig_h}", "Original Size"),
                            unsafe_allow_html=True)
            with m3:
                st.markdown(metric_box(img.mode, "Image Mode"),
                            unsafe_allow_html=True)

            # Object bounding-box coordinates table (shown if CSV match found)
            if geo_objects:
                dt_str = geo_objects[0]["datetime"][:10]
                st.markdown(
                    f'<div style="margin-top:1.2rem;padding:0.5rem 1rem 0.3rem;'
                    f'background:rgba(0,170,255,0.06);border:1px solid rgba(0,170,255,0.22);'
                    f'border-radius:10px;">'
                    f'<span style="font-size:0.82rem;font-weight:600;color:#00aaff;">'
                    f'📍 Detected Object Coordinates&nbsp;</span>'
                    f'<span style="font-size:0.75rem;color:rgba(180,210,255,0.55);">'
                    f'({len(geo_objects)} object{"s" if len(geo_objects)>1 else ""}'
                    f' · {dt_str})</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                obj_rows = []
                for i, obj in enumerate(geo_objects, 1):
                    obj_rows.append({
                        "#":          i,
                        "UL Lon":     round(obj["obj_ul_lon"], 6),
                        "UL Lat":     round(obj["obj_ul_lat"], 6),
                        "UR Lon":     round(obj["obj_ur_lon"], 6),
                        "UR Lat":     round(obj["obj_ur_lat"], 6),
                        "BR Lon":     round(obj["obj_br_lon"], 6),
                        "BR Lat":     round(obj["obj_br_lat"], 6),
                        "BL Lon":     round(obj["obj_bl_lon"], 6),
                        "BL Lat":     round(obj["obj_bl_lat"], 6),
                    })
                obj_df = pd.DataFrame(obj_rows)
                st.dataframe(
                    obj_df,
                    hide_index=True,
                    use_container_width=True,
                )


# ─────────────────────────────────────────────────────────────────────────────
# BATCH SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
if len(results_all) > 1:
    st.markdown("---")
    st.markdown("### Batch Summary")

    n_oil   = sum(1 for r in results_all if r["pred"] == 1)
    n_clean = len(results_all) - n_oil
    avg_conf = np.mean([r["conf"] for r in results_all]) * 100

    c1, c2, c3, c4 = st.columns(4)
    for col, (val, lbl) in zip(
        [c1, c2, c3, c4],
        [(str(len(results_all)), "Total Images"),
         (str(n_oil),   "Oil Detected"),
         (str(n_clean), "Clean"),
         (f"{avg_conf:.1f}%", "Avg Confidence")],
    ):
        with col:
            st.markdown(metric_box(val, lbl), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # pandas already imported at module level
    rows = [{
        "File":              r["name"],
        "Prediction":        CLASS_NAMES[r["pred"]],
        "Confidence (%)":    f"{r['conf']*100:.2f}",
        "P(No Oil) (%)":     f"{r['probs'][0]*100:.2f}",
        "P(Oil Spill) (%)":  f"{r['probs'][1]*100:.2f}",
        "Inference (ms)":    f"{r['time_ms']:.1f}",
    } for r in results_all]

    df = pd.DataFrame(rows)
    st.dataframe(df, width='stretch', hide_index=True)

    st.download_button(
        "Download Results as CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="oil_spill_results.csv",
        mime="text/csv",
    )
