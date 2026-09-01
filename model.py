"""
model.py — CNNSwinHybrid definition, loading, and inference.
Preprocessing matches training exactly:
  Grayscale(3-ch) → Resize(224) → ToTensor → Normalize(ImageNet stats)
"""

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import timm

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "cnn_swin_v2_best.pth")
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ["No Oil Spill", "Oil Spill Detected"]


# ─────────────────────────────────────────────────────────────────────────────
# MODEL ARCHITECTURE
# ─────────────────────────────────────────────────────────────────────────────
class CNNSwinHybrid(nn.Module):
    """
    CNN branch : ResNet-18 (fc replaced with Identity) → 512-d
    Swin branch: timm swin_tiny_patch4_window7_224, num_classes=0 → 768-d
    Fusion     : Linear(1280→512) → BN → ReLU → Drop(0.3)
                 → Linear(512→128) → ReLU → Drop(0.2) → Linear(128→2)
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

        # Fusion MLP
        self.fusion = nn.Sequential(
            nn.Linear(512 + 768, 512),   # index 0
            nn.BatchNorm1d(512),         # index 1
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),         # index 4
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2),           # index 7
        )

    def forward(self, x):
        cnn_feat  = self.cnn(x)
        swin_feat = self.swin(x)
        combined  = torch.cat([cnn_feat, swin_feat], dim=1)
        return self.fusion(combined)


# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
TRANSFORM = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def preprocess(pil_img: Image.Image) -> torch.Tensor:
    """Convert any PIL image to a model-ready (1, 3, 224, 224) batch."""
    if pil_img.mode != "L":
        pil_img = pil_img.convert("L")
    return TRANSFORM(pil_img).unsqueeze(0).to(DEVICE)


# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODEL (call once at startup)
# ─────────────────────────────────────────────────────────────────────────────
def load_model() -> CNNSwinHybrid:
    mdl   = CNNSwinHybrid()
    state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    mdl.load_state_dict(state, strict=True)
    mdl.to(DEVICE)
    mdl.eval()
    return mdl


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def infer(mdl: CNNSwinHybrid, pil_img: Image.Image, threshold: float = 0.5) -> dict:
    """
    Run inference on a single PIL image.
    Returns a dict with: pred_class, prediction, confidence,
                         prob_oil, prob_clean, inference_ms, image_size.
    """
    w, h = pil_img.size
    tensor = preprocess(pil_img)

    t0    = time.time()
    logits = mdl(tensor)
    probs  = F.softmax(logits, dim=1)[0].cpu().numpy()
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
