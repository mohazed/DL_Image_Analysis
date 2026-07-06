"""Guess-the-calories demo: 5-fold CalorieNet ensemble over known dataset images.

Run: uvicorn main:app --reload --port 8000   (from this directory)
"""
import math
import os
from contextlib import asynccontextmanager

import io

import cv2
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel

# --- paths (adjust here if your layout differs) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
MODEL_DIR = os.path.join(PROJECT_DIR, "model", "prepared")
TRAIN_LABELS_CSV = os.path.join(PROJECT_DIR, "m2-food-calorie-estimation", "train_labels.csv")
TRAIN_IMAGES_DIR = os.path.join(PROJECT_DIR, "m2-food-calorie-estimation", "train", "images")
STATIC_DIR = os.path.join(BASE_DIR, "static")

N_FOLDS = 5
BACKBONE_NAME = "convnext_tiny"
SOURCE_EMB_DIM = 16
HEAD_HIDDEN = 256
HEAD_DROPOUT = 0.2
IMG_SIZE = 384
LOG1P_CLAMP_MAX = math.log1p(4000.0)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

SOURCE_IDX = {"A": 0, "B": 1}


class CalorieNet(nn.Module):
    """Must match calorie_pipeline.py's CalorieNet exactly."""

    def __init__(self, backbone_name, source_emb_dim, head_hidden, head_dropout, n_sources=2):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=False, num_classes=0, global_pool="avg")
        feat_dim = self.backbone.num_features
        self.source_emb = nn.Embedding(n_sources, source_emb_dim)
        self.head = nn.Sequential(
            nn.Linear(feat_dim + source_emb_dim, head_hidden),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, x, source_idx):
        feat = self.backbone(x)
        emb = self.source_emb(source_idx)
        z = self.head(torch.cat([feat, emb], dim=1)).squeeze(1)
        return z


def derive_source(filename: str) -> str:
    ext = filename.rsplit(".", 1)[-1].lower()
    if ext == "png":
        return "A"
    elif ext in ("jpg", "jpeg"):
        return "B"
    raise ValueError(f"Unrecognized extension in filename: {filename}")


def letterbox_resize(img: np.ndarray, size: int, pad_value: int = 0) -> np.ndarray:
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(img, (nw, nh), interpolation=interp)
    canvas = np.full((size, size, 3), pad_value, dtype=np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas


def preprocess_pil(img: Image.Image) -> torch.Tensor:
    """Letterbox + ImageNet-normalize a PIL image into a CHW float tensor.

    Matches training preprocessing exactly (see calorie_pipeline.py §11/§13):
    EXIF-transpose, RGB, aspect-preserving letterbox to 384, black pad, no crop.
    """
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    arr = np.array(img)
    arr = letterbox_resize(arr, IMG_SIZE)
    arr = arr.astype(np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = arr.transpose(2, 0, 1)  # HWC -> CHW
    return torch.from_numpy(arr).float()


def load_and_preprocess(path: str) -> torch.Tensor:
    with Image.open(path) as img:
        return preprocess_pil(img)


# --- app state, populated at startup ---
class AppState:
    models: list
    labels_df: pd.DataFrame
    id_to_row: dict
    clip_bounds: dict


state = AppState()


def load_models() -> list:
    models = []
    for fold in range(N_FOLDS):
        ckpt_path = os.path.join(MODEL_DIR, f"fold{fold}.pt")
        if not os.path.exists(ckpt_path):
            raise RuntimeError(f"Missing checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = CalorieNet(BACKBONE_NAME, SOURCE_EMB_DIM, HEAD_HIDDEN, HEAD_DROPOUT)
        model.load_state_dict(ckpt["model"], strict=True)  # fail fast if it doesn't load clean
        model.eval()
        models.append(model)
        print(f"[startup] loaded fold {fold} from {ckpt_path}")
    return models


def load_labels() -> tuple:
    import json

    if not os.path.exists(TRAIN_LABELS_CSV):
        raise RuntimeError(f"Missing train labels csv: {TRAIN_LABELS_CSV}")
    df = pd.read_csv(TRAIN_LABELS_CSV)
    df["source"] = df["filename"].apply(derive_source)
    df["exists"] = df["filename"].apply(lambda f: os.path.exists(os.path.join(TRAIN_IMAGES_DIR, f)))
    n_missing = (~df["exists"]).sum()
    if n_missing:
        print(f"[startup] warning: {n_missing} rows reference missing image files, skipping those")
    df = df[df["exists"]].reset_index(drop=True)
    if df.empty:
        raise RuntimeError(f"No usable images found under {TRAIN_IMAGES_DIR}")
    id_to_row = {row["image_id"]: row for _, row in df.iterrows()}

    clip_bounds_path = os.path.join(MODEL_DIR, "clip_bounds.json")
    if not os.path.exists(clip_bounds_path):
        raise RuntimeError(f"Missing clip bounds: {clip_bounds_path}")
    with open(clip_bounds_path) as f:
        clip_bounds = json.load(f)

    print(f"[startup] loaded {len(df)} usable rows from {TRAIN_LABELS_CSV}")
    return df, id_to_row, clip_bounds


@torch.no_grad()
def ensemble_predict(x: torch.Tensor, source: str) -> dict:
    """Run the 5-fold ensemble on a single preprocessed image tensor.

    Averages in kcal space (expm1 first, then mean), then clips to the
    per-source [min, p995] bounds. Returns the clipped prediction plus the
    raw pre-clip average and the individual fold predictions (for showing
    ensemble spread / uncertainty).
    """
    if x.dim() == 3:
        x = x.unsqueeze(0)  # (1,3,384,384)
    src_idx = torch.tensor([SOURCE_IDX[source]], dtype=torch.long)
    fold_kcal = []
    for model in state.models:
        z = model(x, src_idx)
        z = torch.clamp(z, max=LOG1P_CLAMP_MAX)
        fold_kcal.append(torch.expm1(z).item())
    avg_kcal = float(np.mean(fold_kcal))
    bounds = state.clip_bounds[source]
    clipped = float(np.clip(avg_kcal, bounds["min"], bounds["p995"]))
    return {
        "predicted_calories": clipped,
        "raw_calories": avg_kcal,
        "was_clipped": clipped != avg_kcal,
        "fold_predictions": fold_kcal,
        "source": source,
    }


def predict_kcal(image_path: str, source: str) -> float:
    x = load_and_preprocess(image_path)
    return ensemble_predict(x, source)["predicted_calories"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    state.models = load_models()
    state.labels_df, state.id_to_row, state.clip_bounds = load_labels()
    yield


app = FastAPI(title="Guess the Calories", lifespan=lifespan)


class PredictRequest(BaseModel):
    image_id: str


@app.get("/api/random-sample")
def random_sample():
    row = state.labels_df.sample(n=1).iloc[0]
    return {
        "image_id": row["image_id"],
        "image_url": f"/api/image/{row['image_id']}",
        "true_calories": float(row["calories"]),
        "source": row["source"],
    }


@app.get("/api/image/{image_id}")
def get_image(image_id: str):
    row = state.id_to_row.get(image_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Unknown image_id")
    path = os.path.join(TRAIN_IMAGES_DIR, row["filename"])
    return FileResponse(path)


@app.post("/api/predict")
def predict(req: PredictRequest):
    row = state.id_to_row.get(req.image_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Unknown image_id")
    path = os.path.join(TRAIN_IMAGES_DIR, row["filename"])
    predicted = predict_kcal(path, row["source"])
    true_calories = float(row["calories"])
    return {
        "predicted_calories": predicted,
        "true_calories": true_calories,
        "source": row["source"],
        "absolute_error": abs(predicted - true_calories),
    }


def resolve_source(filename: str, requested: str) -> str:
    """Pick the source embedding for an uploaded image.

    `requested` may be "A", "B", or "auto". "auto" derives from the file
    extension exactly like the training data (.png -> A, .jpg/.jpeg -> B);
    an unrecognized extension falls back to B (jpg-like photos), since the
    model requires one of the two known sources.
    """
    requested = (requested or "auto").strip().upper()
    if requested in SOURCE_IDX:
        return requested
    try:
        return derive_source(filename or "")
    except ValueError:
        return "B"


@app.post("/api/predict-upload")
async def predict_upload(file: UploadFile = File(...), source: str = Form("auto")):
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file upload")
    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="Could not decode the uploaded file as an image")

    resolved_source = resolve_source(file.filename, source)
    x = preprocess_pil(img)
    result = ensemble_predict(x, resolved_source)
    result["filename"] = file.filename
    result["source_auto"] = (source or "auto").strip().lower() in ("", "auto")
    return result


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
