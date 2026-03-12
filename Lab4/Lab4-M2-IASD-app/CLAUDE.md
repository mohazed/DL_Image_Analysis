# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Pneumonia detection lab using chest X-ray images. Binary classification (PNEUMONIA vs. NORMAL) with three pluggable PyTorch architectures, a FastAPI backend for training/inference, and a Streamlit frontend.

## Running the Application

```bash
# Install dependencies
pip install -r requirements.txt

# Terminal 1: Start FastAPI backend (port 8000)
cd api && uvicorn main:app --reload --port 8000

# Terminal 2: Start Streamlit frontend (port 8501)
cd app && streamlit run streamlit_app.py
```

No testing framework or linting is configured for this project.

## Architecture

**Three-tier design:**
- `app/streamlit_app.py` — Multi-page UI (Home, Train, Results, Submit). Sends HTTP requests to the FastAPI backend and manages state via `st.session_state`.
- `api/main.py` — Two endpoints: `POST /train` (runs training loop, returns metrics/curves) and `POST /predict` (loads best weights, generates CSV predictions for unlabeled test set). Saves best model weights to `api/weights/` based on validation AUC.
- `models/` — Three PyTorch architectures registered in `MODEL_REGISTRY` (in `models/__init__.py`): `"U-Net"`, `"ResNet"`, `"Inception"` (exact string keys). All output `(B, 1)` logits for BCEWithLogitsLoss.

**Model architectures:**
- `unet.py` — Encoder-decoder with skip connections, 4 encoder/decoder levels, GlobalAvgPool to classifier head
- `resnet.py` — ResNet-18-style with 4 residual layers, configurable base filters, Kaiming init
- `inception.py` — 4 Inception blocks with 4 parallel branches (1×1, 1×1→3×3, 1×1→5×5, MaxPool→1×1), concatenated outputs

**Data layout:**
```
data/train/{PNEUMONIA,NORMAL}/   # Training images
data/val/{PNEUMONIA,NORMAL}/     # Validation images
data/test_for_students/          # Unlabeled test set (~530 images)
submission/sample_submission.csv # Output format: id, prediction
```

## Key Implementation Notes

- Model architecture files (`models/unet.py`, `models/resnet.py`, `models/inception.py`) may contain `# TO COMPLETE` skeleton sections — this is an educational lab where students fill in implementations. Each model file has a `__main__` block for quick shape sanity checks (`python models/resnet.py`).
- The `MODEL_REGISTRY` dict in `models/__init__.py` maps string names to model classes; add new architectures there to make them available in the API and UI.
- Training metrics tracked per epoch: loss, accuracy, ROC-AUC for both train and val splits.
- Best model checkpoint is saved based on **validation AUC**, not accuracy. Weights saved to `api/weights/<ModelName>_best.pt`.
- `/predict` loads weights by model name — use the same `image_size` at inference as was used during training.
- Data loading uses `torchvision.datasets.ImageFolder` — directory structure determines class labels (class index 0 = NORMAL, 1 = PNEUMONIA, alphabetical order).
