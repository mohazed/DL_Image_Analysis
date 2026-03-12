"""
run_experiments.py
──────────────────────────────────────────────────────────────────────────────
Train multiple model configurations, then generate a prediction CSV for each.

Usage:
    python run_experiments.py

Edit CONFIGS below to add/remove/modify experiments.

Outputs (created automatically):
    weights/<config_id>_best.pt           — best checkpoint per config
    submissions/<config_id>_submission.csv — test-set predictions per config
    submissions/summary.csv               — all configs + best val AUC in one table

Dataset facts that informed config choices
──────────────────────────────────────────
  Train : 3 970 PNEUMONIA  +  1 341 NORMAL  (≈ 3:1 imbalance)
  Val   : 8 PNEUMONIA  +  8 NORMAL  (only 16 images — val AUC will be noisy)
  Test  : 529 unlabelled images

  pos_weight = num_negative / num_positive = 1341 / 3970 ≈ 0.338
  Setting pos_weight < 1 down-weights the majority class (PNEUMONIA) so the
  loss treats both classes roughly equally during training.
"""

import sys
import csv
import time
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import roc_auc_score
from PIL import Image

# Allow importing models from the project root
sys.path.append(str(Path(__file__).resolve().parent))
from models import MODEL_REGISTRY

# ── Paths ──────────────────────────────────────────────────────────────────────

ROOT         = Path(__file__).resolve().parent
TRAIN_DIR    = ROOT / "data" / "train"
VAL_DIR      = ROOT / "data" / "val"
TEST_DIR     = ROOT / "data" / "test_for_students"
WEIGHTS_DIR  = ROOT / "weights"
SUBMISSIONS_DIR = ROOT / "submissions"

WEIGHTS_DIR.mkdir(exist_ok=True)
SUBMISSIONS_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Experiment configurations ──────────────────────────────────────────────────
# Each dict becomes one training + prediction run.
# "id" names the weight file and submission CSV.
# "pos_weight": float  → passes pos_weight to BCEWithLogitsLoss to counter the
#                         3:1 class imbalance (1341 NORMAL / 3970 PNEUMONIA ≈ 0.338)
#               None   → unweighted loss (ablation / control)

CONFIGS = [
    # ── ResNet: primary architecture, two learning rates ───────────────────────
    # ResNet-18-style is the most reliable baseline for binary classification.
    # lr=3e-4 is a common sweet-spot for Adam on medical imaging tasks.
    {
        "id":            "resnet_lr3e4_weighted",
        "model_name":    "ResNet",
        "learning_rate": 3e-4,
        "epochs":        20,
        "batch_size":    32,
        "dropout_rate":  0.3,
        "image_size":    224,
        "pos_weight":    0.338,   # down-weights majority class (PNEUMONIA)
    },
    # lr=1e-3 trains faster; useful to see if it converges to the same quality.
    {
        "id":            "resnet_lr1e3_weighted",
        "model_name":    "ResNet",
        "learning_rate": 1e-3,
        "epochs":        15,
        "batch_size":    32,
        "dropout_rate":  0.3,
        "image_size":    224,
        "pos_weight":    0.338,
    },
    # Ablation: same as above but no pos_weight → measures imbalance impact.
    {
        "id":            "resnet_lr3e4_noweight",
        "model_name":    "ResNet",
        "learning_rate": 3e-4,
        "epochs":        20,
        "batch_size":    32,
        "dropout_rate":  0.3,
        "image_size":    224,
        "pos_weight":    None,
    },

    # ── Inception: multi-scale features suit chest X-ray pathology ────────────
    # Parallel 1×1/3×3/5×5 branches capture fine and coarse patterns at once.
    {
        "id":            "inception_lr3e4_weighted",
        "model_name":    "Inception",
        "learning_rate": 3e-4,
        "epochs":        20,
        "batch_size":    32,
        "dropout_rate":  0.4,
        "image_size":    224,
        "pos_weight":    0.338,
    },
    {
        "id":            "inception_lr1e3_weighted",
        "model_name":    "Inception",
        "learning_rate": 1e-3,
        "epochs":        15,
        "batch_size":    32,
        "dropout_rate":  0.4,
        "image_size":    224,
        "pos_weight":    0.338,
    },

    # ── U-Net: encoder-decoder features; heavier → smaller batch ──────────────
    # Skip connections force the network to preserve spatial context which can
    # highlight lung regions; lr=5e-4 balances speed vs. stability.
    {
        "id":            "unet_lr5e4_weighted",
        "model_name":    "U-Net",
        "learning_rate": 5e-4,
        "epochs":        15,
        "batch_size":    16,   # smaller batch: UNet is memory-heavy
        "dropout_rate":  0.3,
        "image_size":    224,
        "pos_weight":    0.338,
    },
]

# ── Transforms ─────────────────────────────────────────────────────────────────

def get_transforms(image_size: int, augment: bool = False):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    if augment:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            normalize,
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])


# ── Data loaders ───────────────────────────────────────────────────────────────

def get_dataloaders(image_size: int, batch_size: int):
    train_ds = datasets.ImageFolder(TRAIN_DIR, transform=get_transforms(image_size, augment=True))
    val_ds   = datasets.ImageFolder(VAL_DIR,   transform=get_transforms(image_size, augment=False))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, val_loader


# ── Train / eval helpers ───────────────────────────────────────────────────────

def run_epoch(model, loader, optimizer, criterion, training: bool):
    model.train() if training else model.eval()
    total_loss = 0.0
    all_labels, all_probs = [], []

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.float().unsqueeze(1).to(DEVICE)

            if training:
                optimizer.zero_grad()

            logits = model(images)
            loss   = criterion(logits, labels)

            if training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            all_probs.extend(probs.flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

    avg_loss = total_loss / len(loader.dataset)
    accuracy = float(np.mean((np.array(all_probs) >= 0.5) == np.array(all_labels)))
    auc      = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.5
    return avg_loss, accuracy, auc


# ── Training loop ──────────────────────────────────────────────────────────────

def train(cfg: dict) -> float:
    """Train one config. Returns best validation AUC."""
    print(f"\n{'='*60}")
    pw_str = f"{cfg.get('pos_weight'):.3f}" if cfg.get("pos_weight") is not None else "none"
    print(f"  Config : {cfg['id']}")
    print(f"  Model  : {cfg['model_name']}  |  LR {cfg['learning_rate']}  |  "
          f"Epochs {cfg['epochs']}  |  BS {cfg['batch_size']}  |  "
          f"Dropout {cfg['dropout_rate']}  |  Size {cfg['image_size']}  |  pos_weight {pw_str}")
    print(f"  Device : {DEVICE}")
    print(f"{'='*60}")

    if cfg["model_name"] not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{cfg['model_name']}'. Available: {list(MODEL_REGISTRY)}")

    train_loader, val_loader = get_dataloaders(cfg["image_size"], cfg["batch_size"])

    model     = MODEL_REGISTRY[cfg["model_name"]](in_channels=3, dropout_rate=cfg["dropout_rate"]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])

    pw = cfg.get("pos_weight")
    if pw is not None:
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw], device=DEVICE))
    else:
        criterion = nn.BCEWithLogitsLoss()

    best_val_auc  = 0.0
    weight_path   = WEIGHTS_DIR / f"{cfg['id']}_best.pt"

    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()
        tr_loss, tr_acc, tr_auc = run_epoch(model, train_loader, optimizer, criterion, training=True)
        vl_loss, vl_acc, vl_auc = run_epoch(model, val_loader,   optimizer, criterion, training=False)
        elapsed = time.time() - t0

        if vl_auc > best_val_auc:
            best_val_auc = vl_auc
            torch.save(model.state_dict(), weight_path)
            marker = " ✓ saved"
        else:
            marker = ""

        print(f"  Epoch {epoch:3d}/{cfg['epochs']} ({elapsed:.1f}s) | "
              f"Train  loss {tr_loss:.4f}  acc {tr_acc:.4f}  auc {tr_auc:.4f} | "
              f"Val    loss {vl_loss:.4f}  acc {vl_acc:.4f}  auc {vl_auc:.4f}"
              f"{marker}")

    print(f"\n  Best val AUC: {best_val_auc:.4f}  →  {weight_path}")
    return best_val_auc


# ── Prediction ─────────────────────────────────────────────────────────────────

def predict(cfg: dict):
    """Load best weights for a config and generate a submission CSV."""
    weight_path = WEIGHTS_DIR / f"{cfg['id']}_best.pt"
    if not weight_path.exists():
        print(f"  [skip predict] No weights found at {weight_path}")
        return

    model = MODEL_REGISTRY[cfg["model_name"]](in_channels=3).to(DEVICE)
    model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
    model.eval()

    transform   = get_transforms(cfg["image_size"], augment=False)
    image_paths = sorted(TEST_DIR.glob("*.jpeg")) + \
                  sorted(TEST_DIR.glob("*.jpg"))  + \
                  sorted(TEST_DIR.glob("*.png"))

    if not image_paths:
        print(f"  [skip predict] No test images found in {TEST_DIR}")
        return

    rows = []
    with torch.no_grad():
        for img_path in image_paths:
            image  = Image.open(img_path).convert("RGB")
            tensor = transform(image).unsqueeze(0).to(DEVICE)
            prob   = torch.sigmoid(model(tensor)).item()
            rows.append((img_path.stem, round(prob, 6)))

    out_path = SUBMISSIONS_DIR / f"{cfg['id']}_submission.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "prediction"])
        writer.writerows(rows)

    print(f"  Predictions ({len(rows)} images)  →  {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print(f"Running {len(CONFIGS)} experiment(s) on {DEVICE}\n")

    summary_rows = []

    for cfg in CONFIGS:
        best_auc = train(cfg)
        predict(cfg)
        summary_rows.append({
            "id":            cfg["id"],
            "model_name":    cfg["model_name"],
            "learning_rate": cfg["learning_rate"],
            "epochs":        cfg["epochs"],
            "batch_size":    cfg["batch_size"],
            "dropout_rate":  cfg["dropout_rate"],
            "image_size":    cfg["image_size"],
            "pos_weight":    cfg.get("pos_weight", ""),
            "best_val_auc":  round(best_auc, 6),
        })

    # Write summary table
    summary_path = SUBMISSIONS_DIR / "summary.csv"
    fieldnames = list(summary_rows[0].keys())
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\n{'='*60}")
    print(f"  Summary  →  {summary_path}")
    print(f"{'='*60}")
    print(f"  {'Config':<30} {'Model':<12} {'Best Val AUC':>12}")
    print(f"  {'-'*56}")
    for row in sorted(summary_rows, key=lambda r: r["best_val_auc"], reverse=True):
        print(f"  {row['id']:<30} {row['model_name']:<12} {row['best_val_auc']:>12.4f}")
    print()


if __name__ == "__main__":
    main()
