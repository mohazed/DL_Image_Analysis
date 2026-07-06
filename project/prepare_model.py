"""One-off prep: turn the 5 fold{f}_best.pt training checkpoints into the clean,
inference-only artifacts the app actually needs (model/prepared/fold{f}.pt +
clip_bounds.json). Matches CalorieNet from calorie_pipeline.py exactly.

Run: python3 prepare_model.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn

CKPT_DIR = "checkpoints"
OUT_DIR = "model/prepared"
TRAIN_LABELS_CSV = "m2-food-calorie-estimation/train_labels.csv"
N_FOLDS = 5

BACKBONE_NAME = "convnext_tiny"
SOURCE_EMB_DIM = 16
HEAD_HIDDEN = 256
HEAD_DROPOUT = 0.2
IMG_SIZE = 384


class CalorieNet(nn.Module):
    """Must match calorie_pipeline.py's CalorieNet exactly (backbone + source
    embedding + MLP head) so fold*_best.pt state_dicts load with strict=True."""

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


def prepare_fold(fold: int) -> dict:
    ckpt_path = f"{CKPT_DIR}/fold{fold}_best.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model"]

    model = CalorieNet(BACKBONE_NAME, SOURCE_EMB_DIM, HEAD_HIDDEN, HEAD_DROPOUT)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        print(f"\n[FOLD {fold}] STATE_DICT MISMATCH -- stopping.\n{e}\n")
        sys.exit(1)
    model.eval()

    # dummy forward pass: random image + both source indices
    torch.manual_seed(0)
    x = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    outputs = {}
    with torch.no_grad():
        for src_name, src_idx in (("A", 0), ("B", 1)):
            out = model(x, torch.tensor([src_idx], dtype=torch.long))
            val = out.item()
            if not np.isfinite(val):
                print(f"\n[FOLD {fold}] Dummy forward pass produced non-finite output ({val}) for source {src_name} -- stopping.\n")
                sys.exit(1)
            outputs[src_name] = val

    out_path = f"{OUT_DIR}/fold{fold}.pt"
    torch.save({"model": state_dict, "fold": fold}, out_path)

    return {
        "fold": fold,
        "epoch": ckpt.get("epoch"),
        "best_val_mae": ckpt.get("best_val_mae"),
        "out_path": out_path,
        "out_size": os.path.getsize(out_path),
        "in_size": os.path.getsize(ckpt_path),
        "dummy_out_A": outputs["A"],
        "dummy_out_B": outputs["B"],
    }


def compute_clip_bounds() -> dict:
    df = pd.read_csv(TRAIN_LABELS_CSV)
    df["source"] = df["filename"].apply(derive_source)
    bounds = {}
    for src in ["A", "B"]:
        sub = df.loc[df["source"] == src, "calories"]
        bounds[src] = {"min": float(sub.min()), "p995": float(sub.quantile(0.995))}
    return bounds


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"{'='*70}\nPreparing {N_FOLDS} fold checkpoints -> {OUT_DIR}/\n{'='*70}")
    results = []
    for fold in range(N_FOLDS):
        print(f"\n[fold {fold}] loading {CKPT_DIR}/fold{fold}_best.pt ...")
        r = prepare_fold(fold)
        print(f"[fold {fold}] state_dict loaded OK (strict=True). "
              f"trained epoch={r['epoch']}, best_val_mae={r['best_val_mae']:.4f}")
        print(f"[fold {fold}] dummy forward pass: source A -> {r['dummy_out_A']:.6f} "
              f"(log1p-kcal), source B -> {r['dummy_out_B']:.6f} (log1p-kcal) -- both finite.")
        print(f"[fold {fold}] saved stripped checkpoint -> {r['out_path']} "
              f"({r['out_size']/1e6:.1f} MB, was {r['in_size']/1e6:.1f} MB)")
        results.append(r)

    print(f"\n{'='*70}\nDeriving source + computing per-source clip bounds from {TRAIN_LABELS_CSV}\n{'='*70}")
    clip_bounds = compute_clip_bounds()
    clip_bounds_path = f"{OUT_DIR}/clip_bounds.json"
    with open(clip_bounds_path, "w") as f:
        json.dump(clip_bounds, f, indent=2)
    print(f"clip_bounds = {json.dumps(clip_bounds, indent=2)}")
    print(f"saved -> {clip_bounds_path}")

    # --- summary table ---
    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    header = f"{'fold':<6}{'loaded':<10}{'epoch':<8}{'best_val_mae':<15}{'dummy_out(A)':<15}{'dummy_out(B)':<15}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r['fold']:<6}{'OK':<10}{r['epoch']:<8}{r['best_val_mae']:<15.4f}"
              f"{r['dummy_out_A']:<15.6f}{r['dummy_out_B']:<15.6f}")

    print("\nPer-source clip bounds [train_min, train_p99.5] (kcal):")
    for src, b in clip_bounds.items():
        print(f"  source {src}: min={b['min']:.2f}, p99.5={b['p995']:.2f}")

    out_files = sorted(os.listdir(OUT_DIR))
    n_fold_files = sum(1 for f in out_files if f.startswith("fold") and f.endswith(".pt"))
    has_clip_bounds = "clip_bounds.json" in out_files
    print(f"\n{OUT_DIR}/ contains {len(out_files)} files: {out_files}")
    assert n_fold_files == N_FOLDS, f"expected {N_FOLDS} fold*.pt files, found {n_fold_files}"
    assert has_clip_bounds, "clip_bounds.json missing"
    print(f"Confirmed: exactly {N_FOLDS} fold checkpoints + clip_bounds.json present.")

    # --- size comparison ---
    orig_size = sum(r["in_size"] for r in results)
    prepared_size = sum(r["out_size"] for r in results) + os.path.getsize(clip_bounds_path)
    print(f"\n{'='*70}\nSIZE COMPARISON\n{'='*70}")
    print(f"Original 5 fold*_best.pt checkpoints: {orig_size/1e6:.1f} MB ({orig_size/1e9:.3f} GB)")
    print(f"Prepared model/prepared/ output:       {prepared_size/1e6:.1f} MB ({prepared_size/1e9:.3f} GB)")
    print(f"Reduction: {(1 - prepared_size/orig_size)*100:.1f}% smaller, "
          f"saved {(orig_size - prepared_size)/1e6:.1f} MB")


if __name__ == "__main__":
    main()
