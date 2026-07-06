# CLAUDE.md

## Building a website/app to test the calorie-estimation model

Use the artifacts in `model/prepared/`, NOT the raw files in `checkpoints/`
(those are full training checkpoints with optimizer/scheduler/RNG state, 3x
larger, and unnecessary for inference).

### Files to load

- `model/prepared/fold{0,1,2,3,4}.pt` — 5 inference-only checkpoints, one per
  CV fold. Each is `{"model": state_dict, "fold": f}`. Produced by
  `prepare_model.py` from `checkpoints/fold{f}_best.pt` (the `_last.pt` files
  are mid-training snapshots and were intentionally ignored).
- `model/prepared/clip_bounds.json` — per-source `{"min": ..., "p995": ...}`
  in kcal, e.g. `{"A": {...}, "B": {...}}`. Used to clip final predictions.

### Model architecture ("CalorieNet")

Defined in `calorie_pipeline.py` (search `class CalorieNet`) and duplicated
standalone in `prepare_model.py`. Reuse the `prepare_model.py` version — it
has no dependency on downloading pretrained ImageNet weights (fine for
inference since the full state_dict is loaded with `strict=True` anyway):

```
image (3x384x384) -> timm.create_model("convnext_tiny", pretrained=False,
    num_classes=0, global_pool="avg")  -> pooled feature (768-d)
concat with nn.Embedding(2, 16)[source_idx]   # source_idx: A=0, B=1
-> Linear(feat_dim+16, 256) -> GELU -> Dropout(0.2) -> Linear(256, 1)
-> scalar z, interpreted as log1p(kcal)
```

Load each fold's state_dict with `strict=True` — if it doesn't load clean,
something is wrong (don't paper over it).

### Preprocessing (must match training exactly, see `calorie_pipeline.py` §11/§13)

- Source is 100% determined by filename extension: `.png` -> `"A"` (index 0),
  `.jpg`/`.jpeg` -> `"B"` (index 1).
- Letterbox resize (aspect-preserving, black-padded to square) to 384x384 —
  no center-crop, no squash.
- Normalize with standard ImageNet mean/std: `[0.485,0.456,0.406]` /
  `[0.229,0.224,0.225]`.
- No augmentation at inference (that's train-only).

### Inference recipe (see `calorie_pipeline.py` §22-23 for the full original logic)

1. Run all 5 folds on the input image, average predictions **in kcal space**
   (apply `expm1` to each fold's log1p output first, then average) — not in
   log space.
2. The original pipeline also does 8-way dihedral TTA for source A and hflip
   TTA for source B, plus a per-source calibration step before clipping (see
   §21-22). A minimal test website can skip TTA/calibration and just do the
   5-fold ensemble average + clip below; note this explicitly as a
   simplification if you do.
3. Clip the final per-image prediction to `[clip_bounds[source]["min"],
   clip_bounds[source]["p995"]]` from `model/prepared/clip_bounds.json`.

### Data for sample images / sanity checks

- `m2-food-calorie-estimation/train/images/`, `train_labels.csv` — has ground
  truth calories, useful for spot-checking predictions.
- `m2-food-calorie-estimation/test/images/`, `test_ids.csv`,
  `sample_submission.csv` — no ground truth, matches the original
  competition's held-out set.

### Reference docs

- `analysis.md` — ground-truth EDA (per-source distributions, medians, the
  ~214.6 MAE per-source-median baseline every model must beat).
- `overview.md` — original competition rules/format.
- `calorie_pipeline.py` / `calorie_pipeline.ipynb` — full original
  train+inference pipeline (Colab notebook, exported to `.py`). Treat as the
  source of truth for exact preprocessing/model/inference behavior.
