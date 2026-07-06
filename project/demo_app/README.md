# Guess the Calories — Demo App

A local demo for the CalorieNet 5-fold ensemble model, with two modes:

- **Dataset game** — picks a random photo from the training set (for which the
  true calorie count is already known), lets the model predict the calorie
  count, then reveals the true value next to the prediction.
- **Upload a photo** — upload (or drag-and-drop) any food photo of your own and
  run the exact same inference pipeline (letterbox → ImageNet-normalize →
  5-fold ensemble → per-source clip) to get a predicted calorie count. Since
  uploads have no ground truth, it shows the prediction plus the per-fold
  ensemble spread instead of an error against a true value.

## Project structure

```
demo_app/
├── README.md          this file
├── requirements.txt    pinned Python dependencies
├── main.py             FastAPI backend: loads the 5-fold ensemble once at
│                        startup, exposes /api/random-sample, /api/image/{id},
│                        /api/predict, /api/predict-upload, and serves the frontend
└── static/
    ├── index.html       page markup (Dataset game + Upload tabs)
    ├── style.css         dark theme styling
    └── app.js            tab switching, shuffle/guess/reveal, and upload/predict logic
```

### Where the backend reads its data from

`main.py` resolves everything relative to the project root (one directory up
from `demo_app/`), so it works regardless of your current working directory
as long as the layout below is intact:

```
project/
├── demo_app/                                    <- this app
├── model/prepared/
│   ├── fold0.pt ... fold4.pt                     inference-only checkpoints
│   └── clip_bounds.json                          per-source {min, p995} kcal
└── m2-food-calorie-estimation/
    ├── train_labels.csv                          columns: image_id, filename, calories
    └── train/images/                             the actual jpg/png files
```

No files under `checkpoints/` (the full training checkpoints) are used by the
demo — only the prepared, inference-only artifacts in `model/prepared/`.

## Setup

Requires Python 3.10+.

```bash
cd demo_app
pip install -r requirements.txt
```

## Running

```bash
cd demo_app
uvicorn main:app --reload --port 8000
```

Then open **http://127.0.0.1:8000** in a browser.

On startup the server:
1. Builds 5 `CalorieNet` instances and loads `model/prepared/fold{0..4}.pt`
   with `strict=True` — it fails fast with a clear error if any checkpoint
   doesn't load cleanly.
2. Loads `train_labels.csv`, derives each row's source (`.png` → A, `.jpg`/
   `.jpeg` → B), and drops any rows whose image file is missing on disk
   (logged to the console).
3. Loads `clip_bounds.json`.

This all happens once — not per-request — so the first request after startup
is already fast.

## Using the demo

1. **Shuffle** — loads a new random sample photo (with its source tag, A or
   B) from the training set. The true calorie count is hidden at this point.
2. **Guess the calories** — runs the 5-fold ensemble on the currently shown
   photo and reveals the predicted calories next to the true calories, plus
   the absolute error:
   - green: error < 50 kcal
   - amber: error 50–150 kcal
   - red: error > 150 kcal
3. Click **Shuffle** again to try another photo.

### Upload a photo

Switch to the **Upload a photo** tab to run the model on your own image:

1. Click the drop zone (or drag a photo onto it) to choose an image.
2. Pick a **Source**: `Auto` derives it from the file type exactly like the
   training data (`.png` → A, `.jpg`/`.jpeg` → B); or force `A`/`B` explicitly.
   An unrecognized extension in Auto mode falls back to B.
3. Click **Predict calories**. You get the ensemble-averaged, clipped
   prediction, the per-fold breakdown (a rough sense of model confidence), and
   a note if the raw average was clipped to the source range.

There is no "true" value for uploads, so no error badge is shown — the model
never saw these images and has no ground truth to compare against.

## How a prediction is computed

For the currently displayed image:

1. **Preprocess** — EXIF-transpose, convert to RGB, letterbox-resize
   (aspect-preserving, black-padded) to 384×384, normalize with ImageNet
   mean/std. This matches training preprocessing exactly (no crop, no
   squash, no augmentation).
2. **Ensemble** — run all 5 fold models on the same preprocessed image with
   the image's source embedding, clamp each model's output at
   `log1p(4000)`, then `expm1` to convert to kcal. Average the 5 kcal values
   (averaged in kcal space, not log space).
3. **Clip** — clip the averaged kcal prediction to
   `clip_bounds.json[source]["min"..."p995"]` for that image's source.

Simplification vs. the original training pipeline: no test-time augmentation
(dihedral TTA for source A, hflip TTA for source B) and no per-source
calibration step — just the plain 5-fold ensemble average + clip.

## API reference

| Method | Path                    | Description |
|--------|-------------------------|--------------|
| GET    | `/api/random-sample`    | Returns a random sample: `{ image_id, image_url, true_calories, source }` (no prediction). |
| GET    | `/api/image/{image_id}` | Serves the image file for that `image_id`. |
| POST   | `/api/predict`          | Body `{ image_id }`. Returns `{ predicted_calories, true_calories, source, absolute_error }`. |
| POST   | `/api/predict-upload`   | Multipart form: `file` (image) + optional `source` (`auto`/`A`/`B`). Returns `{ predicted_calories, raw_calories, was_clipped, fold_predictions, source, source_auto, filename }`. |

`image_id` is always resolved server-side against the loaded labels table —
arbitrary/unknown ids return `404`. For `/api/predict-upload`, an empty or
undecodable file returns `400`.

## Notes

- Google Fonts (Space Grotesk, IBM Plex Sans, IBM Plex Mono) are loaded from
  `fonts.googleapis.com` at page load, so an internet connection is needed
  the first time the page renders.
- Runs on CPU fine; no GPU required for inference with 5 ConvNeXt-Tiny
  models on single images.
