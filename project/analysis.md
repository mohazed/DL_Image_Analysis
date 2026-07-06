# Food Calorie Estimation — Dataset & Problem Analysis

> **Purpose.** Complete analysis of the "Can your model eat with its eyes?" calorie‑estimation challenge and the **full dataset** (now present and verified on disk). Written to be handed to a planning model (Claude Opus 4.8) as the full context for designing a strong, reproducible pipeline. All numbers below are computed from the actual `train_labels.csv` (3,098 rows) and by decoding every train/test image. The final section is an explicit brief for the planner.

---

## 1. The task in one paragraph

Predict the **total calorie content (kcal)** of a dish from a **single RGB photo**. It is a **regression** task (continuous float target) scored by **Mean Absolute Error (MAE)** — lower is better. Train = **3,098** images, test = **547** images. Test is split **public 30% / private 70%**; the public leaderboard is *not* the final grade, so **overfitting the public split is a real risk**. Max **10 submissions/day**. **All training must run inside a Kaggle notebook, be fully reproducible on re‑run, and use no external datasets.** Calories can't be read from appearance alone (a salad and a pasta bowl of equal size differ by hundreds of kcal), so the model must implicitly reason about **ingredient identity, density, and portion**.

---

## 2. Data integrity (verified)

The full dataset is present and clean:

| Item | Count | Notes |
|---|---|---|
| `train/images/` | **3,098** | 0 zero‑byte, **0 corrupt/undecodable**, all `RGB` |
| `train_labels.csv` | 3,098 rows | `image_id, filename, calories`; `image_id` == filename stem (verified) |
| `test/images/` | **547** | all decode, all `RGB` |
| `test_ids.csv` | 547 rows | `image_id, filename` |
| `sample_submission.csv` | 547 rows | constant `500` placeholder |

No non‑positive calorie values, no missing labels. Defensive image loading is still worth keeping for the reproducible re‑run, but the provided data itself is clean.

---

## 3. The two sources — the dominant structural fact

The brief mixes **two undisclosed sources** with different camera style/angle/lighting and calls generalizing across them "part of the challenge." **The file extension deterministically identifies the source**, confirmed by resolution, aspect ratio, and direct visual inspection of train images:

| Property | **Source A — `.png`** | **Source B — `.jpg`** |
|---|---|---|
| Viewpoint | **Top‑down / overhead** | **Oblique / side angle** |
| Train count | **2,355 (76.0%)** | **743 (24.0%)** |
| Test count | **417 (76.2%)** | **130 (23.8%)** |
| Resolution | **Uniform 640 × 480** (0.31 MP) | **Uniform 12.2 MP**: 3024×4032 (portrait, n=468) or 4032×3024 (landscape, n=275) |
| Aspect ratio | 1.33 (4:3 landscape only) | 0.75 or 1.33 (both orientations) |
| Color mode | RGB | RGB |
| Look | Controlled **overhead lab rig** | **Smartphone photo** on dark table |

**Train and test share the same 76/24 source split** — so any per‑source calibration learned on train transfers directly to test.

**Visual character (confirmed on train images):**
- **Source A (PNG, 640×480, top‑down)** — A **lab capture rig**: single dish on a white plate shot from directly above, with mounting hardware visible in frame (metal frame, screws, wires, glass turntable). Webcam‑grade quality. Classic **Nutrition5k‑style overhead capture**: fixed camera height ⇒ **the plate is a reliable physical scale reference**. Example: `train_0809.png` = a small salmon fillet, **84 kcal**, occupying a small central region of a large plate.
- **Source B (JPG, 12.2 MP, oblique)** — **Handheld smartphone photos** at an angle on a **dark tablecloth**, often **multi‑item composite meals**. Varying distance/angle ⇒ plate size is **not** a stable scale cue. Example: `train_2278.jpg` = lasagna + roast chicken + beef slices + a bread roll, **3,066 kcal**.

---

## 4. The target: calories

Computed from `train_labels.csv` (see `eda_calories.png` for histograms):

**Overall**
| stat | value |
|---|---|
| min / max | 50.0 / 3724.15 |
| mean / median | 380.8 / **220.0** |
| std | 545.0 |
| p25 / p75 | 119 / 397 |
| p90 / p95 / p99 | 715 / 1140 / 3123 |
| **skew** | **3.90** (heavy right tail) |
| skew of `log1p` | **0.66** (near‑symmetric) |
| unique values | 703 / 3098 |

**⇒ The right‑skew is severe; `log1p` almost fully symmetrizes it.** A log‑target is well motivated.

**Per source — the distributions are dramatically different:**
| source | n | mean | median | std | p5 | p95 | min | max |
|---|---|---|---|---|---|---|---|---|
| **A top‑down** | 2355 | 212.5 | **172** | 152 | 60 | 483 | 50 | 3051 |
| **B side** | 743 | 914.3 | **603** | 890 | 147 | 3092 | 67 | 3724 |

Source B dishes are **~3.5× higher in calories** and far more variable. Cross‑tab by calorie band makes the near‑separation explicit:

| kcal band | A top‑down | B side |
|---|---|---|
| 0–100 | 572 | 22 |
| 100–200 | 766 | 44 |
| 200–300 | 479 | 81 |
| 300–500 | 437 | 128 |
| 500–800 | 90 | 227 |
| 800–1500 | 10 | 119 |
| 1500–4000 | 1 | 122 |

Source A is almost entirely **<500 kcal**; Source B carries essentially **all** the high‑calorie mass. (The apparent `corr(calories, megapixels)=0.55` is *not* physical — megapixels is a perfect proxy for source, and source drives calories. Don't feed raw resolution as a "feature"; it just re‑encodes the domain.)

---

## 5. Baselines to beat (MAE)

Because MAE is minimized by the **conditional median**, constant‑predictor baselines are:

| Predictor | MAE |
|---|---|
| Global mean (380.8) | 297.8 |
| Global median (220) | 255.9 |
| Per‑source **mean** | 237.3 |
| **Per‑source median** (A→172, B→603) | **214.6** |

**The real bar is ~214.6 MAE** (predict each source's median). A model that doesn't clearly beat this adds nothing over knowing the file extension. This baseline should be computed and logged first, and every model compared against it **per source**.

---

## 6. ⚠️ Duplicate‑dish leakage risk (important for validation)

Labels repeat heavily: **95.7% of rows share their calorie value with at least one other row.**
- Source A: only **500 unique** calorie values across 2,355 images (~4.7 images/value).
- Source B: only **207 unique** across 743 images (~3.6 images/value).

Given the overhead‑rig capture style, this strongly suggests **the same physical dish photographed multiple times** (multiple frames per plate, identical label). If those near‑duplicate frames are split across train/validation folds, **CV will be optimistically biased** and model selection will be wrong.

**Mitigation for the planner to decide:** group‑aware splitting (e.g. `GroupKFold` with a group id derived from exact calorie value, or from perceptual‑hash / near‑duplicate image clusters), rather than plain random K‑fold. At minimum, quantify how much random‑vs‑grouped CV differs before trusting a number.

---

## 7. Modeling consequences (summary of the signals above)

1. **Domain is known at inference** (extension → source). Use it: as a conditioning feature, for per‑source output calibration, or via source‑specific heads. It's essentially free information the brief didn't hide well.
2. **Domain gap is the main generalization risk.** Public/private is over the whole test set; a model that fits A's clean look but not B's messy oblique meals (or vice‑versa) scores well publicly and badly privately. **Validate with source‑stratified (and ideally group‑aware) CV; always report per‑source MAE.**
3. **Sources ≈ calorie regimes** (A mostly <500, B mostly >500). The hard, high‑variance, high‑error mass lives in Source B — that's where MAE is won or lost. Absolute errors there are large, so B likely dominates total MAE despite being only 24% of images.
4. **Resolution mismatch (0.31 MP vs 12.2 MP).** Downscaling B's 12 MP oblique meals to 224² discards texture that signals density/portion; upscaling A's 640×480 costs little. Consider a **larger input and/or content‑aware crop for B**, and note B is the resolution‑sensitive one.
5. **Framing varies; food can occupy few pixels** (small item on a big plate in A; plate in a large dark margin in B). Plain center‑crop can cut food; global pooling dilutes it with background. Weigh plate/food localization (no masks provided — would be heuristic/self‑supervised) or higher resolution.
6. **Target & loss must match the metric.** MAE ⇒ predict the **median** ⇒ train with **L1 / Huber (smooth‑L1)**, not MSE. Consider `log1p` target (fixes skew) but reconcile the *loss × transform × metric* interaction deliberately and **clip predictions** to a sane kcal range (e.g. train min/max or robust percentiles) to avoid tail blow‑ups.

---

## 8. Constraints that shape the pipeline

| Constraint | Implication |
|---|---|
| **Kaggle notebook only**, no external GPU | Free GPU = **P100 16 GB** or **2×T4 16 GB**. Fit ≤16 GB/device. Session ~9–12 h but keep well under. |
| Must **also run on Colab Pro (A100 40 GB)** | Make batch size / precision configurable (bf16 + bigger batch on A100). One config, two ceilings. |
| **Fully reproducible re‑run** | Seed Python/NumPy/torch, deterministic cuDNN & dataloader order, saved fixed splits, pinned lib versions. |
| **No external datasets** | ImageNet‑pretrained backbones are fine (weights ≠ dataset), **but re‑run may have internet OFF** → bundle weights as a Kaggle *Models*/dataset attachment or `timm` offline cache. Don't rely on a live download. |
| **≤10 submissions/day**, public≠private | Trust **CV**, not the LB. Group‑aware, source‑stratified CV is the decision signal. |
| Images **not resized on disk** (0.31 MP ↔ 12.2 MP, mixed orientation) | Resize inside the `DataLoader`; handle both regimes and portrait/landscape. Pre‑resize/cache B to speed epochs. |
| "Doesn't take long" | Favor **fine‑tuning a pretrained CNN/ViT** at modest resolution over training from scratch. Target ≲1–2 h on A100, a few hours on Kaggle P100. |

---

## 9. Established facts (quick reference)

- **Task:** single image → total kcal, **regression**, metric **MAE**.
- **Data (clean, verified):** 3,098 train / 547 test; test public 30% / private 70%; public LB ≠ grade.
- **Two extension‑separable sources, same 76/24 split in train & test:** A = 640×480 top‑down lab rig (Nutrition5k‑like), median **172** kcal; B = 12.2 MP oblique smartphone on dark table, median **603** kcal.
- **Target right‑skewed** (skew 3.90; `log1p` skew 0.66); range 50–3724; median 220.
- **Baseline to beat: ~214.6 MAE** (per‑source median).
- **Leakage risk:** 95.7% of labels repeat ⇒ likely duplicate dishes ⇒ use **group‑aware CV**.
- **Constraints:** Kaggle‑only reproducible training, ≤16 GB GPU, also fit Colab A100, ≤10 subs/day, no external datasets, offline pretrained weights.

---

## 10. Brief for the planner (Claude Opus 4.8)

Using everything above, design a **complete, reproducible training + inference pipeline** for this competition. Make concrete, justified decisions (not a menu) on:

1. **Baselines** — reproduce the per‑source‑median bar (~214.6 MAE) and a simple pretrained‑backbone regressor as references.
2. **Backbone & input** — which pretrained architecture(s) (e.g. EfficientNet / ConvNeXt / ViT via `timm`), input resolution(s), and whether **Source B** gets a larger input or content‑aware crop. Justify against the 16 GB / "runs fast" budget and specify how weights are available **offline** on Kaggle.
3. **Source handling** — single model with a `source` flag vs source‑conditioned normalization vs two heads / two models, plus any **per‑source output calibration**, grounded in the §4 per‑source distributions and the fact that Source B dominates MAE.
4. **Target & loss** — transform (raw vs `log1p`) and loss (**L1 vs Huber**), consistent with MAE being median‑optimal; plus prediction **clipping**.
5. **Augmentation** — what's safe for a *portion/calorie* task (flips, small rotations, color jitter OK; aggressive crop/zoom/mixup change apparent portion and are risky → justify). Handle mixed orientation & the "food fills little of the frame" case.
6. **Validation** — **group‑aware, source‑stratified K‑fold** (groups from repeated calorie value / near‑duplicate clusters), report **overall and per‑source MAE**, and a rule for trusting CV over the public LB. Optional TTA / fold ensembling within the time budget.
7. **Reproducibility & runtime** — seeding, deterministic dataloaders, pinned versions, offline pretrained weights, one config that runs on **Kaggle P100/T4** and scales up on **Colab A100** within a "doesn't take long" budget.
8. **Inference & submission** — dataloader resizing for the two size regimes, robust loading, clipping, and writing the exact `image_id,predicted_calories` format (547 rows, correct header) required.

Open the notebook by recomputing the §5 baselines and a quick group‑vs‑random CV check, then let those numbers confirm or adjust the choices above before the full run.
