# ==============================================================================
# CALORIE REGRESSION — V2 "MAX PERFORMANCE" PIPELINE (single-cell, run top to bottom)
# ==============================================================================
# Safe to run IN PARALLEL with the v1 notebook on the SAME Google Drive project:
#   - reads the same PROJECT_DIR/data and the SAME shared folds.csv (so v1/v2
#     OOF predictions are directly blendable afterwards)
#   - every output is isolated under PROJECT_DIR/v2/ (cache, checkpoints, oof, logs)
#   - final artifact: PROJECT_DIR/submission_v2.csv
#
# Upgrades over v1 — each targets the actual error budget (fold results showed
# Source B contributes ~63% of total MAE despite being 24% of images):
#   1. Backbone : convnext_tiny/IN-1k -> convnext_base.fb_in22k_ft_in1k_384
#                 (IN-22k pretraining includes many food classes; FT'd at 384px)
#   2. Input    : 384 -> 448 letterbox + PLATE-CROP for Source B at cache time.
#                 B = 12MP oblique smartphone photo of a plate on a DARK cloth;
#                 Otsu-threshold the dark background away and crop to the bright
#                 plate/food region -> the food gets several x more pixels at
#                 the same input size. Deterministic, applied to train AND test,
#                 with conservative fallbacks to the full frame.
#   3. Schedule : 14 -> 26 epochs (v1's val MAE was still improving at epoch 14),
#                 2-epoch warmup + cosine; EMA of weights (decay 0.999, ramped);
#                 best checkpoint = whichever of raw/EMA validates better.
#   4. Speed/reg: drop_path 0.2 on the backbone, channels_last + bf16 on A100.
#   Kept from v1 (correct as designed): portion-safe source-specific augs,
#   kcal-space L1 on a log1p head, grouped+stratified CV, cross-validated
#   per-source calibration, per-source clipping, dihedral TTA for A.
#
# Runtime: ~3.5-4.5h full run on Colab A100 40GB. Resumable per epoch: on a
# disconnect just re-run this cell. For a Kaggle T4 re-run, set
# BACKBONE="convnext_small.fb_in22k_ft_in1k_384", IMG_SIZE=384 (one line each).
# ==============================================================================

import sys, subprocess, os, json, math, time, random, glob, shutil, warnings, copy
from collections import defaultdict

try:
    import google.colab  # noqa: F401
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "timm==1.0.*", "imagehash==4.*", "scikit-learn>=1.2"], check=True)

import numpy as np
import pandas as pd
import cv2
from PIL import Image, ImageOps
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import timm

warnings.filterwarnings("ignore", category=UserWarning)

# ------------------------------- config --------------------------------------
SMOKE_TEST      = False   # True = tiny subset / 1 fold / 2 epochs, verifies the full path
BACKBONE        = "convnext_base.fb_in22k_ft_in1k_384"
IMG_SIZE        = 448
N_FOLDS         = 5
SEED            = 42
EPOCHS          = 26
WARMUP_EPOCHS   = 2
EFFECTIVE_BATCH = 32
HEAD_LR         = 1e-3
BACKBONE_LR     = 6e-5    # gentler than v1's 1e-4: bigger pretrained model
WEIGHT_DECAY    = 0.05
DROP_PATH       = 0.2
HEAD_HIDDEN     = 256
HEAD_DROPOUT    = 0.2
SOURCE_EMB_DIM  = 16
EMA_DECAY       = 0.999
GRAD_CLIP       = 1.0
PHASH_THRESH    = 4       # only used if the shared folds.csv is missing
N_CAL_BINS      = 5
NUM_WORKERS     = 8 if IN_COLAB else 0
SMOKE_PER_SOURCE = 40
LOG1P_CLAMP_MAX = math.log1p(4000.0)  # numeric safety clamp, not a prediction clip

# ------------------------------- paths ---------------------------------------
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    _candidates = ["/content/drive/MyDrive/calorie_comp", "/content/drive/MyDrive/calorie_project"]
    PROJECT_DIR = next((p for p in _candidates if os.path.exists(f"{p}/data/train/images")), None)
    assert PROJECT_DIR, f"No data found under any of {_candidates} -- edit PROJECT_DIR here."
    LOCAL_CACHE_DIR = "/content/local_cache_v2"
else:
    PROJECT_DIR = os.path.abspath("local_run")
    LOCAL_CACHE_DIR = f"{PROJECT_DIR}/v2/local_cache"

DATA_DIR   = f"{PROJECT_DIR}/data"
V2_DIR     = f"{PROJECT_DIR}/v2"          # ALL v2 outputs live here (parallel-safe vs v1)
CACHE_DIR  = f"{V2_DIR}/cache"
WEIGHTS_DIR= f"{PROJECT_DIR}/weights"     # shared dir, v2 uses its own filename
CKPT_DIR   = f"{V2_DIR}/checkpoints"
OOF_DIR    = f"{V2_DIR}/oof"
LOG_DIR    = f"{V2_DIR}/logs"
SHARED_FOLDS_CSV = f"{PROJECT_DIR}/folds.csv"   # written by v1; reused read-only
SUBMISSION_PATH  = f"{PROJECT_DIR}/submission_v2.csv"
for d in [CACHE_DIR, WEIGHTS_DIR, CKPT_DIR, OOF_DIR, LOG_DIR, LOCAL_CACHE_DIR]:
    os.makedirs(d, exist_ok=True)
for name, p in {"train images": f"{DATA_DIR}/train/images", "test images": f"{DATA_DIR}/test/images",
                "train_labels.csv": f"{DATA_DIR}/train_labels.csv", "test_ids.csv": f"{DATA_DIR}/test_ids.csv"}.items():
    assert os.path.exists(p), f"missing {name}: {p}"
print("PROJECT_DIR:", PROJECT_DIR, "| v2 outputs ->", V2_DIR)

# --------------------------- smoke overrides ----------------------------------
N_FOLDS_TO_RUN = N_FOLDS
if SMOKE_TEST:
    EPOCHS, WARMUP_EPOCHS, N_FOLDS_TO_RUN = 2, 0, 1
    if not IN_COLAB:  # local CPU dry-run: smallest possible real pass
        BACKBONE, SMOKE_PER_SOURCE, EPOCHS = "convnext_tiny", 8, 1
    print(f"SMOKE_TEST: epochs={EPOCHS}, folds={N_FOLDS_TO_RUN}, backbone={BACKBONE}")

# ----------------------------- reproducibility --------------------------------
def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
seed_everything(SEED)

def seed_worker(worker_id):
    ws = torch.initial_seed() % (2**32)
    np.random.seed(ws); random.seed(ws)

def make_generator(seed):
    g = torch.Generator(); g.manual_seed(int(seed)); return g

# ------------------------------ GPU detection ---------------------------------
def detect_device():
    if not torch.cuda.is_available():
        return {"device": torch.device("cpu"), "micro_batch": 2, "amp_dtype": torch.float32, "use_scaler": False, "name": "cpu"}
    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if vram >= 35:   mb, dt, sc = 16, torch.bfloat16, False   # A100 40GB @ 448px convnext_base
    elif vram >= 20: mb, dt, sc = 8, torch.bfloat16, False
    elif vram >= 10: mb, dt, sc = 4, torch.float16, True      # T4/L4 16GB (slow -- see header)
    else:            mb, dt, sc = 2, torch.float16, True
    return {"device": torch.device("cuda"), "micro_batch": mb, "amp_dtype": dt, "use_scaler": sc, "name": f"{name} ({vram:.0f}GB)"}
GPU = detect_device()
ACCUM_STEPS = max(1, round(EFFECTIVE_BATCH / GPU["micro_batch"]))
print(f"GPU: {GPU['name']} | micro_batch={GPU['micro_batch']} x accum={ACCUM_STEPS} = effective {GPU['micro_batch']*ACCUM_STEPS}")

# ------------------------------- data ------------------------------------------
train_df = pd.read_csv(f"{DATA_DIR}/train_labels.csv")
test_df  = pd.read_csv(f"{DATA_DIR}/test_ids.csv")
for df in (train_df, test_df):
    df["source"] = df["filename"].map(lambda f: "A" if f.lower().endswith(".png") else "B")
train_df["path"] = train_df["filename"].map(lambda f: f"{DATA_DIR}/train/images/{f}")
test_df["path"]  = test_df["filename"].map(lambda f: f"{DATA_DIR}/test/images/{f}")

assert len(train_df) == 3098 and len(test_df) == 547, "unexpected dataset size"
MEDIANS = train_df.groupby("source")["calories"].median().to_dict()
BASE_MAE = {s: (train_df.loc[train_df.source == s, "calories"] - MEDIANS[s]).abs().mean() for s in "AB"}
print(f"per-source-median baselines to beat: A={BASE_MAE['A']:.1f}  B={BASE_MAE['B']:.1f}  "
      f"overall={(train_df['calories'] - train_df['source'].map(MEDIANS)).abs().mean():.1f}")

def mae(y, p): return np.abs(np.asarray(y, float) - np.asarray(p, float)).mean()

# ----------------- folds: reuse v1's shared folds.csv (blendable CV) -----------
if os.path.exists(SHARED_FOLDS_CSV):
    folds_df = pd.read_csv(SHARED_FOLDS_CSV)
    assert set(folds_df["image_id"]) == set(train_df["image_id"]), "shared folds.csv doesn't match train_labels.csv"
    train_df = train_df.merge(folds_df[["image_id", "group_id", "strat_label", "fold"]], on="image_id", how="left")
    print(f"Loaded SHARED folds from {SHARED_FOLDS_CSV} (same CV as v1 -> OOF is blendable).")
else:
    # identical construction to v1 (pHash near-dup + exact-calorie union, per source)
    print("Shared folds.csv not found -- rebuilding with v1's exact procedure/seed...")
    import imagehash
    from scipy.spatial.distance import pdist, squareform
    from sklearn.model_selection import StratifiedGroupKFold

    class UF:
        def __init__(s, n): s.p = list(range(n))
        def find(s, x):
            while s.p[x] != x: s.p[x] = s.p[s.p[x]]; x = s.p[x]
            return x
        def union(s, a, b):
            ra, rb = s.find(a), s.find(b)
            if ra != rb: s.p[ra] = rb

    def phash_safe(path):
        try:
            with Image.open(path) as im:
                im = ImageOps.exif_transpose(im).convert("RGB"); im.thumbnail((256, 256))
                return imagehash.phash(im, hash_size=8)
        except Exception as e:
            warnings.warn(f"pHash failed {path}: {e}")
            return imagehash.ImageHash(np.zeros((8, 8), dtype=bool))

    gid = np.full(len(train_df), -1, np.int64); off = 0
    for src in "AB":
        idx = np.where((train_df["source"] == src).values)[0]
        d = train_df.iloc[idx].reset_index(drop=True); uf = UF(len(d))
        bits = np.stack([phash_safe(p).hash.flatten() for p in d["path"]]).astype(np.uint8)
        ham = squareform(pdist(bits, "hamming")) * bits.shape[1]
        for i, j in np.argwhere(np.triu(ham <= PHASH_THRESH, k=1)): uf.union(int(i), int(j))
        v2i = defaultdict(list)
        for i, v in enumerate(d["calories"].round(2).values): v2i[v].append(i)
        for ix in v2i.values():
            for k in range(1, len(ix)): uf.union(ix[0], ix[k])
        roots = np.array([uf.find(i) for i in range(len(d))])
        _, g = np.unique(roots, return_inverse=True)
        gid[idx] = g + off; off += g.max() + 1
    train_df["group_id"] = gid
    sl = pd.Series(index=train_df.index, dtype=object)
    for src in "AB":
        m = train_df["source"] == src
        b = pd.qcut(train_df.loc[m, "calories"], q=N_CAL_BINS, labels=False, duplicates="drop")
        sl.loc[m] = [f"{src}_{x}" for x in b]
    train_df["strat_label"] = sl
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    train_df["fold"] = -1
    for f, (_, va) in enumerate(sgkf.split(train_df, train_df["strat_label"], train_df["group_id"])):
        train_df.loc[train_df.index[va], "fold"] = f
    train_df[["image_id", "source", "calories", "group_id", "strat_label", "fold"]].to_csv(SHARED_FOLDS_CSV, index=False)
    print(f"Saved folds to {SHARED_FOLDS_CSV}")
assert (train_df["fold"] >= 0).all()

# ------------------------------ smoke subset -----------------------------------
CACHE_SUFFIX = ""
if SMOKE_TEST:
    rng = np.random.default_rng(SEED); keep = []
    for src in "AB":
        ix = train_df.index[train_df["source"] == src].to_numpy()
        keep.append(rng.choice(ix, size=min(SMOKE_PER_SOURCE, len(ix)), replace=False))
    train_df = train_df.loc[np.concatenate(keep)].reset_index(drop=True)
    tkeep = []
    for src in "AB":
        ix = test_df.index[test_df["source"] == src].to_numpy()
        tkeep.append(rng.choice(ix, size=min(max(5, SMOKE_PER_SOURCE // 6), len(ix)), replace=False))
    test_df = test_df.loc[np.sort(np.concatenate(tkeep))].reset_index(drop=True)
    CACHE_SUFFIX = "_smoke"
    print(f"SMOKE subset: train={len(train_df)}, test={len(test_df)}")

# -------- cache build: EXIF-safe decode + PLATE-CROP for B + 448 letterbox ------
def letterbox(img, size, pad=0):
    h, w = img.shape[:2]; s = size / max(h, w)
    nh, nw = max(1, round(h * s)), max(1, round(w * s))
    r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), pad, np.uint8)
    t, l = (size - nh) // 2, (size - nw) // 2
    canvas[t:t + nh, l:l + nw] = r
    return canvas

def plate_crop_B(arr):
    """Crop a Source-B (oblique smartphone, dark tablecloth) photo to the bright
    plate/food region. Union of all large bright components (multi-item meals),
    8% margin, conservative fallbacks to the full frame. Deterministic."""
    h, w = arr.shape[:2]
    s = 512.0 / max(h, w)
    small = cv2.resize(arr, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1: return arr, False
    areas = stats[1:, cv2.CC_STAT_AREA]
    big = np.where(areas >= 0.02 * mask.size)[0] + 1
    if len(big) == 0: return arr, False
    x0 = stats[big, cv2.CC_STAT_LEFT].min(); y0 = stats[big, cv2.CC_STAT_TOP].min()
    x1 = (stats[big, cv2.CC_STAT_LEFT] + stats[big, cv2.CC_STAT_WIDTH]).max()
    y1 = (stats[big, cv2.CC_STAT_TOP] + stats[big, cv2.CC_STAT_HEIGHT]).max()
    frac = (x1 - x0) * (y1 - y0) / float(mask.size)
    if frac < 0.08 or frac > 0.95: return arr, False   # implausible detection -> keep full frame
    mx, my = 0.08 * (x1 - x0), 0.08 * (y1 - y0)
    X0 = max(0, int((x0 - mx) / s)); Y0 = max(0, int((y0 - my) / s))
    X1 = min(w, int((x1 + mx) / s)); Y1 = min(h, int((y1 + my) / s))
    if X1 - X0 < 32 or Y1 - Y0 < 32: return arr, False
    return arr[Y0:Y1, X0:X1], True

def load_image(path, source, size):
    try:
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            arr = np.array(im)
        cropped = False
        if source == "B":
            arr, cropped = plate_crop_B(arr)
        return letterbox(arr, size), cropped
    except Exception as e:
        warnings.warn(f"load failed {path}: {e} -- gray placeholder")
        return np.full((size, size, 3), 128, np.uint8), False

def build_or_load_cache(df, split, size, suffix):
    ap = f"{CACHE_DIR}/{split}_cache_{size}_bcrop{suffix}.npy"
    ip = f"{CACHE_DIR}/{split}_cache_{size}_bcrop{suffix}_index.csv"
    if os.path.exists(ap) and os.path.exists(ip):
        if list(pd.read_csv(ip)["image_id"]) == list(df["image_id"]):
            print(f"[{split}] valid v2 cache found -- skipping rebuild."); return ap
        print(f"[{split}] cache index mismatch -- rebuilding.")
    print(f"[{split}] building {size}px cache (plate-crop for B) for {len(df)} images...")
    t0 = time.time(); arr = np.empty((len(df), size, size, 3), np.uint8); nc = 0
    for i, (p, src) in enumerate(zip(df["path"].values, df["source"].values)):
        arr[i], c = load_image(p, src, size); nc += int(c)
        if (i + 1) % 500 == 0 or i + 1 == len(df):
            print(f"  {i+1}/{len(df)} ({time.time()-t0:.0f}s)")
    nB = int((df["source"] == "B").sum())
    print(f"[{split}] plate-crop applied to {nc}/{nB} B images ({(nc/max(1,nB))*100:.0f}%; rest = full-frame fallback)")
    np.save(ap, arr); df[["image_id"]].to_csv(ip, index=False)
    print(f"[{split}] saved {ap} ({arr.nbytes/1e9:.2f} GB, {time.time()-t0:.0f}s)")
    return ap

train_arr_path = build_or_load_cache(train_df, "train", IMG_SIZE, CACHE_SUFFIX)
test_arr_path  = build_or_load_cache(test_df, "test", IMG_SIZE, CACHE_SUFFIX)

def to_local(src):
    dst = f"{LOCAL_CACHE_DIR}/{os.path.basename(src)}"
    if not (os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(src)):
        shutil.copy(src, dst)
    return dst
train_cache = np.load(to_local(train_arr_path))
test_cache  = np.load(to_local(test_arr_path))
print("caches in RAM:", train_cache.shape, test_cache.shape)

# --------------------- portion-safe augs (same policy as v1) --------------------
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], np.float32)
SRC2IDX = {"A": 0, "B": 1}

def small_rotate(img, max_deg, rng):
    deg = rng.uniform(-max_deg, max_deg); h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

def scale_jitter(img, lo, hi, rng):
    size = img.shape[0]; s = rng.uniform(lo, hi); ns = max(1, round(size * s))
    r = cv2.resize(img, (ns, ns), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    if ns == size: return r
    if ns < size:
        c = np.zeros((size, size, 3), np.uint8); o = (size - ns) // 2
        c[o:o + ns, o:o + ns] = r; return c
    o = (ns - size) // 2; return r[o:o + size, o:o + size]

def color_jitter(img, rng):
    img = img.astype(np.float32)
    img *= rng.uniform(0.85, 1.15)
    mg = img.mean(); img = (img - mg) * rng.uniform(0.85, 1.15) + mg
    g = img.mean(axis=2, keepdims=True); img = (img - g) * rng.uniform(0.9, 1.1) + g
    return np.clip(img, 0, 255).astype(np.uint8)

def augment(img, source, rng):
    if rng.random() < 0.5: img = img[:, ::-1, :]
    if source == "A":
        if rng.random() < 0.5: img = img[::-1, :, :]
        k = int(rng.integers(0, 4))
        if k: img = np.rot90(img, k, axes=(0, 1))
    else:
        img = small_rotate(img, 10.0, rng)
    img = color_jitter(img, rng)
    img = scale_jitter(img, 0.9, 1.1, rng)
    return np.ascontiguousarray(img)

def to_tensor(img):
    img = (img.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(img.transpose(2, 0, 1)).float()

class DS(Dataset):
    def __init__(self, cache, df, train):
        self.cache, self.df, self.train = cache, df.reset_index(drop=True), train
        self.labeled = "calories" in self.df.columns
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        img = self.cache[i]; src = self.df.at[i, "source"]
        if self.train:
            wi = torch.utils.data.get_worker_info()
            rng = np.random.default_rng(((wi.seed if wi else torch.initial_seed()) + i) % 2**32)
            img = augment(img, src, rng)
        x = to_tensor(img); si = torch.tensor(SRC2IDX[src], dtype=torch.long)
        if self.labeled:
            return x, si, torch.tensor(float(self.df.at[i, "calories"]), dtype=torch.float32)
        return x, si, self.df.at[i, "image_id"]

def make_loader(cache, df, train, bs, seed):
    return DataLoader(DS(cache, df, train), batch_size=bs, shuffle=train, drop_last=False,
                      num_workers=NUM_WORKERS, worker_init_fn=seed_worker, generator=make_generator(seed),
                      pin_memory=torch.cuda.is_available(), persistent_workers=NUM_WORKERS > 0)

# ------------------- backbone weights (download once, offline after) -----------
WEIGHTS_PATH = f"{WEIGHTS_DIR}/{BACKBONE.replace('.', '_')}.pth"
if not os.path.exists(WEIGHTS_PATH):
    print(f"Downloading {BACKBONE} weights (one-time)...")
    _m = timm.create_model(BACKBONE, pretrained=True)
    torch.save(_m.state_dict(), WEIGHTS_PATH); del _m
print("backbone weights:", WEIGHTS_PATH)

class CalorieNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(BACKBONE, pretrained=False, num_classes=0,
                                          global_pool="avg", drop_path_rate=DROP_PATH)
        sd = torch.load(WEIGHTS_PATH, map_location="cpu")
        missing, _ = self.backbone.load_state_dict(sd, strict=False)
        bad = [m for m in missing if not m.startswith("head.")]
        assert not bad, f"missing backbone keys: {bad}"
        fd = self.backbone.num_features
        self.source_emb = nn.Embedding(2, SOURCE_EMB_DIM)
        self.head = nn.Sequential(nn.Linear(fd + SOURCE_EMB_DIM, HEAD_HIDDEN), nn.GELU(),
                                  nn.Dropout(HEAD_DROPOUT), nn.Linear(HEAD_HIDDEN, 1))
    def forward(self, x, si):
        f = self.backbone(x)
        return self.head(torch.cat([f, self.source_emb(si)], 1)).squeeze(1)  # log1p(kcal)

class ModelEMA:
    """EMA of weights with ramped decay; validated alongside raw each epoch."""
    def __init__(self, model, decay=EMA_DECAY):
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters(): p.requires_grad_(False)
        self.decay, self.updates = decay, 0
    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        d = min(self.decay, (1 + self.updates) / (10 + self.updates))
        msd = model.state_dict()
        for k, v in self.module.state_dict().items():
            if v.dtype.is_floating_point: v.mul_(d).add_(msd[k].detach(), alpha=1 - d)
            else: v.copy_(msd[k])

def kcal_l1(z, y):   # loss in kcal space -> gradients scale with kcal, upweighting B (metric-matched)
    return (torch.expm1(torch.clamp(z, max=LOG1P_CLAMP_MAX)) - y).abs().mean()

def build_opt(model):
    return torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": BACKBONE_LR},
        {"params": list(model.source_emb.parameters()) + list(model.head.parameters()), "lr": HEAD_LR},
    ], weight_decay=WEIGHT_DECAY)

def build_sched(opt, spe, epochs, warmup):
    total, wu = max(1, spe * epochs), spe * warmup
    def fn(step):
        if wu > 0 and step < wu: return (step + 1) / wu
        pr = min(max((step - wu) / max(1, total - wu), 0.0), 1.0)
        return 0.5 * (1 + math.cos(math.pi * pr))
    return torch.optim.lr_scheduler.LambdaLR(opt, fn)

DEVICE = GPU["device"]
USE_CL = DEVICE.type == "cuda"   # channels_last: big speedup for convnext + AMP

def to_dev(x):
    x = x.to(DEVICE, non_blocking=True)
    return x.contiguous(memory_format=torch.channels_last) if USE_CL and x.dim() == 4 else x

def train_epoch(model, ema, loader, opt, sched, scaler):
    model.train(); tot, n = 0.0, 0
    opt.zero_grad(set_to_none=True)
    for step, (x, si, y) in enumerate(loader):
        x, si, y = to_dev(x), si.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
        with torch.autocast("cuda" if DEVICE.type == "cuda" else "cpu", dtype=GPU["amp_dtype"], enabled=DEVICE.type == "cuda"):
            loss = kcal_l1(model(x, si), y) / ACCUM_STEPS
        (scaler.scale(loss) if scaler else loss).backward()
        if (step + 1) % ACCUM_STEPS == 0 or (step + 1) == len(loader):
            if scaler:
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(opt); scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP); opt.step()
            opt.zero_grad(set_to_none=True); sched.step(); ema.update(model)
        tot += loss.item() * ACCUM_STEPS * x.size(0); n += x.size(0)
    return tot / max(1, n)

@torch.no_grad()
def validate(model, loader):
    model.eval(); P, T, S = [], [], []
    for x, si, y in loader:
        x, sid = to_dev(x), si.to(DEVICE, non_blocking=True)
        with torch.autocast("cuda" if DEVICE.type == "cuda" else "cpu", dtype=GPU["amp_dtype"], enabled=DEVICE.type == "cuda"):
            z = model(x, sid)
        P.append(torch.expm1(torch.clamp(z, max=LOG1P_CLAMP_MAX)).float().cpu().numpy())
        T.append(y.numpy()); S.append(si.numpy())
    P, T, S = map(np.concatenate, (P, T, S))
    per = {s: mae(T[S == i], P[S == i]) for s, i in SRC2IDX.items() if (S == i).any()}
    return mae(T, P), per, P

# ------------------------------ training loop ----------------------------------
oof_records, cv_log = [], []
folds_to_run = sorted(train_df["fold"].unique())[:N_FOLDS_TO_RUN]
print(f"\nTraining folds {folds_to_run} | {BACKBONE} @ {IMG_SIZE}px | {EPOCHS} epochs")

for fold in folds_to_run:
    print(f"\n{'='*60}\nFOLD {fold}\n{'='*60}")
    trm = (train_df["fold"] != fold).values
    tr_df, va_df = train_df[trm].reset_index(drop=True), train_df[~trm].reset_index(drop=True)
    tl = make_loader(train_cache[trm], tr_df, True, GPU["micro_batch"], SEED + fold)
    vl = make_loader(train_cache[~trm], va_df, False, GPU["micro_batch"] * 2, SEED + fold)

    model = CalorieNet().to(DEVICE)
    if USE_CL: model = model.to(memory_format=torch.channels_last)
    ema = ModelEMA(model)
    opt = build_opt(model)
    sched = build_sched(opt, math.ceil(len(tl) / ACCUM_STEPS), EPOCHS, WARMUP_EPOCHS)
    scaler = torch.amp.GradScaler("cuda") if GPU["use_scaler"] else None

    last_p, best_p = f"{CKPT_DIR}/fold{fold}_last.pt", f"{CKPT_DIR}/fold{fold}_best.pt"
    start_ep, best_mae = 0, float("inf")
    if os.path.exists(last_p):
        ck = torch.load(last_p, map_location="cpu", weights_only=False)  # our own checkpoint (RNG states etc.)
        model.load_state_dict(ck["model"]); ema.module.load_state_dict(ck["ema"]); ema.updates = ck["ema_updates"]
        opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        start_ep, best_mae = ck["epoch"] + 1, ck["best_mae"]
        print(f"Resuming fold {fold} at epoch {start_ep} (best={best_mae:.2f})")

    for ep in range(start_ep, EPOCHS):
        t0 = time.time()
        tr_loss = train_epoch(model, ema, tl, opt, sched, scaler)
        v_raw, per_raw, _ = validate(model, vl)
        v_ema, per_ema, _ = validate(ema.module, vl)
        use_ema = v_ema <= v_raw
        v, per = (v_ema, per_ema) if use_ema else (v_raw, per_raw)
        tag = "ema" if use_ema else "raw"
        ps = " ".join(f"MAE_{s}={m:.1f}" for s, m in per.items())
        print(f"  ep {ep+1}/{EPOCHS} | loss={tr_loss:.1f} | val={v:.2f} ({tag}; raw={v_raw:.2f} ema={v_ema:.2f}) | {ps} | {time.time()-t0:.0f}s")
        torch.save({"epoch": ep, "model": model.state_dict(), "ema": ema.module.state_dict(),
                    "ema_updates": ema.updates, "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "best_mae": min(best_mae, v)}, last_p)
        if v < best_mae:
            best_mae = v
            torch.save({"model": (ema.module if use_ema else model).state_dict(), "tag": tag, "val_mae": v}, best_p)
            print(f"    -> new best {v:.2f} ({tag}) saved")
        cv_log.append({"fold": fold, "epoch": ep, "train_loss": tr_loss, "val_mae": v, "tag": tag,
                       **{f"val_mae_{s}": m for s, m in per.items()}})
        pd.DataFrame(cv_log).to_csv(f"{LOG_DIR}/cv_results_v2.csv", index=False)

    bm = CalorieNet().to(DEVICE)
    if USE_CL: bm = bm.to(memory_format=torch.channels_last)
    bm.load_state_dict(torch.load(best_p, map_location="cpu", weights_only=False)["model"])
    _, per_best, oof_pred = validate(bm, vl)
    print(f"  FOLD {fold} BEST -> " + " ".join(f"MAE_{s}={m:.2f} (baseline {BASE_MAE[s]:.0f})" for s, m in per_best.items()))
    fo = va_df[["image_id", "source", "calories"]].copy()
    fo["fold"] = fold; fo["pred_raw"] = oof_pred
    oof_records.append(fo)
    del model, ema, opt, sched, bm
    if torch.cuda.is_available(): torch.cuda.empty_cache()

oof_df = pd.concat(oof_records, ignore_index=True)
oof_df.to_csv(f"{OOF_DIR}/oof_predictions_v2.csv", index=False)

print(f"\n{'='*60}\nOOF SUMMARY (v2)\n{'='*60}")
print(f"overall OOF MAE: {mae(oof_df['calories'], oof_df['pred_raw']):.2f} "
      f"(baseline {(train_df['calories'] - train_df['source'].map(MEDIANS)).abs().mean():.1f})")
for s in "AB":
    g = oof_df[oof_df.source == s]
    if len(g): print(f"  source {s}: OOF MAE {mae(g['calories'], g['pred_raw']):.2f} (baseline {BASE_MAE[s]:.1f})")

# --------- per-source calibration: cross-validated selection (as v1) -----------
def fit_shift(p, t): return {"shift": float(np.median(t - p))}
def app_shift(p, pr): return p + pr["shift"]
def fit_affine(p, t):
    a, b = np.polyfit(p, t, 1); return {"a": float(a), "b": float(b)}
def app_affine(p, pr): return pr["a"] * p + pr["b"]
CAL = {"identity": (lambda p, t: {}, lambda p, pr: p), "shift": (fit_shift, app_shift), "affine": (fit_affine, app_affine)}

selected = {}
for s in "AB":
    so = oof_df[oof_df.source == s]; fp = sorted(so["fold"].unique())
    if len(fp) < 2 or len(so) < 20:
        selected[s] = {"method": "identity", "params": {}, "apply": CAL["identity"][1]}
        print(f"source {s}: too few folds/rows for CV'd calibration -> identity"); continue
    fm = {m: [] for m in CAL}
    for f in fp:
        fit, ev = so[so.fold != f], so[so.fold == f]
        for name, (ffn, afn) in CAL.items():
            pr = ffn(fit["pred_raw"].values, fit["calories"].values)
            fm[name].append(mae(ev["calories"].values, afn(ev["pred_raw"].values, pr)))
    idm, ids = np.mean(fm["identity"]), np.std(fm["identity"])
    best, bm_ = "identity", idm
    for name in ("shift", "affine"):
        cm = np.mean(fm[name])
        if (idm - cm) > ids and cm < bm_: best, bm_ = name, cm
    ffn, afn = CAL[best]
    pr = ffn(so["pred_raw"].values, so["calories"].values)
    selected[s] = {"method": best, "params": pr, "apply": afn}
    print(f"source {s}: calibration='{best}' (identity {idm:.2f}±{ids:.2f} -> {bm_:.2f}) params={pr}")

# --------------- inference: fold ensemble + per-source TTA ----------------------
def tta_variants(name, arr):
    ops = {"identity": lambda a: a, "hflip": lambda a: a[:, :, ::-1, :],
           "rot90": lambda a: np.rot90(a, 1, (1, 2)), "rot90_hflip": lambda a: np.rot90(a, 1, (1, 2))[:, :, ::-1, :],
           "rot180": lambda a: np.rot90(a, 2, (1, 2)), "rot180_hflip": lambda a: np.rot90(a, 2, (1, 2))[:, :, ::-1, :],
           "rot270": lambda a: np.rot90(a, 3, (1, 2)), "rot270_hflip": lambda a: np.rot90(a, 3, (1, 2))[:, :, ::-1, :]}
    return np.ascontiguousarray(ops[name](arr))

@torch.no_grad()
def predict(model, arr, sidx, bs):
    model.eval(); out = np.empty(len(arr), np.float32)
    for st in range(0, len(arr), bs):
        en = min(st + bs, len(arr))
        x = to_dev(torch.stack([to_tensor(im) for im in arr[st:en]]))
        si = torch.tensor(sidx[st:en], dtype=torch.long, device=DEVICE)
        with torch.autocast("cuda" if DEVICE.type == "cuda" else "cpu", dtype=GPU["amp_dtype"], enabled=DEVICE.type == "cuda"):
            z = model(x, si)
        out[st:en] = torch.expm1(torch.clamp(z, max=LOG1P_CLAMP_MAX)).float().cpu().numpy()
    return out

A_TTA = ["identity", "hflip", "rot90", "rot90_hflip", "rot180", "rot180_hflip", "rot270", "rot270_hflip"]
B_TTA = ["identity", "hflip"]
if SMOKE_TEST and not IN_COLAB: A_TTA = B_TTA = ["identity"]

ckpts = sorted(glob.glob(f"{CKPT_DIR}/fold*_best.pt"))
assert len(ckpts) == len(folds_to_run), f"expected {len(folds_to_run)} best ckpts, found {ckpts}"
sidx = test_df["source"].map(SRC2IDX).values
isA = (test_df["source"] == "A").values
acc, cnt = np.zeros(len(test_df)), np.zeros(len(test_df))
for cp in ckpts:
    m = CalorieNet().to(DEVICE)
    if USE_CL: m = m.to(memory_format=torch.channels_last)
    m.load_state_dict(torch.load(cp, map_location="cpu", weights_only=False)["model"])
    for t in sorted(set(A_TTA) | set(B_TTA)):
        mask = np.ones(len(test_df), bool) if (t in A_TTA and t in B_TTA) else (isA if t in A_TTA else ~isA)
        if not mask.any(): continue
        p = predict(m, tta_variants(t, test_cache), sidx, GPU["micro_batch"] * 2)
        acc[mask] += p[mask]; cnt[mask] += 1
    del m
    if torch.cuda.is_available(): torch.cuda.empty_cache()
assert (cnt > 0).all()
raw_pred = acc / cnt
pd.DataFrame({"image_id": test_df["image_id"], "source": test_df["source"], "pred_raw": raw_pred}) \
  .to_csv(f"{OOF_DIR}/test_pred_raw_v2.csv", index=False)   # saved for later v1+v2 blending

# ----------------- calibrate -> clip -> submission ------------------------------
final = raw_pred.copy()
for s, i in SRC2IDX.items():
    msk = sidx == i
    final[msk] = selected[s]["apply"](final[msk], selected[s]["params"])
for s, i in SRC2IDX.items():
    sub = train_df.loc[train_df.source == s, "calories"]
    lo, hi = float(sub.min()), float(sub.quantile(0.995))
    msk = sidx == i
    final[msk] = np.clip(final[msk], lo, hi)
    print(f"clip source {s}: [{lo:.0f}, {hi:.0f}]")

submission = pd.DataFrame({"image_id": test_df["image_id"].values, "predicted_calories": final})
if not SMOKE_TEST:
    order = pd.read_csv(f"{DATA_DIR}/test_ids.csv")["image_id"].tolist()
    submission = submission.set_index("image_id").loc[order].reset_index()
    assert len(submission) == 547 and list(submission["image_id"]) == order
assert list(submission.columns) == ["image_id", "predicted_calories"]
assert submission["predicted_calories"].isna().sum() == 0 and (submission["predicted_calories"] > 0).all()
submission.to_csv(SUBMISSION_PATH, index=False)
print(f"\nWrote {SUBMISSION_PATH} ({len(submission)} rows)")
print("\nArtifacts:")
print(f"  OOF:            {OOF_DIR}/oof_predictions_v2.csv   (same folds.csv as v1 -> blendable)")
print(f"  test raw preds: {OOF_DIR}/test_pred_raw_v2.csv")
print(f"  CV log:         {LOG_DIR}/cv_results_v2.csv")
print("\nBlend recipe (after both runs finish): merge v1+v2 OOF on image_id, grid-search")
print("per-source weight w in [0,1] minimizing OOF MAE of w*v2+(1-w)*v1, keep w only if")
print("it beats the better single model by more than its fold std, apply to test raw preds.")
