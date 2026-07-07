import sys
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights

DATASET_DIR  = Path("/home/atml_team060/tml26_task4/Dataset")
TEMP_OUT_DIR = Path("/home/atml_team060/tml26_task4/submission_temp_pref")
FILE_PATH    = "/home/atml_team060/tml26_task4/submission_pref.zip"
SOURCE_ROOT  = DATASET_DIR / "watermarked_sources"
TARGET_DIR   = DATASET_DIR / "clean_targets"

CATEGORIES = [
    ("WM_1", 1, 25),   ("WM_2", 26, 50),  ("WM_3", 51, 75),  ("WM_4", 76, 100),
    ("WM_5", 101, 125),("WM_6", 126, 150),("WM_7", 151, 175),("WM_8", 176, 200),
]

BATCHES = [wm for wm, _, _ in CATEGORIES]

WARM_START_BATCHES = {"WM_3", "WM_4", "WM_5"}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- PGD ---
PGD_STEPS   = 40
PGD_STEP_SZ = 2.0 / 255.0     # L_inf step, images normalized to [0,1]

# --- adaptive eps calibration ---
EPS_MIN               = 8.0 / 255.0
EPS_MAX               = 128.0 / 255.0
EPS_GROWTH            = 1.6
EPS_CALIBRATION_SAMPLES = 5

# --- LPIPS-budget blend search  ---
LPIPS_BUDGET       = 0.045
BLEND_ITERS        = 12
CALIBRATION_SAMPLES = 5

# --- feature layers used for the preference direction (multi-scale) ---
FEATURE_LAYERS = ["layer2", "layer3", "layer4"]


# ----------------------------------------------------------------------
# Frozen feature extractor
# ----------------------------------------------------------------------
class MultiLayerFeatures(torch.nn.Module):
    def __init__(self, layers):
        super().__init__()
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(DEVICE).eval()
        for p in backbone.parameters():
            p.requires_grad_(False)
        self.backbone = backbone
        self.layers = layers
        self._acts = {}
        for name in layers:
            getattr(backbone, name).register_forward_hook(self._make_hook(name))
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.to(DEVICE)

    def _make_hook(self, name):
        def hook(_module, _inp, out):
            self._acts[name] = out
        return hook

    def forward(self, x01):
        """x01: (B,3,H,W) in [0,1]. Returns concatenated global-avg-pooled
        features from self.layers, L2-normalized per layer before concat."""
        x = (x01 - self.mean) / self.std
        self._acts = {}
        self.backbone(x)
        feats = []
        for name in self.layers:
            f = self._acts[name]
            f = F.adaptive_avg_pool2d(f, 1).flatten(1)
            f = F.normalize(f, dim=1)
            feats.append(f)
        return torch.cat(feats, dim=1)


_extractor = None
def extractor():
    global _extractor
    if _extractor is None:
        _extractor = MultiLayerFeatures(FEATURE_LAYERS)
    return _extractor


# ----------------------------------------------------------------------
# Optional BM3D warm-start 
# ----------------------------------------------------------------------
try:
    from forgev4 import estimate as _bm3d_estimate, resize_to as _bm3d_resize_to, ALPHA as _BM3D_ALPHA
    _BM3D_IMPORT_OK = True
except Exception as e:
    _BM3D_IMPORT_OK = False
    print(f"[info] BM3D warm-start disabled, forgev4 unavailable ({e}); "
          f"PGD will start from the clean image for all batches.")

_bm3d_residual_cache = {}


def warm_init_fn(wm):
 
    if wm not in WARM_START_BATCHES or not _BM3D_IMPORT_OK:
        return lambda _clean01: None

    def fn(clean01):
        if wm not in _bm3d_residual_cache:
            try:
                _bm3d_residual_cache[wm] = _bm3d_estimate(wm)
            except Exception as e:
                print(f"  [BM3D warm-start unavailable for {wm}: {e}]")
                _bm3d_residual_cache[wm] = None
        w_est = _bm3d_residual_cache[wm]
        if w_est is None:
            return None
        h, w = clean01.shape[1], clean01.shape[2]
        w_resized = _bm3d_resize_to(w_est, (h, w))
        clean255 = clean01.permute(1, 2, 0).numpy() * 255.0
        warm255 = np.clip(clean255 + _BM3D_ALPHA[wm] * w_resized, 0, 255)
        return torch.from_numpy((warm255 / 255.0).astype(np.float32)).permute(2, 0, 1)

    return fn


# ----------------------------------------------------------------------
# IO helpers
# ----------------------------------------------------------------------
def load01(p, size=None):
    img = Image.open(p).convert("RGB")
    if size is not None:
        img = img.resize(size, Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)  # (3,H,W)


def to_uint8(x01):
    arr = (x01.clamp(0, 1) * 255.0).round().byte().permute(1, 2, 0).cpu().numpy()
    return arr


# ----------------------------------------------------------------------
# Preference direction per batch
# ----------------------------------------------------------------------
_clean_paths_cache = None
def all_clean_paths():
    global _clean_paths_cache
    if _clean_paths_cache is None:
        _clean_paths_cache = sorted(TARGET_DIR.glob("*.png"), key=lambda p: int(p.stem))
    return _clean_paths_cache


@torch.no_grad()
def batch_features(paths, size):
    ext = extractor()
    feats = []
    for p in paths:
        x = load01(p, size=size).unsqueeze(0).to(DEVICE)
        feats.append(ext(x).cpu())
    return torch.cat(feats, dim=0)


def preference_direction(wm_name, lo, hi):
    """direction (D,) unit vector, and the working (w,h) size used for this
    batch's feature extraction (native size of its sources)."""
    src_paths = sorted((SOURCE_ROOT / wm_name).glob("*.png"))
    with Image.open(src_paths[0]) as im:
        size = im.size  # (w, h)

    pos_feats = batch_features(src_paths, size)

    neg_paths = [p for p in all_clean_paths()
                 if not (lo <= int(p.stem) <= hi)]
    # Cap negative pool for speed; a random-ish spread across ids is fine.
    neg_paths = neg_paths[::max(1, len(neg_paths) // 100)]
    neg_feats = batch_features(neg_paths, size)

    diff = pos_feats.mean(0) - neg_feats.mean(0)
    pooled_std = torch.sqrt(
        (pos_feats.var(0, unbiased=True) + neg_feats.var(0, unbiased=True)) / 2.0 + 1e-6)
    direction = diff / pooled_std
    direction = F.normalize(direction, dim=0)
    return direction.to(DEVICE), size


# ----------------------------------------------------------------------
# PGD toward the preference direction
# ----------------------------------------------------------------------
def pgd_forge(clean01, direction, steps=PGD_STEPS, step_sz=PGD_STEP_SZ, eps=EPS_MIN, init01=None):
    ext = extractor()
    x0 = clean01.unsqueeze(0).to(DEVICE)
    if init01 is not None:
        x = init01.unsqueeze(0).to(DEVICE)
        x = x0 + torch.clamp(x - x0, -eps, eps)   # keep the warm start inside this eps ball
        x = x.clamp(0.0, 1.0)
    else:
        x = x0.clone()
    x = x.requires_grad_(True)
    for _ in range(steps):
        feat = ext(x)
        score = (feat * direction.unsqueeze(0)).sum(dim=1).mean()
        grad = torch.autograd.grad(score, x)[0]
        with torch.no_grad():
            x = x + step_sz * grad.sign()
            x = x0 + torch.clamp(x - x0, -eps, eps)
            x = x.clamp(0.0, 1.0)
        x = x.detach().requires_grad_(True)
    return x.detach().squeeze(0).cpu()


def steps_for_eps(eps, step_sz=PGD_STEP_SZ, base_steps=PGD_STEPS):
    """Enough sign-step iterations to actually reach the eps boundary (a
    larger ball needs more steps to walk to its edge), plus margin."""
    return max(base_steps, int(np.ceil(eps / step_sz * 1.5)))


def calibrate_eps(direction, sample_paths, budget=LPIPS_BUDGET,
                   eps_min=EPS_MIN, eps_max=EPS_MAX, growth=EPS_GROWTH, init_fn=None):

    eps = eps_min
    while eps <= eps_max:
        steps = steps_for_eps(eps)
        vals = []
        for p in sample_paths:
            clean01 = load01(p)
            init01 = init_fn(clean01) if init_fn else None
            adv01 = pgd_forge(clean01, direction, steps=steps, eps=eps, init01=init01)
            vals.append(lpips_dist(clean01, adv01))
        if float(np.mean(vals)) >= budget:
            return eps
        eps *= growth
    return eps_max


# ----------------------------------------------------------------------
# LPIPS-budget blend search (mirrors forgev4 / task_template calibration)
# ----------------------------------------------------------------------
_lpips_model = None
def lpips_model():
    global _lpips_model
    if _lpips_model is None:
        import lpips as lpips_lib
        _lpips_model = lpips_lib.LPIPS(net="alex").to(DEVICE).eval()
    return _lpips_model


def lpips_dist(a01, b01):
    to_t = lambda t: (t.unsqueeze(0).to(DEVICE) * 2.0 - 1.0)
    with torch.no_grad():
        return lpips_model()(to_t(a01), to_t(b01)).item()


def blend_to_budget(clean01, adv01, budget=LPIPS_BUDGET, iters=BLEND_ITERS):
    """Binary-search t in [0,1] s.t. forged = clean + t*(adv-clean) has
    LPIPS(clean, forged) ~= budget. Returns forged01."""
    lo, hi = 0.0, 1.0
    best = clean01
    for _ in range(iters):
        mid = (lo + hi) / 2.0
        cand = clean01 + mid * (adv01 - clean01)
        d = lpips_dist(clean01, cand)
        if d > budget:
            hi = mid
        else:
            lo = mid
            best = cand
    return best


# ----------------------------------------------------------------------
# Build
# ----------------------------------------------------------------------
def build(verbose=True):
    TEMP_OUT_DIR.mkdir(parents=True, exist_ok=True)
    n = 0
    for wm, lo, hi in CATEGORIES:
        if wm not in BATCHES:
            continue
        direction, size = preference_direction(wm, lo, hi)
        init_fn = warm_init_fn(wm)

        sample_paths = [TARGET_DIR / f"{i}.png"
                        for i in range(lo, min(hi, lo + EPS_CALIBRATION_SAMPLES - 1) + 1)]
        eps = calibrate_eps(direction, sample_paths, init_fn=init_fn)
        steps = steps_for_eps(eps)
        warm_tag = " [BM3D warm-start]" if wm in WARM_START_BATCHES and _BM3D_IMPORT_OK else ""
        if verbose:
            print(f"[{wm}] preference direction ready (dim={direction.shape[0]}, size={size})"
                  f"  calibrated eps={eps*255:.1f}/255  steps={steps}{warm_tag}")

        for num in range(lo, hi + 1):
            p = TARGET_DIR / f"{num}.png"
            clean01 = load01(p)
            init01 = init_fn(clean01)
            adv01 = pgd_forge(clean01, direction, steps=steps, eps=eps, init01=init01)
            forged01 = blend_to_budget(clean01, adv01)
            Image.fromarray(to_uint8(forged01)).save(TEMP_OUT_DIR / f"{num}.png")
            n += 1
        if verbose:
            print(f"  done: images {lo}-{hi}")

    if n != 200 and set(BATCHES) == {wm for wm, _, _ in CATEGORIES}:
        print(f"[WARNING] processed {n} images, expected 200")

    with zipfile.ZipFile(FILE_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        for img in sorted(TEMP_OUT_DIR.glob("*.png"), key=lambda x: int(x.stem)):
            zf.write(img, arcname=img.name)
    print(f"Saved {n} -> {FILE_PATH}")


def eval_lpips():
    print(f"{'batch':6} {'LPIPS':>9} {'S_qlt':>8}")
    for wm, lo, hi in CATEGORIES:
        if wm not in BATCHES:
            continue
        vals = []
        for num in range(lo, hi + 1):
            c = load01(TARGET_DIR / f"{num}.png")
            f = load01(TEMP_OUT_DIR / f"{num}.png")
            vals.append(lpips_dist(c, f))
        m = float(np.mean(vals))
        print(f"{wm:6} {m:9.4f} {np.exp(-8*m):8.4f}")


def sweep():

    for wm, lo, hi in CATEGORIES:
        if wm not in BATCHES:
            continue
        direction, _ = preference_direction(wm, lo, hi)
        init_fn = warm_init_fn(wm)
        nums = list(range(lo, hi + 1))[:CALIBRATION_SAMPLES]
        sample_paths = [TARGET_DIR / f"{i}.png" for i in nums]

        eps = calibrate_eps(direction, sample_paths, init_fn=init_fn)
        base_steps = steps_for_eps(eps)
        capped = " [EPS_MAX REACHED -- still short of budget, raise EPS_MAX]" if eps >= EPS_MAX else ""
        warm_tag = " [BM3D warm-start]" if wm in WARM_START_BATCHES and _BM3D_IMPORT_OK else ""
        print(f"\n[{wm}]  calibrated eps={eps*255:.1f}/255  base_steps={base_steps}{capped}{warm_tag}")

        for mult in (0.5, 1.0, 2.0):
            steps = max(1, int(round(base_steps * mult)))
            deltas = []
            for num in nums:
                clean01 = load01(TARGET_DIR / f"{num}.png")
                init01 = init_fn(clean01)
                adv01 = pgd_forge(clean01, direction, steps=steps, eps=eps, init01=init01)
                forged01 = blend_to_budget(clean01, adv01)
                deltas.append(lpips_dist(clean01, forged01))
            print(f"  steps={steps:<4} mean_LPIPS_after_blend={np.mean(deltas):.4f}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "build"
    if mode == "sweep":
        sweep()
    elif mode == "eval":
        build()
        eval_lpips()
    else:
        build()
