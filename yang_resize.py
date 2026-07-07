"""
Watermark Forgery Attack — merged pipeline (v2)
================================================

Combines:
  A) Yang et al. "mean-difference" estimator:
         raw_pattern = mean(watermarked_25) - mean(clean_200)
     This removes low-frequency CONTENT better than a scalar DC offset,
     because it uses a real estimate of "what an average natural image
     looks like" rather than just each channel's global mean.

  B) High-pass cleanup of that raw_pattern:
     Because the 25-image mean and 200-image mean have different sampling
     noise floors, raw_pattern still contains residual low-frequency
     "content mismatch" noise that is NOT the watermark. We suppress it
     with a mild high-pass (keeps the wm signal if it's spread-spectrum /
     high-frequency, which is the common case for additive watermarks).
     Set KEEP_LOWFREQ_FRACTION > 0 to retain some low-freq energy in case
     the true watermark has low-frequency components (tune per batch).

  C) NVF perceptual masking (Kutter 2000): scale injection by local
     texture of the TARGET image, so watermark hides in busy regions and
     stays weak in flat regions.

  D) LPIPS-calibrated strength per batch (binary search), so quality
     budget is spent close to your target ceiling instead of guessed.

Usage:
    python3 forge_watermarks_v2.py
"""

import os
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, uniform_filter

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
ZIP_FILE      = "/home/atml_team060/tml26_task4/Dataset.zip"
DATASET_DIR   = Path("/home/atml_team060/tml26_task4/Dataset")
TEMP_OUT_DIR  = Path("/home/atml_team060/tml26_task4/submission_yang_iter3")
FILE_PATH     = "/home/atml_team060/tml26_task4/submission_yang_iter3.zip"

CATEGORIES = [
    ("WM_1",  1,   25),
    ("WM_2",  26,  50),
    ("WM_3",  51,  75),
    ("WM_4",  76,  100),
    ("WM_5",  101, 125),
    ("WM_6",  126, 150),
    ("WM_7",  151, 175),
    ("WM_8",  176, 200),
]

# --- Watermark estimation ---
USE_MEAN_DIFF          = True   # (A) subtract clean_mean instead of scalar DC
USE_HIGHPASS_CLEANUP   = True   # (B) high-pass the raw pattern to kill residual content noise
HIGHPASS_SIGMA         = 3.0    # bigger sigma = removes MORE low-freq (be less aggressive than pure highpass extraction, since mean-diff already killed most content)
KEEP_LOWFREQ_FRACTION  = 0.15   # 0=pure highpass, 1=no highpass at all; blend factor

# --- Perceptual masking (Kutter NVF) ---
NVF_WINDOW      = 5
MASK_BASELINE   = 0.35    # min fraction of strength applied even in flat areas
USE_MASK        = True

# --- Strength calibration ---
LPIPS_BUDGET         = 0.045
MIN_STRENGTH         = 0.25
MAX_STRENGTH         = 25.0
CALIBRATION_SAMPLES  = 5
CALIBRATION_ITERS    = 14
FALLBACK_STRENGTH    = 3.0   # used only if lpips package unavailable

# ----------------------------------------------------------------------
# Optional LPIPS setup
# ----------------------------------------------------------------------
_LPIPS_AVAILABLE = False
try:
    import torch
    import lpips as lpips_lib
    _lpips_model = lpips_lib.LPIPS(net='alex')
    _lpips_model.eval()
    _LPIPS_AVAILABLE = True
    print("[info] lpips found — will auto-calibrate strength per batch.")
except Exception as e:
    print(f"[info] lpips unavailable ({e}); using FALLBACK_STRENGTH={FALLBACK_STRENGTH}.")


def lpips_distance(clean_arr: np.ndarray, forged_arr: np.ndarray) -> float:
    def to_tensor(a):
        t = torch.from_numpy(a).float().permute(2, 0, 1) / 127.5 - 1.0
        return t.unsqueeze(0)
    with torch.no_grad():
        return _lpips_model(to_tensor(clean_arr), to_tensor(forged_arr)).item()


# ----------------------------------------------------------------------
# 1. Unzip dataset if needed
# ----------------------------------------------------------------------
if not (DATASET_DIR / "watermarked_sources").exists():
    if not os.path.exists(ZIP_FILE):
        raise FileNotFoundError(f"Could not find {ZIP_FILE}.")
    print(f"Unzipping {ZIP_FILE}...")
    with zipfile.ZipFile(ZIP_FILE, "r") as zip_ref:
        zip_ref.extractall(DATASET_DIR)
else:
    print("Dataset already extracted.")

TEMP_OUT_DIR.mkdir(parents=True, exist_ok=True)
clean_dir = DATASET_DIR / "clean_targets"


# ----------------------------------------------------------------------
# 2. Load the 200-image clean pool once (used for mean-diff estimator)
# ----------------------------------------------------------------------
print("\nLoading clean reference pool...")
clean_paths_all = sorted(clean_dir.glob("*.png"))
clean_pool = [np.array(Image.open(p).convert("RGB")).astype(np.float32) for p in clean_paths_all]
print(f"Loaded {len(clean_pool)} clean images.")


def resize_to(arr: np.ndarray, h: int, w: int) -> np.ndarray:
    if arr.shape[:2] == (h, w):
        return arr
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
    return np.array(img).astype(np.float32)


def resize_signed(arr: np.ndarray, h: int, w: int) -> np.ndarray:
    """Resize a zero-centered signed residual (values can be negative)."""
    if arr.shape[:2] == (h, w):
        return arr
    img = Image.fromarray(np.clip(arr + 128, 0, 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
    return np.array(img).astype(np.float64) - 128.0


# ----------------------------------------------------------------------
# 3. Watermark pattern estimation (A + B)
# ----------------------------------------------------------------------
# RESIZE_METHOD controls how source images get resized to the batch's
# target resolution when they differ. LANCZOS preserves higher-frequency
# detail better than BILINEAR, which matters when the watermark itself is
# high-frequency (common for additive/spread-spectrum watermarks). If your
# watermarking scheme is known/suspected to be low-frequency (e.g. large
# block patterns), BILINEAR may be safer/less prone to ringing artifacts.
RESIZE_METHOD = Image.LANCZOS


def estimate_pattern(source_paths, clean_pool, target_h, target_w):
    """
    target_h, target_w: the NATIVE resolution of the clean target images
    this batch will be applied to. If the source watermarked images are a
    different resolution, they are resized to (target_h, target_w) BEFORE
    extraction/averaging -- not after -- so the mean-diff / high-pass
    computation happens at the resolution we'll actually inject into.
    Resizing an already-finalized pattern (post-hoc) risks warping the
    watermark's frequency content via interpolation, especially on 2x
    upsamples (e.g. 256->512) which can smooth out fine structure a
    detector relies on.
    """
    wm_stack = []
    src_native_size = None
    for p in source_paths:
        img = Image.open(p).convert("RGB")
        if src_native_size is None:
            src_native_size = img.size  # (W, H)
        if img.size != (target_w, target_h):
            img = img.resize((target_w, target_h), RESIZE_METHOD)
        wm_stack.append(np.array(img).astype(np.float32))
    wm_mean = np.mean(wm_stack, axis=0)
    s_h, s_w = wm_mean.shape[:2]  # now == (target_h, target_w)

    if src_native_size != (target_w, target_h):
        print(f"  [note] resized sources from {src_native_size} to "
              f"{(target_w, target_h)} before extraction (mismatched resolution)")

    if USE_MEAN_DIFF:
        # Only use clean-pool images that are ALSO this resolution for the
        # mean-diff estimator, to avoid diluting it with resized content
        # from a different image population. If none match, fall back to
        # resizing the whole pool (less ideal but better than nothing).
        same_res_clean = [c for c in clean_pool if c.shape[:2] == (target_h, target_w)]
        pool_to_use = same_res_clean if same_res_clean else clean_pool
        clean_resized = [resize_to(c, s_h, s_w) for c in pool_to_use]
        clean_mean = np.mean(clean_resized, axis=0)
        raw_pattern = wm_mean - clean_mean
    else:
        dc = wm_mean.mean(axis=(0, 1), keepdims=True)
        raw_pattern = wm_mean - dc

    if USE_HIGHPASS_CLEANUP:
        blurred = np.stack(
            [gaussian_filter(raw_pattern[:, :, c], sigma=HIGHPASS_SIGMA) for c in range(3)],
            axis=2,
        )
        highpassed = raw_pattern - blurred
        pattern = highpassed + KEEP_LOWFREQ_FRACTION * blurred
    else:
        pattern = raw_pattern

    return pattern.astype(np.float64)


# ----------------------------------------------------------------------
# 4. NVF perceptual mask
# ----------------------------------------------------------------------
def nvf_mask(target_arr, window=5):
    img = target_arr.astype(np.float64)
    acc = np.zeros(img.shape[:2])
    for c in range(3):
        m = uniform_filter(img[:, :, c], size=window)
        sq = uniform_filter(img[:, :, c] ** 2, size=window)
        var = np.maximum(sq - m ** 2, 0)
        acc += 1.0 / (1.0 + var)
    return acc / 3.0


def injection_weight(target_arr, baseline, window):
    nvf = nvf_mask(target_arr, window=window)              # 1=flat, 0=textured
    weight = baseline + (1.0 - baseline) * (1.0 - nvf)      # flat->baseline, textured->1
    return weight[:, :, None]


# ----------------------------------------------------------------------
# 5. Forging + calibration
# ----------------------------------------------------------------------
def forge(target_arr, pattern, strength, use_mask, baseline, window):
    t = target_arr.astype(np.float64)
    h, w = t.shape[:2]
    p = resize_signed(pattern, h, w)
    if use_mask:
        weight = injection_weight(target_arr, baseline, window)
        forged = t + strength * weight * p
    else:
        forged = t + strength * p
    return np.clip(forged, 0, 255).astype(np.uint8)


def calibrate_strength(sample_targets, pattern, budget):
    lo, hi = MIN_STRENGTH, MAX_STRENGTH
    best = lo

    def avg_lpips_at(strength):
        dists = []
        for arr in sample_targets:
            forged = forge(arr, pattern, strength, USE_MASK, MASK_BASELINE, NVF_WINDOW)
            dists.append(lpips_distance(arr, forged))
        return float(np.mean(dists))

    for _ in range(CALIBRATION_ITERS):
        mid = (lo + hi) / 2.0
        d = avg_lpips_at(mid)
        if d > budget:
            hi = mid
        else:
            best = mid
            lo = mid
    return best


# ----------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------
print("=" * 70)
print("Watermark forgery v2 — mean-diff + highpass cleanup + NVF mask")
print(f"USE_MEAN_DIFF={USE_MEAN_DIFF}  USE_HIGHPASS_CLEANUP={USE_HIGHPASS_CLEANUP} "
      f"HIGHPASS_SIGMA={HIGHPASS_SIGMA}  LPIPS_BUDGET={LPIPS_BUDGET}")
print("=" * 70)

total_processed = 0

for source_wm, target_start, target_stop in CATEGORIES:
    print(f"\n[{source_wm}] estimating pattern...")
    source_dir = DATASET_DIR / "watermarked_sources" / source_wm
    source_paths = sorted(source_dir.glob("*.png"))
    if not source_paths:
        print(f"  [WARNING] no source images found in {source_dir}, skipping")
        continue

    # Detect this batch's native target resolution from its first target
    # image. NOTE: this assumes all targets within one WM_i batch share the
    # same resolution (true per your batch mapping: e.g. all of 101-125 are
    # 128x128, all of 151-200 are 512x512). If that assumption breaks for
    # some batch, this script would need per-image pattern resizing again --
    # check target_h/target_w prints below if results look off.
    first_target = Image.open(clean_dir / f"{target_start}.png")
    target_w, target_h = first_target.size
    print(f"  batch target resolution: {target_w}x{target_h}")

    pattern = estimate_pattern(source_paths, clean_pool, target_h, target_w)
    print(f"  pattern stats: min={pattern.min():.2f} max={pattern.max():.2f} "
          f"std={pattern.std():.4f}  per-channel std={tuple(round(s,4) for s in pattern.std(axis=(0,1)))}")

    target_ids = list(range(target_start, target_stop + 1))
    if _LPIPS_AVAILABLE:
        sample_ids = target_ids[:CALIBRATION_SAMPLES]
        sample_arrs = [np.array(Image.open(clean_dir / f"{i}.png").convert("RGB")) for i in sample_ids]
        strength = calibrate_strength(sample_arrs, pattern, LPIPS_BUDGET)
        print(f"  calibrated strength = {strength:.3f} (LPIPS budget {LPIPS_BUDGET})")
    else:
        strength = FALLBACK_STRENGTH
        print(f"  using fallback strength = {strength}")

    for number in target_ids:
        target_arr = np.array(Image.open(clean_dir / f"{number}.png").convert("RGB"))
        forged = forge(target_arr, pattern, strength, USE_MASK, MASK_BASELINE, NVF_WINDOW)
        Image.fromarray(forged).save(TEMP_OUT_DIR / f"{number}.png")
        total_processed += 1

    print(f"  done: images {target_start}-{target_stop}")

print(f"\nTotal forged: {total_processed}/200 images")
if total_processed != 200:
    print("[WARNING] Expected 200 images!")

print(f"\nPackaging into {FILE_PATH}...")
with zipfile.ZipFile(FILE_PATH, "w", zipfile.ZIP_DEFLATED) as zipf:
    for img_path in sorted(TEMP_OUT_DIR.glob("*.png")):
        zipf.write(img_path, arcname=img_path.name)

print(f"Submission saved to {FILE_PATH}")