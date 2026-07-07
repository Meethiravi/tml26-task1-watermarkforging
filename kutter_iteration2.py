import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import requests
from PIL import Image
from scipy.ndimage import uniform_filter

# CONFIG
ZIP_FILE = "/home/atml_team060/tml26_task4/Dataset.zip"
DATASET_DIR = Path("/home/atml_team060/tml26_task4/Dataset")
TEMP_OUT_DIR = Path("/home/atml_team060/tml26_task4/submission_temp")
FILE_PATH = "/home/atml_team060/tml26_task4/submission.zip"

WIENER_WINDOW = 5    
ALPHA = 0.5       
STRENGTH = 8.0      
 
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
 
# 1. UNZIP IF NEEDED
if not (DATASET_DIR / "watermarked_sources").exists():
    print(f"Unzipping {ZIP_FILE}...")
    with zipfile.ZipFile(str(ZIP_FILE), "r") as zip_ref:
        zip_ref.extractall(str(BASE_DIR))
else:
    print("Dataset already extracted.")
 
TEMP_OUT_DIR.mkdir(exist_ok=True)
 
 
def predict_clean_image(img_channel, window=5):
    """
    Kutter (2000) Section 4: Predict clean image x̂ from watermarked y.
    Uses adaptive Wiener/Lee filter (MAP-estimate, Eq. 13):
        x̂ = mu + (sigma_x^2 / (sigma_x^2 + sigma_n^2)) * (y - mu)
 
    This is a LOCAL PREDICTOR — it uses spatial correlations in natural
    images to predict what the pixel should look like without the watermark.
    The watermark is the HIGH-FREQUENCY residual that the predictor cannot
    explain from the local neighborhood.
    """
    y = img_channel.astype(np.float64)
    local_mean = uniform_filter(y, size=window)
    local_sq_mean = uniform_filter(y ** 2, size=window)
    local_var = np.maximum(local_sq_mean - local_mean ** 2, 0)
    # Global noise variance estimate (Eq. 12 denominator)
    noise_var = np.mean(local_var)
    signal_var = np.maximum(local_var - noise_var, 0)
    # Lee filter gain (Eq. 13)
    gain = signal_var / (signal_var + noise_var + 1e-8)
    x_hat = local_mean + gain * (y - local_mean)
    return x_hat
 
 
def extract_watermark_signal(wm_img_arr, window=5):
    """
    Kutter (2000) Eq. 2: ŵ = y - x̂
 
    KEY INSIGHT: The Wiener filter is used as a HIGH-PASS / PREDICTION filter.
    x̂ is the low-frequency prediction of the clean image content.
    ŵ = y - x̂ is the HIGH-FREQUENCY RESIDUAL — this is where the watermark lives.
 
    This is fundamentally different from simple averaging because:
    - Simple averaging: cancels content by averaging many images
    - Kutter prediction: cancels content per-image using spatial prediction
      so even with 1 image you get a watermark estimate
    """
    wm = wm_img_arr.astype(np.float64)
    w_hat = np.zeros_like(wm)
    for c in range(3):
        x_hat = predict_clean_image(wm[:, :, c], window=window)
        w_hat[:, :, c] = wm[:, :, c] - x_hat  # high-freq residual = watermark
    return w_hat
 
 
def compute_nvf(target_channel, window=5):
    """
    Kutter (2000) Eq. 18 — Noise Visibility Function (non-stationary Gaussian):
        NVF(i,j) = 1 / (1 + sigma_x^2(i,j))
 
    NVF values:
        - Flat regions (low variance)    → NVF ≈ 1  → watermark must be WEAK
        - Textured regions (high variance) → NVF ≈ 0  → watermark can be STRONG
                                                         (HVS masking hides it)
 
    This is the PERCEPTUAL MASKING component — it ensures the inserted
    watermark is invisible to the human visual system, maximizing LPIPS score.
    """
    img = target_channel.astype(np.float64)
    local_mean = uniform_filter(img, size=window)
    local_sq = uniform_filter(img ** 2, size=window)
    local_var = np.maximum(local_sq - local_mean ** 2, 0)
    nvf = 1.0 / (1.0 + local_var)
    return nvf
 
 
def compute_insertion_weight(target_arr, alpha=0.5, window=5):
    """
    Kutter (2000) Eq. 20:
        W(i,j) = [(1 - NVF(i,j)) + NVF(i,j) * (1 - alpha)] * luminance(i,j)
 
    Breaking this down:
        - (1 - NVF):         high in textured areas → put watermark in edges/textures
        - NVF * (1 - alpha): residual watermark in flat areas, scaled by (1-alpha)
        - * luminance:       Weber-Fechner law — brighter pixels can hide more noise
 
    alpha=0: equal weight everywhere (ignoring texture masking)
    alpha=1: watermark only in textured areas (maximum masking, best LPIPS)
    """
    img = target_arr.astype(np.float64)
    # Per-pixel luminance normalized to [0,1]
    lum = img.mean(axis=2) / 255.0 + 1e-8  # (H, W)
 
    W = np.zeros(img.shape[:2], dtype=np.float64)
    for c in range(3):
        nvf_c = compute_nvf(img[:, :, c], window=window)
        W += ((1.0 - nvf_c) + nvf_c * (1.0 - alpha))
    W /= 3.0  # average across channels
    W = W * lum  # apply luminance masking
    return W  # shape: (H, W)
 
 
def insert_watermark_kutter(target_arr, w_hat_agg, alpha=0.5, strength=8.0, window=5):
    """
    Kutter (2000) Eq. 21:
        t̃ = t + delta * W * sign(ŵ)
 
    Only the SIGN of the aggregated watermark is used — not its magnitude.
    The magnitude is entirely controlled by W (perceptual mask) and delta (strength).
 
    This ensures:
    1. The watermark signal direction is correct (from the watermark estimate)
    2. The watermark magnitude is adapted to the TARGET image's local texture
       → invisible in flat areas, hidden in textured areas
    """
    target = target_arr.astype(np.float64)
    t_h, t_w = target.shape[:2]
    w_h, w_w = w_hat_agg.shape[:2]
 
    # Resize aggregated watermark to target size if needed
    if (t_h, t_w) != (w_h, w_w):
        w_img = np.clip(w_hat_agg + 128, 0, 255).astype(np.uint8)
        w_pil = Image.fromarray(w_img).resize((t_w, t_h), Image.BILINEAR)
        w_hat_agg = np.array(w_pil).astype(np.float64) - 128.0
 
    # Compute perceptual insertion weight from TARGET image
    W = compute_insertion_weight(target_arr, alpha=alpha, window=window)  # (H, W)
    W = W[:, :, np.newaxis]  # broadcast to (H, W, 3)
 
    # Eq. 21: insert sign of watermark, scaled by perceptual weight
    sign_w = np.sign(w_hat_agg)  # only direction, not magnitude
    forged = target + strength * W * sign_w
    forged = np.clip(forged, 0, 255).astype(np.uint8)
    return forged
 
 
print("=" * 60)
print("Kutter (2000) True Watermark Copy Attack")
print(f"window={WIENER_WINDOW}, alpha={ALPHA}, strength={STRENGTH}")
print("=" * 60)
 
total_processed = 0
clean_dir = DATASET_DIR / "clean_targets"
 
for source_wm, target_start, target_stop in CATEGORIES:
    print(f"\n[{source_wm}] Predicting watermark from {source_wm} images...")
 
    source_dir = DATASET_DIR / "watermarked_sources" / source_wm
    source_paths = sorted(source_dir.glob("*.png"))
 
    if not source_paths:
        print(f"  [Warning] No source images found in {source_dir}")
        continue
 
    # ----------------------------------------------------------------
    # STEP 1: Per-image watermark prediction via high-pass filtering
    # For each source image: ŵ_i = y_i - Wiener(y_i)
    # Then aggregate by averaging — reduces noise across 25 estimates
    # ----------------------------------------------------------------
    w_hat_list = []
    for p in source_paths:
        img_arr = np.array(Image.open(p).convert("RGB"))
        w_hat_i = extract_watermark_signal(img_arr, window=WIENER_WINDOW)
        w_hat_list.append(w_hat_i)
 
    w_hat_agg = np.mean(w_hat_list, axis=0)
 
    print(f"  Watermark signal stats: "
          f"min={w_hat_agg.min():.3f}, "
          f"max={w_hat_agg.max():.3f}, "
          f"std={w_hat_agg.std():.4f}")
 
    # ----------------------------------------------------------------
    # STEP 2+3: Adapt to target using NVF masking and insert (Eq. 20, 21)
    # The NVF ensures watermark is perceptually invisible in target image
    # ----------------------------------------------------------------
    for number in range(target_start, target_stop + 1):
        target_path = clean_dir / f"{number}.png"
        target_arr = np.array(Image.open(target_path).convert("RGB"))
 
        forged = insert_watermark_kutter(
            target_arr, w_hat_agg,
            alpha=ALPHA, strength=STRENGTH, window=WIENER_WINDOW
        )
 
        Image.fromarray(forged).save(TEMP_OUT_DIR / f"{number}.png")
        total_processed += 1
 
    print(f"  Done: images {target_start}–{target_stop}")
 
print(f"\nTotal forged: {total_processed}/200 images")
if total_processed != 200:
    print("[WARNING] Expected 200 images!")
 
# PACKAGE INTO ZIP
print(f"\nPackaging into {FILE_PATH}...")
with zipfile.ZipFile(FILE_PATH, "w", zipfile.ZIP_DEFLATED) as zipf:
    for img_path in sorted(TEMP_OUT_DIR.glob("*.png")):
        zipf.write(img_path, arcname=img_path.name)
 
print(f"Submission saved to {FILE_PATH}")
 