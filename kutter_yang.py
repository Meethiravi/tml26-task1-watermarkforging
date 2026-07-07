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

if not (DATASET_DIR / "watermarked_sources").exists():
    print(f"Unzipping {ZIP_FILE}...")
    with zipfile.ZipFile(str(ZIP_FILE), "r") as zip_ref:
        zip_ref.extractall(str(DATASET_DIR))
else:
    print("Dataset already extracted.")
 
TEMP_OUT_DIR.mkdir(exist_ok=True)

def wiener_predict(channel, window=5):
    """
    Kutter (2000) Eq. 13 — Lee/Wiener filter as spatial predictor.
    Predicts the clean image content from local neighborhood statistics.
    Returns x̂: the predicted clean channel.
    """
    y = channel.astype(np.float64)
    local_mean = uniform_filter(y, size=window)
    local_sq_mean = uniform_filter(y ** 2, size=window)
    local_var = np.maximum(local_sq_mean - local_mean ** 2, 0)
    noise_var = np.mean(local_var)
    signal_var = np.maximum(local_var - noise_var, 0)
    gain = signal_var / (signal_var + noise_var + 1e-8)
    x_hat = local_mean + gain * (y - local_mean)
    return x_hat
 
 
def extract_per_image_residual(wm_arr, window=5):
    """
    Kutter (2000) Eq. 2 — Per-image watermark prediction:
        ŵ_i = y_i - x̂_i
 
    High-frequency residual after spatial prediction.
    Works per image — doesn't require multiple images to cancel content.
    """
    wm = wm_arr.astype(np.float64)
    w_hat = np.zeros_like(wm)
    for c in range(3):
        x_hat = wiener_predict(wm[:, :, c], window=window)
        w_hat[:, :, c] = wm[:, :, c] - x_hat
    return w_hat
 
 
def compute_nvf(target_arr, window=5):
    """
    Kutter (2000) Eq. 18 — Noise Visibility Function:
        NVF(i,j) = 1 / (1 + sigma^2(i,j))
 
    Near 1 in flat regions (watermark must be weak).
    Near 0 in textured regions (watermark can be stronger — HVS masking).
    """
    img = target_arr.astype(np.float64)
    nvf = np.zeros(img.shape[:2], dtype=np.float64)
    for c in range(3):
        local_mean = uniform_filter(img[:, :, c], size=window)
        local_sq = uniform_filter(img[:, :, c] ** 2, size=window)
        local_var = np.maximum(local_sq - local_mean ** 2, 0)
        nvf += 1.0 / (1.0 + local_var)
    nvf /= 3.0
    return nvf
 
 
def compute_weight(target_arr, alpha=0.5, window=5):
    """
    Kutter (2000) Eq. 20:
        W = ((1 - NVF) + NVF * (1 - alpha)) * luminance
 
    Combines texture masking (NVF) with luminance masking (Weber-Fechner law).
    Higher W = more watermark energy allowed at that pixel.
    """
    nvf = compute_nvf(target_arr, window=window)
    lum = target_arr.astype(np.float64).mean(axis=2) / 255.0 + 1e-8
    W = ((1.0 - nvf) + nvf * (1.0 - alpha)) * lum
    return W
 
 
def resize_pattern(pattern, target_h, target_w):
    """Resize watermark pattern to target image dimensions if needed."""
    p_h, p_w = pattern.shape[:2]
    if (p_h, p_w) == (target_h, target_w):
        return pattern
    p_img = Image.fromarray(np.clip(pattern + 128, 0, 255).astype(np.uint8))
    p_img = p_img.resize((target_w, target_h), Image.BILINEAR)
    return np.array(p_img).astype(np.float64) - 128.0
 
 
# 2. LOAD CLEAN POOL FOR YANG SUBTRACTION
# Yang et al. (2024): subtract mean(clean images) instead of DC offset.
# Better cancels residual image content that 25-image averaging leaves behind.
print("Loading clean reference pool (Yang et al.)...")
clean_dir = DATASET_DIR / "clean_targets"
clean_pool = []
for p in sorted(clean_dir.glob("*.png")):
    clean_pool.append(np.array(Image.open(p).convert("RGB")).astype(np.float32))
print(f"Loaded {len(clean_pool)} clean reference images.")
 
print("\n" + "=" * 60)
print("Yang (2024) + Kutter (2000) Combined Attack")
print(f"window={WIENER_WINDOW}, alpha={ALPHA}, strength={STRENGTH}")
print("=" * 60)
 
total_processed = 0
 
for source_wm, target_start, target_stop in CATEGORIES:
    print(f"\n[{source_wm}] Processing -> images {target_start} to {target_stop}...")
 
    source_dir = DATASET_DIR / "watermarked_sources" / source_wm
    source_paths = sorted(source_dir.glob("*.png"))
 
    if not source_paths:
        print(f"  [Warning] No source images found in {source_dir}")
        continue
 
    s_h, s_w = np.array(Image.open(source_paths[0])).shape[:2]
 
    # ----------------------------------------------------------------
    # STEP 1 — KUTTER: Per-image residual extraction
    # ŵ_i = y_i - Wiener(y_i)  for each of the 25 source images
    # Isolates watermark as high-frequency prediction residual per image.
    # ----------------------------------------------------------------
    residuals = []
    for p in source_paths:
        img_arr = np.array(Image.open(p).convert("RGB"))
        w_hat_i = extract_per_image_residual(img_arr, window=WIENER_WINDOW)
        residuals.append(w_hat_i)
 
    mean_residual = np.mean(residuals, axis=0)  # average Kutter residuals
 
    # ----------------------------------------------------------------
    # STEP 2 — YANG: Subtract clean image mean to remove content bias
    # Yang et al. (2024): watermark = mean(WM residuals) - mean(clean residuals)
    # The clean pool residuals represent the "average content high-frequencies"
    # that leak through the Wiener filter — subtracting them purifies the signal.
    # ----------------------------------------------------------------
    clean_residuals = []
    for c_arr in clean_pool:
        c_resized = resize_pattern(
            np.zeros_like(c_arr, dtype=np.float64),  # dummy — just need shape
            s_h, s_w
        )
        # Actually extract residual from clean image
        c_arr_f = c_arr.astype(np.float32)
        c_h, c_w = c_arr_f.shape[:2]
        if (c_h, c_w) != (s_h, s_w):
            c_pil = Image.fromarray(c_arr_f.astype(np.uint8)).resize((s_w, s_h), Image.BILINEAR)
            c_arr_f = np.array(c_pil).astype(np.float32)
        clean_residuals.append(extract_per_image_residual(c_arr_f, window=WIENER_WINDOW))
 
    mean_clean_residual = np.mean(clean_residuals, axis=0)
 
    # Final watermark pattern: Kutter residual minus Yang clean bias
    watermark_pattern = mean_residual - mean_clean_residual
 
    print(f"  Watermark pattern stats: "
          f"min={watermark_pattern.min():.3f}, "
          f"max={watermark_pattern.max():.3f}, "
          f"std={watermark_pattern.std():.4f}")
 
    # ----------------------------------------------------------------
    # STEP 3 — KUTTER: NVF perceptual masking insertion (Eq. 20, 21)
    # t̃ = t + delta * W * sign(ŵ)
    # W adapts insertion strength to target image local texture/luminance.
    # Watermark is invisible in flat areas, hidden in textured areas.
    # ----------------------------------------------------------------
    for number in range(target_start, target_stop + 1):
        target_path = clean_dir / f"{number}.png"
        target_arr = np.array(Image.open(target_path).convert("RGB"))
        t_h, t_w = target_arr.shape[:2]
 
        # Resize watermark pattern to target size
        wm_pattern = resize_pattern(watermark_pattern, t_h, t_w)
 
        # Compute perceptual weight from target image
        W = compute_weight(target_arr, alpha=ALPHA, window=WIENER_WINDOW)
        W = W[:, :, np.newaxis]  # (H, W, 1) for broadcast
 
        # Insert: only sign of watermark, magnitude from perceptual mask
        target_f = target_arr.astype(np.float64)
        forged = target_f + STRENGTH * W * np.sign(wm_pattern)
        forged = np.clip(forged, 0, 255).astype(np.uint8)
 
        Image.fromarray(forged).save(TEMP_OUT_DIR / f"{number}.png")
        total_processed += 1
 
    print(f"  Done.")
 
print(f"\nTotal forged: {total_processed}/200 images")
if total_processed != 200:
    print("[WARNING] Expected 200 images!")
 
# PACKAGE INTO ZIP
print(f"\nPackaging into {FILE_PATH}...")
with zipfile.ZipFile(FILE_PATH, "w", zipfile.ZIP_DEFLATED) as zipf:
    for img_path in sorted(TEMP_OUT_DIR.glob("*.png")):
        zipf.write(img_path, arcname=img_path.name)
 
print(f"Submission saved to {FILE_PATH}")