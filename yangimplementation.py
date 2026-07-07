

import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import requests
from PIL import Image

# CONFIG
ZIP_FILE = "/home/atml_team060/tml26_task4/Dataset.zip"
DATASET_DIR = Path("/home/atml_team060/tml26_task4/Dataset")
TEMP_OUT_DIR = Path("/home/atml_team060/tml26_task4/submission_temp")
FILE_PATH = "/home/atml_team060/tml26_task4/submission.zip"

# Leaderboard submission
# BASE_URL  = "http://35.192.205.84:80"
# API_KEY  = "YOUR_API_KEY_HERE"  # REPLACE WITH YOUR API KEY
# TASK_ID   = "22-forging-task"

# 1. UNZIP DATASET
if not DATASET_DIR.exists():
    if not os.path.exists(ZIP_FILE):
        raise FileNotFoundError(f"Could not find {ZIP_FILE}. Please download the dataset first.")

    print(f"Unzipping {ZIP_FILE}...")
    with zipfile.ZipFile(ZIP_FILE, "r") as zip_ref:
        print("Files in zip:", zip_ref.namelist()[:10])  # show first 10 files
        zip_ref.extractall(DATASET_DIR)

else:
    print("Dataset already extracted.")

# Ensure output directory exists
TEMP_OUT_DIR.mkdir(exist_ok=True)

# Map the Dataset structure: (Source_Folder, Size_Subfolder, Target_Folder)
CATEGORIES = [
    ("WM_1", 1, 25),
    ("WM_2", 26, 50),
    ("WM_3", 51, 75),
    ("WM_4", 76, 100),
    ("WM_5", 101, 125),
    ("WM_6", 126, 150),
    ("WM_7", 151, 175),
    ("WM_8", 176, 200),
]

ALPHA = 1.0

print("\nLoading clean reference image pool...")
clean_dir = DATASET_DIR / "clean_targets"
clean_pool = []
for p in sorted(clean_dir.glob("*.png")):
    img = Image.open(p).convert("RGB")
    clean_pool.append(np.array(img).astype(np.float32))
 
print(f"Loaded {len(clean_pool)} clean reference images.")
 
total_processed = 0
 
for source_wm, target_start, target_stop in CATEGORIES:
    print(f"\nProcessing {source_wm} -> images {target_start}.png to {target_stop}.png ...")
 
    source_dir = DATASET_DIR / "watermarked_sources" / source_wm
    source_paths = sorted(source_dir.glob("*.png"))
 
    if not source_paths:
        print(f"  [Warning] No source images found in {source_dir}")
        continue
 
    # Load all watermarked source images
    wm_stack = []
    for p in source_paths:
        img = Image.open(p).convert("RGB")
        wm_stack.append(np.array(img).astype(np.float32))
 
    wm_mean = np.mean(wm_stack, axis=0)  # (H, W, 3)
    s_h, s_w = wm_mean.shape[:2]
 
    clean_resized = []
    for c in clean_pool:
        c_h, c_w = c.shape[:2]
        if (c_h, c_w) != (s_h, s_w):
            c_img = Image.fromarray(c.astype(np.uint8)).resize((s_w, s_h), Image.BILINEAR)
            c = np.array(c_img).astype(np.float32)
        clean_resized.append(c)
 
    clean_mean = np.mean(clean_resized, axis=0)  # (H, W, 3)
 
    # Watermark pattern = watermarked mean - clean mean
    watermark_pattern = wm_mean - clean_mean
 
    print(f"  Watermark pattern stats: "
          f"min={watermark_pattern.min():.2f}, "
          f"max={watermark_pattern.max():.2f}, "
          f"std={watermark_pattern.std():.4f}")
 
    # Apply watermark to each target clean image
    for number in range(target_start, target_stop + 1):
        target_path = clean_dir / f"{number}.png"
        target_pil = Image.open(target_path).convert("RGB")
        target_arr = np.array(target_pil).astype(np.float32)
 
        t_h, t_w = target_arr.shape[:2]
 
        # Resize watermark pattern if target size differs from source size
        if (t_h, t_w) != (s_h, s_w):
            wm_pil = Image.fromarray(
                np.clip(watermark_pattern + 128, 0, 255).astype(np.uint8)
            )
            wm_resized = wm_pil.resize((t_w, t_h), Image.BILINEAR)
            wm_pattern = np.array(wm_resized).astype(np.float32) - 128.0
        else:
            wm_pattern = watermark_pattern
 
        # Forge: add watermark residual to clean target
        forged = target_arr + ALPHA * wm_pattern
        forged = np.clip(forged, 0, 255).astype(np.uint8)
 
        out_path = TEMP_OUT_DIR / f"{number}.png"
        Image.fromarray(forged).save(out_path)
        total_processed += 1
 
    print(f"  Done.")
 
print(f"\nTotal forged: {total_processed}/200 images")
if total_processed != 200:
    print("[WARNING] Expected 200 images!")
 
# 3. PACKAGE INTO ZIP
print(f"\nPackaging into {FILE_PATH}...")
with zipfile.ZipFile(FILE_PATH, "w", zipfile.ZIP_DEFLATED) as zipf:
    for img_path in sorted(TEMP_OUT_DIR.glob("*.png")):
        zipf.write(img_path, arcname=img_path.name)
 
print(f"Submission saved to {FILE_PATH}")