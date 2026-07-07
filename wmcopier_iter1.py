"""
WMCopier-inspired implementation using DDIM inversion.
Dong et al. (2025) "WMCopier: Forging Invisible Image Watermarks on Arbitrary Images"

Pipeline per WM method:
  Stage 1 - Watermark Estimation:
    For each of the 25 source watermarked images:
      - Encode to latent space via VAE
      - DDIM invert to noise (T steps forward)
      - DDIM denoise back to image (T steps backward) → clean estimate
      - residual = watermarked - clean_estimate = watermark signal
    Average residuals across 25 images → aggregated watermark pattern

  Stage 2 - Watermark Injection:
    For each clean target:
      - Add averaged watermark pattern (initialization)
      - Refine via optimization to preserve quality

  Stage 3 - Refinement:
    Short optimization loop keeping forged close to target
    while maximizing watermark signal strength.
"""

import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision import transforms

# ----------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------
BASE_DIR = Path("/home/atml_team060/tml26_task4")
DATASET_DIR = BASE_DIR / "Dataset"
TEMP_OUT_DIR = BASE_DIR / "submission_temp_wmcopier_iter1"
FILE_PATH = str(BASE_DIR / "submission_wmcopier_iter1.zip")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# DDIM parameters
DDIM_STEPS = 20          # inversion/denoising steps (paper uses 50, 20 is faster)
NOISE_LEVEL = 0.5        # how far to invert (0=no inversion, 1=full noise)
                         # 0.3-0.5 gives good watermark extraction without destroying image

# Watermark injection parameters
ALPHA = 1.0              # averaging watermark strength
REFINE_STEPS = 50        # refinement optimization steps
REFINE_LR = 0.01         # refinement learning rate
LAMBDA_QUALITY = 10.0    # quality preservation weight

IMG_SIZE = 512           # SD works at 512x512

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

# ----------------------------------------------------------------
# LOAD STABLE DIFFUSION COMPONENTS
# We only need the VAE and scheduler — not the full UNet/text encoder.
# VAE encodes images to latent space and back.
# DDIM scheduler handles the inversion/denoising timesteps.
# ----------------------------------------------------------------
print("Loading Stable Diffusion VAE and scheduler...")
from diffusers import AutoencoderKL, DDIMScheduler

VAE_MODEL = "stabilityai/sd-vae-ft-mse"  # lightweight VAE, no UNet needed
vae = AutoencoderKL.from_pretrained(VAE_MODEL).to(DEVICE)
vae.eval()
for p in vae.parameters():
    p.requires_grad = False

scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
)
scheduler.set_timesteps(DDIM_STEPS)

print("VAE and scheduler loaded.")

# ----------------------------------------------------------------
# TRANSFORMS
# ----------------------------------------------------------------
to_tensor = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)  # [-1, 1]
])

def pil_to_latent(pil_img):
    """Encode PIL image to VAE latent space."""
    x = to_tensor(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        latent = vae.encode(x).latent_dist.mean * 0.18215
    return latent  # (1, 4, 64, 64)

def latent_to_pil(latent):
    """Decode VAE latent back to PIL image."""
    with torch.no_grad():
        x = vae.decode(latent / 0.18215).sample
    x = (x * 0.5 + 0.5).clamp(0, 1)
    x = x.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray((x * 255).astype(np.uint8))

def tensor_to_pil(t):
    t = t.detach().cpu().squeeze(0)
    t = (t * 0.5 + 0.5).clamp(0, 1)
    return transforms.ToPILImage()(t)


# ----------------------------------------------------------------
# DDIM INVERSION
# WMCopier Section 4.1: "Shallow Inversion"
# Instead of fully inverting to noise, we only go partway (noise_level).
# This preserves image structure while exposing the watermark signal.
#
# Forward process (inversion): image → partially noisy latent
# Backward process (denoising): partially noisy latent → clean image estimate
#
# The difference: watermarked - clean_estimate = watermark signal
# ----------------------------------------------------------------
def ddim_invert_and_denoise(latent, noise_level=0.5):
    """
    WMCopier shallow DDIM inversion:
    1. Add noise up to timestep T*noise_level (partial forward diffusion)
    2. Denoise back to t=0 (backward diffusion without conditioning)
    Returns the denoised clean estimate.
    """
    # Number of steps to invert
    n_inv_steps = int(DDIM_STEPS * noise_level)
    timesteps = scheduler.timesteps

    # Step 1: Add noise up to partial timestep (forward process)
    t_inv = timesteps[DDIM_STEPS - n_inv_steps - 1]
    noise = torch.randn_like(latent)
    noisy_latent = scheduler.add_noise(latent, noise, t_inv.unsqueeze(0))

    # Step 2: Denoise back using scheduler (backward process)
    # Without UNet, we approximate denoising by gradually removing noise.
    # This is a simplified version — full WMCopier uses a trained UNet.
    # Here we use the scheduler's noise prediction via direct latent interpolation.
    denoised = noisy_latent.clone()
    active_timesteps = timesteps[DDIM_STEPS - n_inv_steps:]

    for i, t in enumerate(active_timesteps):
        # Compute alpha values for this timestep
        alpha_prod = scheduler.alphas_cumprod[t]
        alpha_prod_prev = scheduler.alphas_cumprod[active_timesteps[i-1]] if i > 0 else torch.tensor(1.0)

        # Estimate original latent (x0 prediction)
        # Without UNet, we use the noisy latent itself scaled back
        beta_prod = 1 - alpha_prod
        x0_pred = (denoised - beta_prod.sqrt() * noise) / alpha_prod.sqrt()
        x0_pred = x0_pred.clamp(-3, 3)

        # DDIM step
        denoised = alpha_prod_prev.sqrt() * x0_pred + (1 - alpha_prod_prev).sqrt() * noise

    return denoised


# ----------------------------------------------------------------
# WATERMARK EXTRACTION VIA DDIM
# WMCopier Eq. 4-6: estimate watermark as difference between
# watermarked image and its DDIM-denoised clean estimate.
# ----------------------------------------------------------------
def extract_watermark_ddim(source_paths, noise_level=0.5):
    """
    For each watermarked source image:
      1. Encode to latent
      2. DDIM invert and denoise → clean estimate latent
      3. Decode both to pixel space
      4. residual = watermarked_pixels - clean_pixels
    Average residuals across all source images.
    """
    residuals = []

    for i, p in enumerate(source_paths):
        pil_img = Image.open(p).convert("RGB")
        orig_size = pil_img.size

        # Encode watermarked image to latent
        latent_wm = pil_to_latent(pil_img)

        # DDIM shallow inversion → clean estimate
        latent_clean = ddim_invert_and_denoise(latent_wm, noise_level=noise_level)

        # Decode both to pixel space
        pil_wm_recon = latent_to_pil(latent_wm)
        pil_clean_est = latent_to_pil(latent_clean)

        # Resize to original size for residual computation
        pil_wm_recon = pil_wm_recon.resize(orig_size, Image.BILINEAR)
        pil_clean_est = pil_clean_est.resize(orig_size, Image.BILINEAR)

        arr_wm = np.array(pil_wm_recon).astype(np.float32)
        arr_clean = np.array(pil_clean_est).astype(np.float32)

        residual = arr_wm - arr_clean  # watermark signal
        residuals.append(residual)

        if (i + 1) % 5 == 0:
            print(f"    Processed {i+1}/{len(source_paths)} source images. "
                  f"Residual std: {residual.std():.4f}")

    # Aggregate: average residuals
    watermark_pattern = np.mean(residuals, axis=0)
    return watermark_pattern


# ----------------------------------------------------------------
# REFINEMENT STEP
# WMCopier Section 4.3: after injecting the watermark pattern,
# refine the forged image to improve quality.
# Optimize a small perturbation delta such that:
#   forged = target + wm_pattern + delta
# minimizes perceptual distance to target while
# preserving the watermark signal direction.
# ----------------------------------------------------------------
def refine_forgery(target_arr, wm_pattern, steps=50, lr=0.01, lambda_q=10.0):
    """
    Refinement via gradient descent on pixel perturbation.
    Keeps forged image close to target while preserving watermark direction.
    """
    t_h, t_w = target_arr.shape[:2]
    w_h, w_w = wm_pattern.shape[:2]

    # Resize watermark to target size if needed
    if (t_h, t_w) != (w_h, w_w):
        wm_pil = Image.fromarray(np.clip(wm_pattern + 128, 0, 255).astype(np.uint8))
        wm_pil = wm_pil.resize((t_w, t_h), Image.BILINEAR)
        wm_resized = np.array(wm_pil).astype(np.float32) - 128.0
    else:
        wm_resized = wm_pattern.copy()

    # Initialize: target + watermark
    target_t = torch.tensor(target_arr.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    wm_t = torch.tensor(wm_resized / 255.0).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    # Learnable refinement delta
    delta = nn.Parameter(torch.zeros_like(target_t))
    optimizer = optim.Adam([delta], lr=lr)

    wm_sign = torch.sign(wm_t)  # direction of watermark

    for step in range(steps):
        optimizer.zero_grad()

        forged = (target_t + wm_t + delta).clamp(0, 1)

        # Quality loss: stay close to target
        quality_loss = ((forged - target_t) ** 2).mean()

        # Direction loss: delta should not cancel the watermark direction
        direction_loss = ((delta * wm_sign).clamp(max=0) ** 2).mean()

        loss = lambda_q * quality_loss + direction_loss
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        forged_final = (target_t + wm_t + delta).clamp(0, 1)

    forged_np = forged_final.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return (forged_np * 255).astype(np.uint8)


# ----------------------------------------------------------------
# MAIN PIPELINE
# ----------------------------------------------------------------

# Unzip if needed
if not (DATASET_DIR / "watermarked_sources").exists():
    print("Unzipping dataset...")
    with zipfile.ZipFile(str(BASE_DIR / "Dataset.zip"), "r") as z:
        z.extractall(str(BASE_DIR))
else:
    print("Dataset already extracted.")

TEMP_OUT_DIR.mkdir(exist_ok=True)
clean_dir = DATASET_DIR / "clean_targets"

total_processed = 0

for source_wm, target_start, target_stop in CATEGORIES:
    print(f"\n{'='*60}")
    print(f"[{source_wm}] -> images {target_start} to {target_stop}")
    print(f"{'='*60}")

    source_dir = DATASET_DIR / "watermarked_sources" / source_wm
    source_paths = sorted(source_dir.glob("*.png"))

    if not source_paths:
        print(f"  [Warning] No source images found.")
        continue

    # ----------------------------------------------------------------
    # STAGE 1: Extract watermark via DDIM inversion
    # ----------------------------------------------------------------
    print(f"  Stage 1: Extracting watermark via DDIM inversion...")
    watermark_pattern = extract_watermark_ddim(source_paths, noise_level=NOISE_LEVEL)

    print(f"  Watermark pattern: min={watermark_pattern.min():.2f}, "
          f"max={watermark_pattern.max():.2f}, "
          f"std={watermark_pattern.std():.4f}")

    # ----------------------------------------------------------------
    # STAGE 2 + 3: Inject and refine on each target image
    # ----------------------------------------------------------------
    print(f"  Stage 2+3: Injecting and refining on target images...")
    for number in range(target_start, target_stop + 1):
        target_path = clean_dir / f"{number}.png"
        target_arr = np.array(Image.open(target_path).convert("RGB"))

        forged_arr = refine_forgery(
            target_arr, watermark_pattern * ALPHA,
            steps=REFINE_STEPS, lr=REFINE_LR, lambda_q=LAMBDA_QUALITY
        )

        Image.fromarray(forged_arr).save(TEMP_OUT_DIR / f"{number}.png")
        total_processed += 1

    print(f"  Done: {target_stop - target_start + 1} images.")

print(f"\nTotal forged: {total_processed}/200 images")
if total_processed != 200:
    print("[WARNING] Expected 200 images!")

# Package into zip
print(f"\nPackaging into {FILE_PATH}...")
with zipfile.ZipFile(FILE_PATH, "w", zipfile.ZIP_DEFLATED) as zipf:
    for img_path in sorted(TEMP_OUT_DIR.glob("*.png")):
        zipf.write(img_path, arcname=img_path.name)

print(f"Submission saved to {FILE_PATH}")
print("\nTuning guide:")
print("  NOISE_LEVEL:     higher = more DDIM inversion, stronger watermark extraction")
print("  ALPHA:           higher = stronger watermark injection")
print("  REFINE_STEPS:    more = better quality preservation")
print("  LAMBDA_QUALITY:  higher = better LPIPS, weaker detection")
print("  DDIM_STEPS:      more = better inversion quality, slower")