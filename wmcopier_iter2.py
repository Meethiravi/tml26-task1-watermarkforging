"""
WMCopier: Faithful implementation of Dong et al. (2025)
"WMCopier: Forging Invisible Image Watermarks on Arbitrary Images"

Algorithm 1 (from paper):
  Require: Clean image x, noise predictor ε_θ of pretrained M_θ,
           inversion steps T_S, refinement iterations L,
           low noise step t_l, step size η, tradeoff λ

  Stage 1 - Watermark Estimation:
    Train unconditional diffusion model M_θ on watermarked images D_aux
    (Paper: 20k iterations, batch 256, lr=1e-4)
    (Ours: fewer iterations due to only 25 images)

  Stage 2 - Watermark Injection (Shallow Inversion):
    x_Ts = Inversion(x, T_S)   ← DDIM invert clean image to step T_S
    Denoise from x_Ts to x'_0 using trained M_θ  ← forged image x^f

  Stage 3 - Refinement (Eq. 10, 11):
    For i = 1 to L:
      x_tl^f = sqrt(α_tl) * x^f + sqrt(1-α_tl) * z,  z~N(0,I)
      x^f += η * (-1/sqrt(1-α_tl) * ε_θ(x_tl^f, tl) - λ * (x^f - x))
"""

import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision import transforms
from diffusers import DDIMScheduler, UNet2DModel

# ----------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------
BASE_DIR = Path("/home/atml_team060/tml26_task4")
DATASET_DIR = BASE_DIR / "Dataset"
TEMP_OUT_DIR = BASE_DIR / "submission_temp_wmcopier_iter2"
FILE_PATH = str(BASE_DIR / "submission_wmcopier_iter2.zip")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Paper parameters (Algorithm 1 + Section 4)
T = 100             # total DDIM steps (paper uses 100)
T_S = 40            # shallow inversion depth (paper uses 40)
L = 100             # refinement iterations (paper uses 100)
T_L = 1             # low noise step for refinement (paper uses 1)
ETA = 1e-4          # refinement step size (paper uses 1e-4)
LAMBDA = 100        # quality tradeoff (paper uses 100)

# Training parameters (paper: 20k iters, batch 256, lr=1e-4)
# Scaled down for 25 images
TRAIN_ITERS = 5000  # fewer iters since we have 25 images not 5000
TRAIN_LR = 1e-4
TRAIN_BATCH = 8     # small batch since only 25 images
IMG_SIZE = 256      # paper uses resolution matching watermark scheme

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
# TRANSFORMS
# ----------------------------------------------------------------
to_tensor = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)  # [-1, 1]
])

def tensor_to_pil(t, orig_size=None):
    t = t.detach().cpu().squeeze(0)
    t = (t * 0.5 + 0.5).clamp(0, 1)
    pil = transforms.ToPILImage()(t)
    if orig_size:
        pil = pil.resize(orig_size, Image.BILINEAR)
    return pil

def load_tensor(path):
    return to_tensor(Image.open(path).convert("RGB")).unsqueeze(0).to(DEVICE)


# ----------------------------------------------------------------
# UNCONDITIONAL DIFFUSION MODEL (UNet2DModel)
# Paper Section 4.1: train M_θ on D_aux (watermarked images)
# "We adopt a standard DDIM framework for training,
#  following the official HuggingFace tutorial."
# ----------------------------------------------------------------
def build_unet(img_size=256):
    """
    Paper uses a standard unconditional UNet.
    UNet2DModel from diffusers matches the HuggingFace tutorial they reference.
    Scaled down from paper's model for 25-image training.
    """
    return UNet2DModel(
        sample_size=img_size,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(64, 128, 256, 256),  # smaller than paper
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "AttnDownBlock2D",
        ),
        up_block_types=(
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    ).to(DEVICE)


# ----------------------------------------------------------------
# TRAIN UNCONDITIONAL DIFFUSION MODEL ON WATERMARKED IMAGES
# Paper Eq. 3: minimize E[||ε_θ(x_t, t) - ε||^2]
# Paper Appendix G: 20k iterations, batch 256, lr=1e-4
# ----------------------------------------------------------------
def train_diffusion_model(wm_tensors, iters=2000, lr=1e-4, batch_size=4):
    """
    Train unconditional diffusion model on watermarked images.
    The model learns to denoise watermarked images, capturing
    the watermark distribution p_w(x) (Paper Section 4.1).
    """
    noise_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
    )

    unet = build_unet(IMG_SIZE)
    optimizer = optim.AdamW(unet.parameters(), lr=lr)
    mse_loss = nn.MSELoss()

    wm_stack = torch.cat(wm_tensors, dim=0)  # (N, 3, H, W)
    N = wm_stack.shape[0]

    unet.train()
    for iteration in range(iters):
        # Sample random batch from watermarked images
        idx = torch.randint(0, N, (batch_size,))
        x0 = wm_stack[idx].to(DEVICE)

        # Sample random timesteps
        t = torch.randint(0, noise_scheduler.config.num_train_timesteps,
                         (batch_size,), device=DEVICE).long()

        # Add noise to images (forward process, Eq. 1)
        noise = torch.randn_like(x0)
        x_t = noise_scheduler.add_noise(x0, noise, t)

        # Predict noise (Eq. 3)
        noise_pred = unet(x_t, t).sample

        # Loss: ||ε_θ(x_t, t) - ε||^2
        loss = mse_loss(noise_pred, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (iteration + 1) % 500 == 0:
            print(f"    Iter {iteration+1}/{iters}: loss={loss.item():.4f}")

    unet.eval()
    return unet, noise_scheduler


# ----------------------------------------------------------------
# DDIM INVERSION
# Paper Eq. 4: x_T = Inversion(x, T)
# Deterministic DDIM inversion: maps image back to noisy latent.
# ----------------------------------------------------------------
@torch.no_grad()
def ddim_inversion(x0, unet, scheduler, T_S):
    """
    DDIM inversion from x0 to x_{T_S}.
    Paper: "we first apply DDIM inversion to obtain latent x_{T_S}"
    Uses the trained M_θ (unet) to predict noise at each step.
    """
    scheduler.set_timesteps(T)
    timesteps = scheduler.timesteps  # [T-1, ..., 0] descending

    # Get inversion timesteps (ascending: 0 → T_S)
    inv_timesteps = list(reversed(timesteps))[:T_S]

    x = x0.clone()
    for t in inv_timesteps:
        t_batch = torch.tensor([t], device=DEVICE).long()
        noise_pred = unet(x, t_batch).sample

        # Get alpha values
        alpha_prod_t = scheduler.alphas_cumprod[t]
        alpha_prod_t_next = (
            scheduler.alphas_cumprod[inv_timesteps[inv_timesteps.index(t) + 1]]
            if inv_timesteps.index(t) + 1 < len(inv_timesteps)
            else torch.tensor(1.0)
        )

        # DDIM forward step (inversion)
        x0_pred = (x - (1 - alpha_prod_t).sqrt() * noise_pred) / alpha_prod_t.sqrt()
        x = alpha_prod_t_next.sqrt() * x0_pred + (1 - alpha_prod_t_next).sqrt() * noise_pred

    return x  # x_{T_S}


# ----------------------------------------------------------------
# DDIM DENOISING (Shallow)
# Paper Algorithm 1: denoise from x_{T_S} to x'_0 using M_θ
# This injects the watermark distribution learned by M_θ.
# ----------------------------------------------------------------
@torch.no_grad()
def ddim_denoise(x_Ts, unet, scheduler, T_S):
    """
    DDIM denoising from x_{T_S} to x'_0 using trained M_θ.
    Paper Eq. 2: x_{t-1} = sqrt(α_{t-1}) * x0_pred + sqrt(1-α_{t-1}) * ε_t
    The trained model's bias toward watermarked distribution
    naturally guides the output toward p_w(x).
    """
    scheduler.set_timesteps(T)
    timesteps = scheduler.timesteps  # descending

    # Only denoise from T_S back to 0 (shallow)
    active_timesteps = timesteps[T - T_S:]

    x = x_Ts.clone()
    for t in active_timesteps:
        t_batch = torch.tensor([t], device=DEVICE).long()
        noise_pred = unet(x, t_batch).sample

        # DDIM step (Eq. 2)
        x = scheduler.step(noise_pred, t, x).prev_sample

    return x  # x'_0 = forged image x^f


# ----------------------------------------------------------------
# REFINEMENT (Paper Algorithm 1, Eq. 10-11)
# x^{f(i+1)} = x^{f(i)} + η * ∇[log p_w(x^{f(i)}) - λ||x^{f(i)} - x||^2]
# Score function approximated via Eq. 11:
# ∇ log p_w(x^f) ≈ -1/sqrt(1-α_{tl}) * ε_θ(x_{tl}^f, tl)
# ----------------------------------------------------------------
def refine(x_f, x_clean, unet, scheduler, L=100, eta=1e-4, lam=100, t_l=1):
    """
    Refinement procedure from Paper Algorithm 1.
    Iteratively adjusts x^f to:
      1. Stay in high-probability region of watermark distribution p_w(x)
         (via score function estimated by trained M_θ)
      2. Remain close to clean image x
         (via MSE term)
    """
    scheduler.set_timesteps(T)
    alpha_tl = scheduler.alphas_cumprod[t_l].to(DEVICE)

    x_f = x_f.clone().detach()
    x_clean = x_clean.detach()

    for i in range(L):
        x_f.requires_grad_(True)

        # Add noise to low noise step t_l (Eq. 11 setup)
        z = torch.randn_like(x_f)
        x_f_tl = alpha_tl.sqrt() * x_f + (1 - alpha_tl).sqrt() * z

        t_batch = torch.tensor([t_l], device=DEVICE).long()

        # Score function: ∇ log p_w(x^f) ≈ -1/sqrt(1-α_tl) * ε_θ(x_tl^f, tl)
        noise_pred = unet(x_f_tl, t_batch).sample
        score = -noise_pred / (1 - alpha_tl).sqrt()

        # MSE gradient: ∇(-λ||x^f - x||^2) = -2λ(x^f - x)
        mse_grad = -2 * lam * (x_f - x_clean)

        # Gradient step (Eq. 10)
        with torch.no_grad():
            grad = score + mse_grad
            x_f = x_f + eta * grad
            x_f = x_f.clamp(-1, 1)

        if x_f.requires_grad:
            x_f = x_f.detach()

    return x_f.detach()


# ----------------------------------------------------------------
# MAIN PIPELINE (Algorithm 1)
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

    # Load all 25 watermarked source images as tensors
    print(f"  Loading {len(source_paths)} watermarked source images...")
    wm_tensors = [load_tensor(p) for p in source_paths]

    # ----------------------------------------------------------------
    # STAGE 1: Train unconditional diffusion model on D_aux
    # Paper: 20k iters on 5000 images
    # Ours: {TRAIN_ITERS} iters on 25 images (memorization expected)
    # ----------------------------------------------------------------
    print(f"  Stage 1: Training diffusion model on {len(wm_tensors)} watermarked images...")
    print(f"  (Paper note: with few images, model memorizes watermark distribution)")
    unet, scheduler = train_diffusion_model(
        wm_tensors, iters=TRAIN_ITERS, lr=TRAIN_LR, batch_size=TRAIN_BATCH
    )

    # ----------------------------------------------------------------
    # STAGE 2 + 3: For each target clean image:
    #   - DDIM invert to T_S (shallow inversion)
    #   - DDIM denoise using trained M_θ → forged image x^f
    #   - Refinement → final forged image
    # ----------------------------------------------------------------
    print(f"  Stage 2+3: Forging onto {target_stop - target_start + 1} target images...")
    for number in range(target_start, target_stop + 1):
        target_path = clean_dir / f"{number}.png"
        orig_pil = Image.open(target_path).convert("RGB")
        orig_size = orig_pil.size

        x_clean = load_tensor(target_path)

        # Stage 2a: DDIM inversion (x_clean → x_{T_S})
        x_Ts = ddim_inversion(x_clean, unet, scheduler, T_S)

        # Stage 2b: DDIM denoising with trained M_θ (x_{T_S} → x^f)
        x_f = ddim_denoise(x_Ts, unet, scheduler, T_S)

        # Stage 3: Refinement (Eq. 10-11)
        x_f_refined = refine(
            x_f, x_clean, unet, scheduler,
            L=L, eta=ETA, lam=LAMBDA, t_l=T_L
        )

        # Save at original resolution
        forged_pil = tensor_to_pil(x_f_refined, orig_size=orig_size)
        forged_pil.save(TEMP_OUT_DIR / f"{number}.png")
        total_processed += 1

    print(f"  Done: {target_stop - target_start + 1} images.")

    # Free GPU memory between WM methods
    del unet
    torch.cuda.empty_cache()

print(f"\nTotal forged: {total_processed}/200 images")
if total_processed != 200:
    print("[WARNING] Expected 200 images!")

# Package into zip
print(f"\nPackaging into {FILE_PATH}...")
with zipfile.ZipFile(FILE_PATH, "w", zipfile.ZIP_DEFLATED) as zipf:
    for img_path in sorted(TEMP_OUT_DIR.glob("*.png")):
        zipf.write(img_path, arcname=img_path.name)

print(f"Submission saved to {FILE_PATH}")
print("\nPaper parameters used:")
print(f"  T={T}, T_S={T_S}, L={L}, t_l={T_L}, η={ETA}, λ={LAMBDA}")
print(f"  Training: {TRAIN_ITERS} iters, lr={TRAIN_LR}, batch={TRAIN_BATCH}")
print("\nNote: Paper uses 5000 images and 20k iters on A100.")
print("With 25 images, the model memorizes the watermark distribution")
print("rather than generalizing — but this is still the correct algorithm.")