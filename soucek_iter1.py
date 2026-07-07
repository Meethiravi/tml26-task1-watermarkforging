"""
Adapted from: Soucek et al. (2025) "Transferable Black-Box One-Shot Forging of
Watermarks via Image Preference Models", NeurIPS 2025.

FIXES vs the previous version (which produced extremely distorted images):

  1. HARD L-infinity CONSTRAINT (PGD-style projection) on `delta` after every
     optimizer step, in BOTH extraction and forging. This is the standard fix
     for unconstrained adversarial-style optimization blowing out pixels: a
     soft penalty (lambda * MSE) can be arbitrarily outweighed by an
     unbounded score gradient, but a hard clip to [-eps, eps] per pixel
     cannot. This is almost certainly why your images were "extremely
     distorted" -- delta was growing unchecked until clamp(-1,1) saturated
     large regions.

  2. NO DOUBLE RESIZE. Previously: resize down to 256 -> forge -> resize back
     up to original resolution. That round trip blurs/resamples the ENTIRE
     image, inflating LPIPS independent of the watermark. Now: the
     preference model still operates on a 256x256 view (it needs a fixed
     input size), but the optimized delta is upsampled with bilinear
     interpolation and added directly onto the ORIGINAL full-resolution
     image. Only the small delta is resampled, not the whole photo.

  3. Proper ImageNet normalization before the ResNet18 backbone (it was
     pretrained expecting ImageNet mean/std, not just [0,1]).

  4. Preference score is squashed with tanh so both loss terms are on a
     bounded, comparable scale -- removes the scale-mismatch bug where
     pref_loss could dwarf lambda_quality * quality_loss by orders of
     magnitude.

  5. Early stopping per-image once quality_loss exceeds a ceiling, as a
     second safety net on top of the hard epsilon clip.

Everything else (preference-model training on synthetic artifacts, per-batch
watermark extraction+averaging, forging loop) follows the same structure as
before.
"""

import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision import models, transforms

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
BASE_DIR = Path("/home/atml_team060/tml26_task4")
DATASET_DIR = BASE_DIR / "Dataset"
TEMP_OUT_DIR = BASE_DIR / "submission_temp_soucek_iter1"
FILE_PATH = str(BASE_DIR / "submission_soucek_iter1.zip")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- Hyperparameters ---
PREF_MODEL_STEPS = 1000
PREF_LR = 1e-4

EXTRACT_STEPS = 150
EXTRACT_LR = 0.01

FORGE_STEPS = 150
FORGE_LR = 0.01
LAMBDA_QUALITY = 50.0        # much higher now that pref score is tanh-bounded to [-1,1]

# --- Hard perturbation budgets (THE key fix). Values are in the model's
# normalized [-1,1] pixel space. eps=0.06 corresponds to roughly 8/255 in
# standard [0,255] terms (a common "imperceptible" adversarial budget);
# eps=0.12 corresponds to roughly 15/255. Tune down if quality still bad,
# tune up if bit-accuracy is too low after quality is fixed.
EXTRACT_EPS = 0.06
FORGE_EPS = 0.06

# --- Early-stopping safety net ---
MAX_QUALITY_LOSS = 0.01     # stop optimizing an image early if MSE exceeds this

IMG_SIZE = 256               # working resolution for the preference model only

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

# ----------------------------------------------------------------------
# TRANSFORMS
# ----------------------------------------------------------------------
# Model-space normalization: maps [0,1] -> [-1,1] for our own tensors.
normalize = transforms.Normalize([0.5] * 3, [0.5] * 3)
to_tensor_256 = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    normalize,
])
to_tensor_native = transforms.Compose([
    transforms.ToTensor(),
    normalize,
])

# ImageNet normalization applied ON TOP of [0,1] just before the ResNet
# backbone (fix #3).
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def tensor_to_pil(t):
    t = t.detach().cpu().squeeze(0)
    t = (t * 0.5 + 0.5).clamp(0, 1)
    return transforms.ToPILImage()(t)


def load_tensor_256(path):
    return to_tensor_256(Image.open(path).convert("RGB")).unsqueeze(0).to(DEVICE)


def load_tensor_native(path):
    return to_tensor_native(Image.open(path).convert("RGB")).unsqueeze(0).to(DEVICE)


def resize_delta(delta, h, w):
    """Upsample a small (1,3,256,256) delta to (1,3,h,w) via bilinear."""
    return torch.nn.functional.interpolate(
        delta, size=(h, w), mode="bilinear", align_corners=False
    )


# ----------------------------------------------------------------------
# PREFERENCE MODEL
# ----------------------------------------------------------------------
class PreferenceModel(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        for i, child in enumerate(self.features.children()):
            if i < 6:
                for p in child.parameters():
                    p.requires_grad = False
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.register_buffer("imagenet_mean", IMAGENET_MEAN)
        self.register_buffer("imagenet_std", IMAGENET_STD)

    def forward(self, x):
        # x is in [-1,1] (our normalized space) -> back to [0,1] -> ImageNet norm
        x01 = x * 0.5 + 0.5
        x_im = (x01 - self.imagenet_mean.to(x.device)) / self.imagenet_std.to(x.device)
        raw = self.head(self.features(x_im)).squeeze(-1)
        # Fix #4: bound the score so it's on a comparable scale to the
        # (also bounded) quality loss term.
        return torch.tanh(raw)


# ----------------------------------------------------------------------
# SYNTHETIC ARTIFACT GENERATOR (unchanged)
# ----------------------------------------------------------------------
def generate_synthetic_artifact(batch_size, h=IMG_SIZE, w=IMG_SIZE):
    artifacts = []
    for _ in range(batch_size):
        artifact_type = np.random.randint(3)
        if artifact_type == 0:
            freq_x = np.random.uniform(0.01, 0.1)
            freq_y = np.random.uniform(0.01, 0.1)
            phase = np.random.uniform(0, 2 * np.pi)
            x = np.linspace(0, 2 * np.pi * freq_x * w, w)
            y = np.linspace(0, 2 * np.pi * freq_y * h, h)
            xx, yy = np.meshgrid(x, y)
            pattern = np.sin(xx + yy + phase).astype(np.float32)
            pattern = pattern[:, :, np.newaxis].repeat(3, axis=2)
        elif artifact_type == 1:
            pattern = np.random.randn(h, w, 3).astype(np.float32)
            pattern /= (np.abs(pattern).max() + 1e-8)
        else:
            block_size = np.random.randint(8, 32)
            pattern = np.zeros((h, w, 3), dtype=np.float32)
            sign = np.random.choice([-1, 1], size=(h // block_size + 1, w // block_size + 1, 3))
            for i in range(h):
                for j in range(w):
                    pattern[i, j] = sign[i // block_size, j // block_size]
        strength = np.random.uniform(0.01, 0.05)
        artifacts.append(pattern * strength)
    return torch.tensor(np.stack(artifacts)).permute(0, 3, 1, 2).float().to(DEVICE)


# ----------------------------------------------------------------------
# TRAIN PREFERENCE MODEL (unchanged apart from the tanh-bounded output)
# ----------------------------------------------------------------------
def train_preference_model(clean_tensors, steps=1000, lr=1e-4):
    model = PreferenceModel().to(DEVICE)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    n_clean = len(clean_tensors)
    clean_stack = torch.cat(clean_tensors, dim=0)

    print(f"  Training preference model for {steps} steps...")
    model.train()
    for step in range(steps):
        idx = torch.randint(0, n_clean, (8,))
        x_clean = clean_stack[idx]
        artifact = generate_synthetic_artifact(8, h=IMG_SIZE, w=IMG_SIZE)
        x_artifact = (x_clean + artifact).clamp(-1, 1)

        score_clean = model(x_clean)
        score_artifact = model(x_artifact)
        margin = 0.2  # smaller now that scores are tanh-bounded to [-1,1]
        loss = torch.clamp(margin + score_clean - score_artifact, min=0).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % 200 == 0:
            print(f"    Step {step+1}/{steps}: loss={loss.item():.4f}, "
                  f"score_clean={score_clean.mean().item():.3f}, "
                  f"score_artifact={score_artifact.mean().item():.3f}")

    model.eval()
    return model


# ----------------------------------------------------------------------
# EXTRACT WATERMARK — now with hard epsilon-ball projection (fix #1)
# ----------------------------------------------------------------------
def extract_watermark(source_tensors, preference_model, steps=150, lr=0.01, eps=0.06):
    deltas = []
    preference_model.eval()

    for x_wm in source_tensors:
        x_wm = x_wm.to(DEVICE)
        delta = nn.Parameter(torch.zeros_like(x_wm))
        optimizer = optim.Adam([delta], lr=lr)

        for step in range(steps):
            optimizer.zero_grad()
            x_clean_est = (x_wm - delta).clamp(-1, 1)
            loss = preference_model(x_clean_est).mean()
            loss = loss + 0.1 * (delta ** 2).mean()
            loss.backward()
            optimizer.step()

            # HARD PROJECTION: clip delta to the epsilon ball after every step.
            with torch.no_grad():
                delta.clamp_(-eps, eps)

        deltas.append(delta.detach().cpu())

    watermark = torch.stack(deltas).mean(dim=0)
    return watermark  # (1, 3, IMG_SIZE, IMG_SIZE)


# ----------------------------------------------------------------------
# FORGE WATERMARK — hard epsilon projection + native-resolution application
# ----------------------------------------------------------------------
def forge_watermark(target_native, watermark_pattern_256, preference_model,
                     steps=150, lr=0.01, lambda_quality=50.0,
                     eps=0.06, max_quality_loss=0.01):
    """
    target_native: (1,3,H,W) full-resolution tensor in [-1,1].
    watermark_pattern_256: (1,3,256,256) starting delta from extraction.

    We optimize `delta` at 256x256 (cheap, and matches what the preference
    model expects), scoring the preference model on a 256x256 DOWNSAMPLED
    VIEW of (target + upsampled_delta) -- so the model's gradient signal is
    computed on a realistic view of the final image, not a separately
    resized copy that gets kept. Only `delta` (upsampled) is ever added to
    the real, full-resolution output image.
    """
    preference_model.eval()
    target_native = target_native.to(DEVICE)
    h, w = target_native.shape[-2:]

    delta = nn.Parameter(watermark_pattern_256.to(DEVICE).clone())
    optimizer = optim.Adam([delta], lr=lr)

    downsample = transforms.Resize((IMG_SIZE, IMG_SIZE), antialias=True)

    for step in range(steps):
        optimizer.zero_grad()

        delta_full = resize_delta(delta, h, w)
        forged_native = (target_native + delta_full).clamp(-1, 1)

        # Preference model needs a fixed-size input -> downsample a VIEW
        # only for scoring; the actual output image stays full-resolution.
        forged_256 = downsample(forged_native)

        pref_loss = -preference_model(forged_256).mean()
        quality_loss = ((forged_native - target_native) ** 2).mean()

        loss = pref_loss + lambda_quality * quality_loss
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            delta.clamp_(-eps, eps)

        # Safety net: stop early if we've already drifted too far.
        if quality_loss.item() > max_quality_loss:
            break

    with torch.no_grad():
        delta_full = resize_delta(delta, h, w)
        forged_final = (target_native + delta_full).clamp(-1, 1)

    return forged_final.detach()


# ----------------------------------------------------------------------
# MAIN PIPELINE
# ----------------------------------------------------------------------
if not (DATASET_DIR / "watermarked_sources").exists():
    print("Unzipping...")
    with zipfile.ZipFile(str(BASE_DIR / "Dataset.zip"), "r") as z:
        z.extractall(str(BASE_DIR))
else:
    print("Dataset already extracted.")

TEMP_OUT_DIR.mkdir(exist_ok=True)
clean_dir = DATASET_DIR / "clean_targets"

print("Loading clean images (256x256 view for preference-model training)...")
clean_paths = sorted(clean_dir.glob("*.png"))
clean_tensors_256 = [load_tensor_256(p) for p in clean_paths]
print(f"Loaded {len(clean_tensors_256)} clean images.")

print("\n" + "=" * 60)
print("Stage 1: Training preference model (Soucek et al.)")
print("=" * 60)
preference_model = train_preference_model(clean_tensors_256, steps=PREF_MODEL_STEPS, lr=PREF_LR)

total_processed = 0

for source_wm, target_start, target_stop in CATEGORIES:
    print(f"\n{'='*60}")
    print(f"[{source_wm}] -> images {target_start} to {target_stop}")
    print(f"{'='*60}")

    source_dir = DATASET_DIR / "watermarked_sources" / source_wm
    source_paths = sorted(source_dir.glob("*.png"))
    if not source_paths:
        print("  [Warning] No source images found.")
        continue

    source_tensors_256 = [load_tensor_256(p) for p in source_paths]

    print(f"  Stage 2: Extracting watermark from {len(source_tensors_256)} source images "
          f"(eps={EXTRACT_EPS})...")
    watermark_pattern = extract_watermark(
        source_tensors_256, preference_model,
        steps=EXTRACT_STEPS, lr=EXTRACT_LR, eps=EXTRACT_EPS,
    )
    print(f"  Watermark pattern std: {watermark_pattern.std().item():.4f}  "
          f"max_abs: {watermark_pattern.abs().max().item():.4f}")

    print(f"  Stage 3: Forging onto target images (eps={FORGE_EPS}, "
          f"lambda_quality={LAMBDA_QUALITY})...")
    for number in range(target_start, target_stop + 1):
        target_path = clean_dir / f"{number}.png"
        target_native = load_tensor_native(target_path)  # full resolution, no resize

        forged_native = forge_watermark(
            target_native, watermark_pattern, preference_model,
            steps=FORGE_STEPS, lr=FORGE_LR, lambda_quality=LAMBDA_QUALITY,
            eps=FORGE_EPS, max_quality_loss=MAX_QUALITY_LOSS,
        )

        forged_pil = tensor_to_pil(forged_native)  # already native resolution
        forged_pil.save(TEMP_OUT_DIR / f"{number}.png")
        total_processed += 1

    print(f"  Done: {target_stop - target_start + 1} images.")

print(f"\nTotal forged: {total_processed}/200 images")
if total_processed != 200:
    print("[WARNING] Expected 200 images!")

print(f"\nPackaging into {FILE_PATH}...")
with zipfile.ZipFile(FILE_PATH, "w", zipfile.ZIP_DEFLATED) as zipf:
    for img_path in sorted(TEMP_OUT_DIR.glob("*.png")):
        zipf.write(img_path, arcname=img_path.name)

print(f"Submission saved to {FILE_PATH}")
print("\nTuning guide:")
print("  If images are STILL too distorted: lower EXTRACT_EPS / FORGE_EPS (e.g. 0.03),")
print("  or raise LAMBDA_QUALITY further (e.g. 100-200), or lower MAX_QUALITY_LOSS.")
print("  If bit-accuracy is too low once quality looks fine: raise EXTRACT_EPS/FORGE_EPS")
print("  in small steps (0.06 -> 0.08 -> 0.10) and re-check images visually each time.")