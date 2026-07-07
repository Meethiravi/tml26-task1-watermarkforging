import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision import models, transforms

# CONFIG
BASE_DIR = Path("/home/atml_team060/tml26_task4")
DATASET_DIR = BASE_DIR / "Dataset"
TEMP_OUT_DIR = BASE_DIR / "submission_temp_soucek"
FILE_PATH = str(BASE_DIR / "submission_soucek.zip")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Hyperparameters
PREF_MODEL_STEPS = 1000    
PREF_LR = 1e-4             
EXTRACT_STEPS = 100        
EXTRACT_LR = 0.05           
FORGE_STEPS = 100           
FORGE_LR = 0.05             
LAMBDA_QUALITY = 8.0        
IMG_SIZE = 256              

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

normalize = transforms.Normalize([0.5]*3, [0.5]*3)
to_tensor = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    normalize
])

def tensor_to_pil(t):
    t = t.detach().cpu().squeeze(0)
    t = (t * 0.5 + 0.5).clamp(0, 1)
    return transforms.ToPILImage()(t)

def load_tensor(path):
    return to_tensor(Image.open(path).convert("RGB")).unsqueeze(0).to(DEVICE)


class PreferenceModel(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        # Freeze early layers, fine-tune later ones
        for i, child in enumerate(self.features.children()):
            if i < 6:
                for p in child.parameters():
                    p.requires_grad = False
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = x * 0.5 + 0.5  # denormalize to [0,1] for ResNet
        return self.head(self.features(x)).squeeze(-1)

def generate_synthetic_artifact(batch_size, h=IMG_SIZE, w=IMG_SIZE):
    """
    Generate random synthetic artifacts mimicking watermark patterns.
    Souček trains on a mix of:
    1. Sinusoidal/Fourier patterns (common in frequency-domain watermarks)
    2. Random noise patterns (spread-spectrum watermarks)
    3. Structured grid patterns (block-based watermarks)
    """
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
            sign = np.random.choice([-1, 1], size=(h // block_size + 1,
                                                    w // block_size + 1, 3))
            for i in range(h):
                for j in range(w):
                    pattern[i, j] = sign[i // block_size, j // block_size]

        strength = np.random.uniform(0.01, 0.05)
        artifacts.append(pattern * strength)

    return torch.tensor(np.stack(artifacts)).permute(0, 3, 1, 2).float().to(DEVICE)

def train_preference_model(clean_tensors, steps=1000, lr=1e-4):
    model = PreferenceModel().to(DEVICE)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )

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

        margin = 0.5
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


def extract_watermark(source_tensors, preference_model, steps=100, lr=0.05):
    deltas = []
    preference_model.eval()

    for x_wm in source_tensors:
        x_wm = x_wm.to(DEVICE)

        # Optimize delta to make x_wm - delta score LOW (look clean)
        delta = nn.Parameter(torch.zeros_like(x_wm))
        optimizer = optim.Adam([delta], lr=lr)

        for step in range(steps):
            optimizer.zero_grad()
            x_clean_est = (x_wm - delta).clamp(-1, 1)
            # Minimize preference score (make it look clean)
            loss = preference_model(x_clean_est).mean()
            # Regularize: delta should be small
            loss = loss + 0.1 * (delta ** 2).mean()
            loss.backward()
            optimizer.step()

        deltas.append(delta.detach().cpu())

    # Aggregate: average watermark estimate across all source images
    watermark = torch.stack(deltas).mean(dim=0)
    return watermark  # (1, 3, H, W)


def forge_watermark(target_tensor, watermark_pattern, preference_model,
                    steps=100, lr=0.05, lambda_quality=8.0):
    preference_model.eval()
    target = target_tensor.to(DEVICE)

    # Initialize with averaging-based watermark
    delta = nn.Parameter(watermark_pattern.to(DEVICE).clone())
    optimizer = optim.Adam([delta], lr=lr)

    for step in range(steps):
        optimizer.zero_grad()

        forged = (target + delta).clamp(-1, 1)

        # Maximize preference score (make it look watermarked)
        pref_loss = -preference_model(forged).mean()

        # Quality loss: stay close to clean target
        quality_loss = ((forged - target) ** 2).mean()

        loss = pref_loss + lambda_quality * quality_loss
        loss.backward()
        optimizer.step()

    forged_final = (target + delta).clamp(-1, 1).detach()
    return forged_final

# Unzip if needed
if not (DATASET_DIR / "watermarked_sources").exists():
    print(f"Unzipping...")
    with zipfile.ZipFile(str(BASE_DIR / "Dataset.zip"), "r") as z:
        z.extractall(str(BASE_DIR))
else:
    print("Dataset already extracted.")

TEMP_OUT_DIR.mkdir(exist_ok=True)
clean_dir = DATASET_DIR / "clean_targets"

# Load all clean images as tensors (for preference model training)
print("Loading clean images...")
clean_paths = sorted(clean_dir.glob("*.png"))
clean_tensors = [load_tensor(p) for p in clean_paths]
print(f"Loaded {len(clean_tensors)} clean images.")

print("\n" + "="*60)
print("Stage 1: Training preference model (Soucek et al.)")
print("="*60)
preference_model = train_preference_model(
    clean_tensors, steps=PREF_MODEL_STEPS, lr=PREF_LR
)

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

    # Load source watermarked images
    source_tensors = [load_tensor(p) for p in source_paths]

    print(f"  Stage 2: Extracting watermark from {len(source_tensors)} source images...")
    watermark_pattern = extract_watermark(
        source_tensors, preference_model,
        steps=EXTRACT_STEPS, lr=EXTRACT_LR
    )
    print(f"  Watermark pattern std: {watermark_pattern.std().item():.4f}")

    print(f"  Stage 3: Forging onto target images...")
    for number in range(target_start, target_stop + 1):
        target_path = clean_dir / f"{number}.png"
        target_orig = Image.open(target_path).convert("RGB")
        orig_size = target_orig.size  # (W, H) original size

        target_tensor = load_tensor(target_path)

        forged_tensor = forge_watermark(
            target_tensor, watermark_pattern, preference_model,
            steps=FORGE_STEPS, lr=FORGE_LR, lambda_quality=LAMBDA_QUALITY
        )

        # Convert back to PIL and resize to original dimensions
        forged_pil = tensor_to_pil(forged_tensor)
        forged_pil = forged_pil.resize(orig_size, Image.BILINEAR)

        forged_pil.save(TEMP_OUT_DIR / f"{number}.png")
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