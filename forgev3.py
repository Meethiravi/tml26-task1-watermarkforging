

import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

# ----------------------------- CONFIG ----------------------------------------
DATASET_DIR  = Path("/home/atml_team060/tml26_task4/Dataset")
TEMP_OUT_DIR = Path("/home/atml_team060/tml26_task4/submission_temp")
FILE_PATH    = "/home/atml_team060/tml26_task4/submission_new.zip"

SOURCE_ROOT = DATASET_DIR / "watermarked_sources"
TARGET_DIR  = DATASET_DIR / "clean_targets"

CATEGORIES = [
    ("WM_1", 1, 25),   ("WM_2", 26, 50),  ("WM_3", 51, 75),  ("WM_4", 76, 100),
    ("WM_5", 101, 125),("WM_6", 126, 150),("WM_7", 151, 175),("WM_8", 176, 200),
]

METHOD = "mean_sub"          # "mean_sub" | "median_sub" | "denoise"
BLUR_RADIUS = 2.0            # only used by "denoise"



ALPHA = {
    "WM_1": 1.0, "WM_2": 1.0, "WM_3": 1.0, "WM_4": 1.0,
    "WM_5": 1.0, "WM_6": 1.0, "WM_7": 0.6, "WM_8": 0.4,
}
# -----------------------------------------------------------------------------


def load(p):
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.float32)


def resize_arr(arr, target_hw):
    """Resize an (H,W,3) float array, preserving negative residual values."""
    th, tw = target_hw
    if arr.shape[:2] == (th, tw):
        return arr
    out = np.zeros((th, tw, arr.shape[2]), np.float32)
    for c in range(arr.shape[2]):
        im = Image.fromarray(arr[:, :, c], mode="F")
        out[:, :, c] = np.asarray(im.resize((tw, th), Image.BICUBIC), np.float32)
    return out


def highpass(img, radius):
    pil = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    blur = np.asarray(pil.filter(ImageFilter.GaussianBlur(radius=radius)), np.float32)
    return img - blur


# ---- clean reference --------------------------------------------------------

_clean_cache = {}

def clean_reference(size_hw, stat="mean"):
   
    key = (size_hw, stat)
    if key in _clean_cache:
        return _clean_cache[key]
    th, tw = size_hw
    paths = sorted(TARGET_DIR.glob("*.png"), key=lambda x: int(x.stem))

    native = []
    for p in paths:
        with Image.open(p) as im:
            w, h = im.size                  # PIL .size is (width, height)
        if (h, w) == (th, tw):
            native.append(load(p))

    if native:                              # exact-size matches: no resizing
        stack = np.stack(native)
    else:                                   # fallback: resize everything
        stack = np.stack([resize_arr(load(p), size_hw) for p in paths])

    ref = stack.mean(0) if stat == "mean" else np.median(stack, 0)
    _clean_cache[key] = ref
    print(f"    clean_ref @ {th}x{tw}: {stack.shape[0]} native images")
    return ref


def estimate_watermark(source_wm):
   
    paths = sorted((SOURCE_ROOT / source_wm).glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"No source images in {SOURCE_ROOT / source_wm}")
    h, w = load(paths[0]).shape[:2]                       # working size
    stack = np.stack([resize_arr(load(p), (h, w)) for p in paths])  # (25,H,W,3)

    if METHOD == "mean_sub":
        w_est = stack.mean(0) - clean_reference((h, w), "mean")
    elif METHOD == "median_sub":
        w_est = np.median(stack, 0) - clean_reference((h, w), "median")
    elif METHOD == "denoise":
        w_est = np.mean([highpass(img, BLUR_RADIUS) for img in stack], axis=0)
    else:
        raise ValueError(f"Unknown METHOD: {METHOD}")
    return w_est


def build_submission(verbose=True):
    TEMP_OUT_DIR.mkdir(exist_ok=True)
    n = 0
    for source_wm, lo, hi in CATEGORIES:
        w_est = estimate_watermark(source_wm)
        a = ALPHA[source_wm]
        if verbose:
            print(f"{source_wm}: alpha={a}  w_est std={w_est.std():.3f} "
                  f"min={w_est.min():.1f} max={w_est.max():.1f}")
        for num in range(lo, hi + 1):
            tgt = load(TARGET_DIR / f"{num}.png")
            w = resize_arr(w_est, tgt.shape[:2])
            forged = np.clip(tgt + a * w, 0, 255).astype(np.uint8)
            Image.fromarray(forged).save(TEMP_OUT_DIR / f"{num}.png")
            n += 1

    if n != 200:
        print(f"[WARNING] processed {n} images, expected 200 -- will be rejected!")

    with zipfile.ZipFile(FILE_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        for img in sorted(TEMP_OUT_DIR.glob("*.png"), key=lambda x: int(x.stem)):
            zf.write(img, arcname=img.name)
    print(f"Saved {n} images -> {FILE_PATH}")


def eval_lpips():
    
    try:
        import torch
        import lpips as lpips_lib
    except ImportError:
        print("Install first:  pip install lpips torch")
        return

    loss_fn = lpips_lib.LPIPS(net="alex")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loss_fn = loss_fn.to(device).eval()

    def to_t(arr):  # (H,W,3) 0..255 -> (1,3,H,W) in [-1,1]
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
        return t.to(device)

    print(f"{'batch':6} {'mean LPIPS':>11} {'S_qlt':>8}")
    for source_wm, lo, hi in CATEGORIES:
        vals = []
        for num in range(lo, hi + 1):
            clean  = load(TARGET_DIR / f"{num}.png")
            forged = load(TEMP_OUT_DIR / f"{num}.png")
            with torch.no_grad():
                d = loss_fn(to_t(clean), to_t(forged)).item()
            vals.append(d)
        m = float(np.mean(vals))
        print(f"{source_wm:6} {m:11.4f} {np.exp(-8*m):8.4f}")


def sweep_alpha(alphas=(0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.5)):
   
    try:
        import torch
        import lpips as lpips_lib
    except ImportError:
        print("Install first:  pip install lpips torch")
        return

    loss_fn = lpips_lib.LPIPS(net="alex")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loss_fn = loss_fn.to(device).eval()

    def to_t(arr):
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
        return t.to(device)

    SAMPLES_PER_BATCH = 5            # a few targets per batch, for speed

    header = "batch  " + "  ".join(f"a={a:<4}" for a in alphas)
    lpips_table = {}

    print("\nLPIPS by alpha (lower is better):")
    print(header)
    for source_wm, lo, hi in CATEGORIES:
        w_est = estimate_watermark(source_wm)
        sample_nums = list(range(lo, hi + 1))[:SAMPLES_PER_BATCH]
        row = []
        for a in alphas:
            vals = []
            for num in sample_nums:
                tgt = load(TARGET_DIR / f"{num}.png")
                w = resize_arr(w_est, tgt.shape[:2])
                forged = np.clip(tgt + a * w, 0, 255).astype(np.float32)
                with torch.no_grad():
                    d = loss_fn(to_t(tgt), to_t(forged)).item()
                vals.append(d)
            row.append(float(np.mean(vals)))
        lpips_table[source_wm] = row
        print(f"{source_wm:6} " + "  ".join(f"{v:6.3f}" for v in row))

    print("\nS_qlt by alpha (higher is better; aim ~0.70-0.85):")
    print(header)
    for source_wm, _, _ in CATEGORIES:
        row = lpips_table[source_wm]
        print(f"{source_wm:6} " + "  ".join(f"{np.exp(-8*v):6.3f}" for v in row))


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "build"

    if mode == "sweep":
        sweep_alpha()                 # find good per-batch alpha (interactive job)
    elif mode == "eval":
        build_submission()            # build, then measure quality of the build
        eval_lpips()
    else:
        build_submission()            # default: just build the submission zip