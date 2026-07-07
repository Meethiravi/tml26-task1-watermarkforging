

import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter, median_filter

DATASET_DIR  = Path("/home/atml_team060/tml26_task4/Dataset")
TEMP_OUT_DIR = Path("/home/atml_team060/tml26_task4/submission_temp_v5")
FILE_PATH    = "/home/atml_team060/tml26_task4/submission_v5.zip"
SOURCE_ROOT  = DATASET_DIR / "watermarked_sources"
TARGET_DIR   = DATASET_DIR / "clean_targets"

CATEGORIES = [
    ("WM_1", 1, 25),   ("WM_2", 26, 50),  ("WM_3", 51, 75),  ("WM_4", 76, 100),
    ("WM_5", 101, 125),("WM_6", 126, 150),("WM_7", 151, 175),("WM_8", 176, 200),
]

# per-batch denoiser: bm3d everywhere (cleanest cross-batch); wiener optional
DENOISER = {wm: "bm3d" for wm, _, _ in CATEGORIES}


ALPHA = {
    "WM_1": 6.0, "WM_2": 6.0,
    "WM_3": 6.0, "WM_4": 6.0,
    "WM_5": 3.0,
    "WM_6": 1.5,     # hard cap, do not move
    "WM_7": 6.0, "WM_8": 4.0,
}
QUALITY_FLOOR = 0.90                 # target min S_qlt (LPIPS ~0.0132)
CLEAN_BIAS_N  = 80
BM3D_SIGMA    = 3.0


def load(p):
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.float32)


def resize_to(arr, hw):
    th, tw = hw
    if arr.shape[:2] == (th, tw):
        return arr
    out = np.zeros((th, tw, arr.shape[2]), np.float32)
    for c in range(arr.shape[2]):
        im = Image.fromarray(arr[:, :, c], mode="F")
        out[:, :, c] = np.asarray(im.resize((tw, th), Image.BICUBIC), np.float32)
    return out


# ---- denoisers --------------------------------------------------------------
def wiener_denoise(img, window=5):
    out = np.empty_like(img)
    for c in range(3):
        y = img[:, :, c]
        m = uniform_filter(y, window)
        v = np.maximum(uniform_filter(y * y, window) - m * m, 0)
        nv = v.mean()
        g = np.maximum(v - nv, 0) / (np.maximum(v - nv, 0) + nv + 1e-8)
        out[:, :, c] = m + g * (y - m)
    return out


def bm3d_denoise(img, sigma=BM3D_SIGMA):
    import bm3d
    out = np.empty_like(img)
    for c in range(3):
        out[:, :, c] = bm3d.bm3d(img[:, :, c], sigma_psd=sigma)
    return out


DENOISE_FN = {"bm3d": bm3d_denoise, "wiener": wiener_denoise}


# ---- estimator --------------------------------------------------------------
_native_clean, _clean_bias = {}, {}


def native_clean(hw):
    if hw in _native_clean:
        return _native_clean[hw]
    th, tw = hw
    imgs = []
    for p in sorted(TARGET_DIR.glob("*.png"), key=lambda x: int(x.stem)):
        with Image.open(p) as im:
            w, h = im.size
        if (h, w) == (th, tw):
            imgs.append(load(p))
    if not imgs:
        imgs = [resize_to(load(p), hw) for p in TARGET_DIR.glob("*.png")]
    _native_clean[hw] = imgs
    return imgs


def clean_bias(hw, dn_name):
    key = (hw, dn_name)
    if key in _clean_bias:
        return _clean_bias[key]
    fn = DENOISE_FN[dn_name]
    imgs = native_clean(hw)[:CLEAN_BIAS_N]
    _clean_bias[key] = np.mean([c - fn(c) for c in imgs], axis=0)
    return _clean_bias[key]


def batch_stack(wm):
    paths = sorted((SOURCE_ROOT / wm).glob("*.png"))
    h, w = load(paths[0]).shape[:2]
    return np.stack([resize_to(load(p), (h, w)) for p in paths]), (h, w)


def estimate(wm):
    dn_name = DENOISER[wm]
    fn = DENOISE_FN[dn_name]
    stack, hw = batch_stack(wm)
    res = np.mean([img - fn(img) for img in stack], axis=0)
    return res - clean_bias(hw, dn_name)


# ---- build ------------------------------------------------------------------
def build(scale=1.0, verbose=True):
    TEMP_OUT_DIR.mkdir(exist_ok=True)
    n = 0
    for wm, lo, hi in CATEGORIES:
        w_est = estimate(wm)
        a = ALPHA[wm] * scale
        if verbose:
            print(f"{wm}: dn={DENOISER[wm]} alpha={a:.2f} "
                  f"wstd={w_est.std():.3f}")
        for num in range(lo, hi + 1):
            tgt = load(TARGET_DIR / f"{num}.png")
            w = resize_to(w_est, tgt.shape[:2])
            forged = np.clip(tgt + a * w, 0, 255).astype(np.uint8)
            Image.fromarray(forged).save(TEMP_OUT_DIR / f"{num}.png")
            n += 1
    if n != 200:
        print(f"[WARNING] {n} images, expected 200")
    with zipfile.ZipFile(FILE_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        for img in sorted(TEMP_OUT_DIR.glob("*.png"), key=lambda x: int(x.stem)):
            zf.write(img, arcname=img.name)
    print(f"Saved {n} -> {FILE_PATH}   (global scale {scale})")


# ---- lpips tools ------------------------------------------------------------
def _lpips():
    import torch, lpips as L
    fn = L.LPIPS(net="alex").eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    fn = fn.to(dev)
    to_t = lambda a: (torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)
                      / 127.5 - 1).to(dev)
    return torch, fn, to_t


def eval_lpips():
    torch, fn, to_t = _lpips()
    print(f"{'batch':6} {'LPIPS':>9} {'S_qlt':>8}")
    for wm, lo, hi in CATEGORIES:
        vals = []
        for num in range(lo, hi + 1):
            c = load(TARGET_DIR / f"{num}.png")
            f = load(TEMP_OUT_DIR / f"{num}.png")
            with torch.no_grad():
                vals.append(fn(to_t(c), to_t(f)).item())
        m = float(np.mean(vals))
        print(f"{wm:6} {m:9.4f} {np.exp(-8*m):8.4f}")


def sweep(alphas=(0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)):
    torch, fn, to_t = _lpips()
    print(f"quality floor S_qlt >= {QUALITY_FLOOR} "
          f"(LPIPS <= {-np.log(QUALITY_FLOOR)/8:.4f})")
    print("max alpha inside floor per batch:")
    for wm, lo, hi in CATEGORIES:
        w_est = estimate(wm)
        nums = list(range(lo, hi + 1))[:5]
        best_a = 0.0
        row = []
        for a in alphas:
            vals = []
            for num in nums:
                tgt = load(TARGET_DIR / f"{num}.png")
                forged = np.clip(tgt + a * resize_to(w_est, tgt.shape[:2]),
                                 0, 255).astype(np.float32)
                with torch.no_grad():
                    vals.append(fn(to_t(tgt), to_t(forged)).item())
            sq = float(np.exp(-8 * np.mean(vals)))
            row.append((a, sq))
            if sq >= QUALITY_FLOOR:
                best_a = a
        tbl = "  ".join(f"a{a}:{sq:.2f}" for a, sq in row)
        print(f"{wm:6} max_a={best_a:<4}  [{tbl}]")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "build"
    scale = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    if mode == "sweep":
        sweep()
    elif mode == "eval":
        build(scale); eval_lpips()
    else:
        build(scale)