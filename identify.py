

import numpy as np
from pathlib import Path
from PIL import Image
import pywt
from scipy.fftpack import dct

SOURCE_ROOT = Path("/home/atml_team060/tml26_task4/Dataset/watermarked_sources")
CATEGORIES = [f"WM_{i}" for i in range(1, 9)]
LENGTHS = [32, 48, 64, 100]

def dct2(b):
    return dct(dct(b.T, norm="ortho").T, norm="ortho")

def read_bits(gray, length):

    LL, _ = pywt.dwt2(gray.astype(np.float32), "haar")
    h, w = LL.shape
    bits = []
    for by in range(0, h - 7, 8):
        for bx in range(0, w - 7, 8):
            if len(bits) >= length:
                break
            block = LL[by:by+8, bx:bx+8]
            c = dct2(block)[3, 3]
            bits.append(int(c > 0))
    return np.array(bits[:length]) if len(bits) >= length else None

def decode_batch(wm, length):
    paths = sorted((SOURCE_ROOT / wm).glob("*.png"))
    bits = []
    for p in paths:
        rgb = np.asarray(Image.open(p).convert("RGB"), np.float32)
        gray = rgb @ np.array([0.299, 0.587, 0.114], np.float32)  # BGR->Y equiv
        b = read_bits(gray, length)
        if b is not None:
            bits.append(b)
    if len(bits) < 5:
        return None
    B = np.stack(bits)
    majority = (B.mean(0) > 0.5).astype(int)
    return float(np.mean(B == majority)), majority

print(f"{'batch':6} {'len':>4} {'agree':>7}")
for wm in CATEGORIES:
    best = (0.0, None, None)
    for L in LENGTHS:
        r = decode_batch(wm, L)
        if r and r[0] > best[0]:
            best = (r[0], L, r[1])
    agree, L, msg = best
    flag = "  <-- IDENTIFIED" if agree > 0.85 else ""
    print(f"{wm:6} {str(L):>4} {agree:7.3f}{flag}")
    if agree > 0.85:
        print("   msg:", "".join(map(str, msg[:24])), "...")