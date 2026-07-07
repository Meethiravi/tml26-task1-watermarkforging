

import json
import itertools
from pathlib import Path

import numpy as np
from PIL import Image

DATASET_DIR = Path("/home/atml_team060/tml26_task4/Dataset")
SOURCE_ROOT = DATASET_DIR / "watermarked_sources"
OUT_JSON    = Path("/home/atml_team060/tml26_task4/schemes.json")
WM_FOLDERS  = [f"WM_{i}" for i in range(1, 9)]

# fine grid of candidate message lengths
LENGTHS = [8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 96, 128, 256]
METHODS = ["dwtDct", "dwtDctSvd"]
IDENTIFIED_THRESHOLD = 0.85


def load_bgr(p):
    rgb = np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8)
    return rgb[:, :, ::-1].copy()


def decode_all(decoder_factory, method, paths):
    """Return (rows, n_ok, n_err): decoded bit arrays, and crash counts."""
    rows, n_ok, n_err = [], 0, 0
    for p in paths:
        try:
            d = decoder_factory()
            bits = d.decode(load_bgr(p), method)
            rows.append(np.asarray(bits).ravel())
            n_ok += 1
        except Exception:
            rows.append(None)
            n_err += 1
    return rows, n_ok, n_err


def consistency(rows):
    good = [r for r in rows if r is not None]
    if len(good) < 2:
        return 0.0, None
    L = min(len(r) for r in good)
    good = [np.asarray(r[:L], dtype=np.int32) for r in good]
    agrees = [np.mean(a == b) for a, b in itertools.combinations(good, 2)]
    consensus = (np.mean(np.stack(good), axis=0) >= 0.5).astype(int)
    return float(np.mean(agrees)), consensus.tolist()


def main():
    try:
        from imwatermark import WatermarkDecoder
    except ImportError:
        print("Install first: pip install invisible-watermark opencv-python-headless")
        return

    schemes = {}
    print("=" * 72)
    print("FINE SWEEP  (consistency: ~0.5 random, >0.85 identified)")
    print("=" * 72)

    for wm in WM_FOLDERS:
        paths = sorted((SOURCE_ROOT / wm).glob("*.png"))
        if not paths:
            print(f"\n{wm}: no images"); continue
        cands = []
        for method in METHODS:
            for L in LENGTHS:
                rows, n_ok, n_err = decode_all(
                    lambda L=L: WatermarkDecoder("bits", L), method, paths)
                cons, consensus = consistency(rows)
                cands.append((cons, method, L, consensus, n_ok, n_err))

        # best = highest consistency; tie-break toward SHORTER length (true len)
        cands.sort(key=lambda c: (round(c[0], 3), -c[2]), reverse=True)
        best = cands[0]
        cons, method, L, consensus, n_ok, n_err = best

        # Among near-best, pick the smallest length (the fundamental, not a multiple)
        near = [c for c in cands if c[0] >= cons - 0.01 and c[0] > 0.7]
        if near:
            near.sort(key=lambda c: c[2])       # smallest length first
            cons, method, L, consensus, n_ok, n_err = near[0]

        ident = cons >= IDENTIFIED_THRESHOLD
        print(f"\n{wm}: best={method}/{L}bits consistency={cons:.3f} "
              f"(ok={n_ok} err={n_err}) {'<== IDENTIFIED' if ident else ''}")
        if n_err == len(paths):
            print(f"    NOTE: every decode crashed -- scheme/params wrong for this size")
        if ident:
            print(f"    message ({L} bits): {''.join(map(str, consensus[:L]))}")
            schemes[wm] = {"method": method, "length": L, "bits": consensus[:L]}

    with open(OUT_JSON, "w") as f:
        json.dump(schemes, f, indent=2)
    print(f"\nWrote {len(schemes)} identified scheme(s) -> {OUT_JSON}")
    print("Identified:", ", ".join(schemes.keys()) if schemes else "(none)")


if __name__ == "__main__":
    main()