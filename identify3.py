import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

DATASET_DIR  = Path("/home/atml_team060/tml26_task4/Dataset")
SOURCE_ROOT  = DATASET_DIR / "watermarked_sources"
CLEAN_DIR    = DATASET_DIR / "clean_targets"
OUT_JSON     = Path("/home/atml_team060/tml26_task4/schemes.json")

CATEGORIES = [
    ("WM_1", 1, 25),   ("WM_2", 26, 50),  ("WM_3", 51, 75),  ("WM_4", 76, 100),
    ("WM_5", 101, 125),("WM_6", 126, 150),("WM_7", 151, 175),("WM_8", 176, 200),
]

IW_METHODS = ["dwtDct", "dwtDctSvd"]     # + rivaGan handled separately (fixed 32 bits, needs model load)
LENGTHS    = [8, 16, 24, 32, 48, 64, 96, 128, 256]
N_TRAIN    = 15                          # of 25 source images
SIG_MARGIN = 0.10                        # test agreement must clear same-size clean baseline by this much
MIN_TEST_AGREE = 0.75                    # and clear this absolute floor
SEED = 0


# ----------------------------------------------------------------------
# invisible-watermark bit-vector decoding
# ----------------------------------------------------------------------
def load_bgr(p):
    rgb = np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8)
    return rgb[:, :, ::-1].copy()


_riva_loaded = False
_riva_unavailable = False
def decode_bits(bgr, method, length):
    from imwatermark import WatermarkDecoder
    global _riva_loaded, _riva_unavailable
    if method == "rivaGan":
        if _riva_unavailable:
            return None
        if not _riva_loaded:
            try:
                WatermarkDecoder.loadModel()
                _riva_loaded = True
            except Exception as e:
                print(f"    [rivaGan unavailable: {e}]")
                _riva_unavailable = True
                return None
    try:
        d = WatermarkDecoder("bits", length)
        bits = np.asarray(d.decode(bgr, method), dtype=np.int32).ravel()
        return bits[:length] if len(bits) >= length else None
    except Exception:
        return None


def held_out_agreement(paths, method, length, n_train=N_TRAIN, seed=SEED):
    """Split paths into train/test, build majority-vote message from train,
    return (mean test agreement, majority message) or (None, None)."""
    idx = np.random.default_rng(seed).permutation(len(paths))
    train_idx, test_idx = idx[:n_train], idx[n_train:]

    train_bits = [b for b in (decode_bits(load_bgr(paths[i]), method, length) for i in train_idx)
                  if b is not None]
    if len(train_bits) < 3:
        return None, None
    majority = (np.mean(np.stack(train_bits), axis=0) >= 0.5).astype(int)

    test_scores = [float(np.mean(b == majority))
                   for b in (decode_bits(load_bgr(paths[i]), method, length) for i in test_idx)
                   if b is not None]
    if not test_scores:
        return None, None
    return float(np.mean(test_scores)), majority


def full_majority(paths, method, length):
    bits = [b for b in (decode_bits(load_bgr(p), method, length) for p in paths) if b is not None]
    if len(bits) < 3:
        return None
    return (np.mean(np.stack(bits), axis=0) >= 0.5).astype(int).tolist()


# ----------------------------------------------------------------------
# TrustMark: string-message API with its own error correction, so
# consistency is measured as "fraction of images decoding to the same
# string", not bit-level majority vote.
# ----------------------------------------------------------------------
def trustmark_consistency(paths):
    try:
        from trustmark import TrustMark
    except ImportError:
        return None
    tm = TrustMark(verbose=False)
    decoded = []
    for p in paths:
        try:
            img = Image.open(p).convert("RGB")
            msg, present, _conf = tm.decode(img)
        except Exception:
            continue
        if present and msg:
            decoded.append(msg)
    if len(decoded) < 3:
        return 0.0, None, len(decoded)
    top_msg, top_count = Counter(decoded).most_common(1)[0]
    return top_count / len(paths), top_msg, len(decoded)


# ----------------------------------------------------------------------
# Per-batch analysis
# ----------------------------------------------------------------------
def analyze_batch(wm, lo, hi):
    src_paths = sorted((SOURCE_ROOT / wm).glob("*.png"))
    clean_paths = [CLEAN_DIR / f"{i}.png" for i in range(lo, hi + 1)]  # same-size baseline

    best = None  # (margin, method, length, test_agree, baseline)
    for method in IW_METHODS + ["rivaGan"]:
        lengths = [32] if method == "rivaGan" else LENGTHS
        for L in lengths:
            test_agree, _ = held_out_agreement(src_paths, method, L)
            if test_agree is None:
                continue
            baseline_agree, _ = held_out_agreement(clean_paths, method, L)
            if baseline_agree is None:
                continue
            margin = test_agree - baseline_agree
            if test_agree >= MIN_TEST_AGREE and margin >= SIG_MARGIN:
                if best is None or margin > best[0]:
                    best = (margin, method, L, test_agree, baseline_agree)

    tm_result = trustmark_consistency(src_paths)
    tm_candidate = None
    if tm_result is not None:
        frac, msg, n_present = tm_result
        baseline_frac, _, _ = trustmark_consistency(clean_paths) or (0.0, None, 0)
        margin = frac - baseline_frac
        if frac >= MIN_TEST_AGREE and margin >= SIG_MARGIN and msg is not None:
            tm_candidate = (margin, frac, baseline_frac, msg)

    result = {}
    if best is not None and (tm_candidate is None or best[0] >= tm_candidate[0]):
        margin, method, L, test_agree, baseline_agree = best
        bits = full_majority(src_paths, method, L)
        result = {"family": "imwatermark", "method": method, "length": L,
                  "bits": bits, "test_agree": test_agree, "baseline": baseline_agree,
                  "margin": margin}
    elif tm_candidate is not None:
        margin, frac, baseline_frac, msg = tm_candidate
        result = {"family": "trustmark", "message": msg, "test_agree": frac,
                  "baseline": baseline_frac, "margin": margin}
    return result or None


def main():
    schemes = {}
    print("=" * 78)
    print("CORRECTED IDENTIFICATION SWEEP  (baseline-relative, held-out validated)")
    print("=" * 78)
    for wm, lo, hi in CATEGORIES:
        print(f"\n[{wm}]")
        res = analyze_batch(wm, lo, hi)
        if res is None:
            print("  not identified (no method cleared baseline+margin on held-out test)")
            continue
        if res["family"] == "imwatermark":
            print(f"  IDENTIFIED: {res['method']}/{res['length']}bits  "
                  f"test_agree={res['test_agree']:.3f}  baseline={res['baseline']:.3f}  "
                  f"margin={res['margin']:.3f}")
        else:
            print(f"  IDENTIFIED: trustmark  test_agree={res['test_agree']:.3f}  "
                  f"baseline={res['baseline']:.3f}  margin={res['margin']:.3f}  "
                  f"message={res['message'][:32]}...")
        schemes[wm] = res

    OUT_JSON.write_text(json.dumps(schemes, indent=2))
    print(f"\nWrote {len(schemes)}/8 identified scheme(s) -> {OUT_JSON}")
    print("Identified:", ", ".join(schemes.keys()) if schemes else "(none)")
    print("Unidentified batches will fall back to existing pipeline output in reembed.py.")


if __name__ == "__main__":
    main()
