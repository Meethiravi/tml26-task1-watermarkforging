

import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

DATASET_DIR   = Path("/home/atml_team060/tml26_task4/Dataset")
TARGET_DIR    = DATASET_DIR / "clean_targets"
SCHEMES_JSON  = Path("/home/atml_team060/tml26_task4/schemes.json")
FALLBACK_V5   = Path("/home/atml_team060/tml26_task4/submission_temp_v5")
FALLBACK_PREF = Path("/home/atml_team060/tml26_task4/submission_temp_pref")
OUT_DIR       = Path("/home/atml_team060/tml26_task4/submission_temp_final")
FILE_PATH     = "/home/atml_team060/tml26_task4/submission_final.zip"

CATEGORIES = [
    ("WM_1", 1, 25),   ("WM_2", 26, 50),  ("WM_3", 51, 75),  ("WM_4", 76, 100),
    ("WM_5", 101, 125),("WM_6", 126, 150),("WM_7", 151, 175),("WM_8", 176, 200),
]

FALLBACK_DIR = {
    "WM_1": FALLBACK_PREF, "WM_2": FALLBACK_PREF, "WM_7": FALLBACK_PREF, "WM_8": FALLBACK_PREF,
    "WM_3": FALLBACK_V5,   "WM_4": FALLBACK_V5,   "WM_5": FALLBACK_V5,   "WM_6": FALLBACK_V5,
}


def load_bgr(p):
    rgb = np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8)
    return rgb[:, :, ::-1].copy()


def bgr_to_pil(bgr):
    return Image.fromarray(bgr[:, :, ::-1].copy())


_riva_loaded = False
def encode_imwatermark(bgr, method, bits):
    from imwatermark import WatermarkEncoder
    global _riva_loaded
    if method == "rivaGan" and not _riva_loaded:
        WatermarkEncoder.loadModel()
        _riva_loaded = True
    enc = WatermarkEncoder()
    enc.set_watermark("bits", bits)
    return enc.encode(bgr, method)


def encode_trustmark(pil_img, message):
    from trustmark import TrustMark
    tm = TrustMark(verbose=False)
    return tm.encode(pil_img, message)


def main():
    schemes = json.loads(SCHEMES_JSON.read_text()) if SCHEMES_JSON.exists() else {}
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    n = 0
    for wm, lo, hi in CATEGORIES:
        info = schemes.get(wm)
        for num in range(lo, hi + 1):
            out_path = OUT_DIR / f"{num}.png"
            if info is None:
                fb_path = FALLBACK_DIR[wm] / f"{num}.png"
                if not fb_path.exists():
                    raise FileNotFoundError(
                        f"No re-embed for {wm} and no fallback at {fb_path}. "
                        f"Run forge_v5.py build / forge_preference.py build first.")
                Image.open(fb_path).convert("RGB").save(out_path)
            elif info["family"] == "imwatermark":
                clean_bgr = load_bgr(TARGET_DIR / f"{num}.png")
                wm_bgr = encode_imwatermark(clean_bgr, info["method"], info["bits"])
                bgr_to_pil(wm_bgr).save(out_path)
            elif info["family"] == "trustmark":
                clean_img = Image.open(TARGET_DIR / f"{num}.png").convert("RGB")
                encode_trustmark(clean_img, info["message"]).save(out_path)
            else:
                raise ValueError(f"Unknown family {info['family']!r} for {wm}")
            n += 1

        if info is None:
            print(f"[{wm}] fallback ({FALLBACK_DIR[wm].name}): images {lo}-{hi}")
        elif info["family"] == "imwatermark":
            print(f"[{wm}] re-embedded ({info['method']}/{info['length']}bits): images {lo}-{hi}")
        else:
            print(f"[{wm}] re-embedded (trustmark): images {lo}-{hi}")

    if n != 200:
        print(f"[WARNING] processed {n} images, expected 200")

    with zipfile.ZipFile(FILE_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        for img in sorted(OUT_DIR.glob("*.png"), key=lambda x: int(x.stem)):
            zf.write(img, arcname=img.name)
    print(f"Saved {n} -> {FILE_PATH}")


if __name__ == "__main__":
    main()
