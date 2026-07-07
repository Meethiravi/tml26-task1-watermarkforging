# TML 2026 — Assignment 4: Watermark Forgery Attack

Final leaderboard score: **0.611576**.

This README explains only how to reproduce our **best** leaderboard result.
Every other script in the repo corresponds to an experiment described in the
report; see "Repository contents" at the bottom.

## Approach (one paragraph)

Our best submission is a **hybrid**. We first identify which of the eight
watermarking schemes are standard, recoverable schemes (`identify3.py`), then
**re-embed** the recovered message on those batches with the scheme's own
encoder (`reembed.py`) for near-perfect bit accuracy at tiny distortion. The
batches we cannot identify fall back to a BM3D residual copy attack
(`forgev4.py`). Five of the eight batches are identified and re-embedded; three
use the copy-attack fallback.

## Environment

```bash
pip install invisible-watermark opencv-python-headless trustmark onnxruntime \
            bm3d lpips torch torchvision --break-system-packages

# trustmark pulls in full opencv-python (needs libGL.so.1, absent in the slim
# CUDA container) as a transitive dep and silently shadows opencv-python-headless.
# Uninstall it and force headless back, or cv2 import fails:
pip uninstall -y opencv-python
pip install --force-reinstall --no-deps opencv-python-headless --break-system-packages
```

`invisible-watermark` imports as `imwatermark`. `rivaGan` and `TrustMark`
download small pretrained weights on first use (needs network access from the
job). The `Dataset/` folder must contain `watermarked_sources/WM_1..WM_8/`
(25 PNGs each) and `clean_targets/1.png..200.png`.

## Reproduce the 0.612 submission (run in this order)

The final zip is assembled by `reembed.py`, which consumes the identified schemes
plus a pre-built BM3D fallback directory. So build the fallback first, then
identify, then assemble.

```bash
# 1. BM3D residual copy attack -> submission_temp_v5/  (fallback for WM_3/4/5)
python forgev4.py build

# 2. Identify recoverable schemes -> schemes.json
#    (writes WM_1 dwtDct, WM_2 rivaGan, WM_6 dwtDct, WM_7 TrustMark Q/ECC,
#     WM_8 TrustMark P/no-ECC). Sweeps TrustMark (model_type, use_ECC) configs.
python identify3.py

# 3. Assemble final zip: re-embed identified batches with their native encoder,
#    pull unidentified batches (WM_3/4/5) from submission_temp_v5/.
#    -> submission_final.zip
python reembed.py

# 4. Validate the zip (200 files named 1.png..200.png, no subfolders)
#    (point check.py's ZIP_PATH at submission_final.zip first)
python check.py

# 5. Submit (set API_KEY and FILE_PATH=submission_final.zip in submission.py)
python submission.py
```

Note: `reembed.py`'s fallback map also references `submission_temp_pref/`
(the preference-model output) for WM_1/2/7/8, but in the final run all four are
identified and re-embedded, so that directory is not used. You only need to run
`forge_preference.py build` if you want to reproduce that experiment separately.

### Per-batch outcome of the final pipeline

| Batch | Resolution | Final source |
|-------|-----------|--------------|
| WM_1  | 256×256   | re-embed `dwtDct` (256-bit) |
| WM_2  | 256×256   | re-embed `rivaGan` (32-bit) |
| WM_3  | 256×256   | BM3D copy attack (`forgev4.py`) |
| WM_4  | 256×256   | BM3D copy attack (`forgev4.py`) |
| WM_5  | 128×128   | BM3D copy attack (`forgev4.py`) |
| WM_6  | 256×256   | re-embed `dwtDct` (16-bit) |
| WM_7  | 512×512   | re-embed `TrustMark` (model Q, use_ECC=True) |
| WM_8  | 512×512   | re-embed `TrustMark` (model P, use_ECC=False) |

The re-embed vs fallback split is driven entirely by which batches
`identify3.py` writes into `schemes.json`; `reembed.py` re-embeds those and pulls
the rest from `submission_temp_v5/`.

## Notes on reproducibility

- `identify3.py` uses a fixed seed (`SEED=0`) for the 15/10 train/test split, so
  `schemes.json` is deterministic.
- TrustMark is swept over `(model_type, use_ECC)` configurations with
  `MODE="binary"`; WM_8 is only detected under model P with `use_ECC=False`
  (decoding a raw payload with ECC enabled detects nothing).
- We cannot measure detection (`S_det`) locally — there is no detector — so the
  `forgev4.py` alpha values were calibrated to an LPIPS quality budget
  (`sweep` mode) and confirmed on the leaderboard.

## Repository contents (experiments referenced in the report)

- `kutter_iteration1.py` — DC-subtraction average-residual baseline (0.257).
- `yang_iteration1.py`, `yang_resize.py`, `yang_iter4.py` — mean-difference
  estimator + high-pass cleanup + NVF mask + LPIPS calibration (0.291).
- `forgev3.py` — mean/median/high-pass estimator ablation with per-batch alpha.
- `kutter_iteration2.py`, `kutter_yang.py` — Kutter copy attack (Wiener + NVF).
- `forgev4.py` — final BM3D residual copy attack.
- `forge_preference.py` — preference-direction feature-space attack (prepared
  fallback for no-residual batches; not used in the final 0.612 run).
- `wmcopier_iter1.py`, `wmcopier_iter2.py` — WMCopier diffusion reproduction.
- `soucek.py` — preference-model reproduction (early version).
- `identify.py`, `identify2.py` — earlier identification probes (buggy: absolute
  threshold, no held-out validation — kept to document the correction).
- `identify3.py` — corrected identification (baseline-relative margin +
  held-out split + TrustMark model/ECC sweep), writes `schemes.json`.
- `reembed.py` — assembles the final hybrid submission.
- `submission.py` — leaderboard uploader.
