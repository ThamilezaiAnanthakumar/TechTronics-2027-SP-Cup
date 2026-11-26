# Ablations (README Section 8.3) — dev splits only

## 1. ROI set (UBFC2 dev: subjects 1, 3, 5, 8–16; 662 windows of 10 s), 26 Nov 2026
`tools/ubfc_patch_ablation.m`; same signal chain for all rows (POS, smoothness-priors detrend λ=100, Butterworth order 2, 10 s windows, per-window SNR-weighted spectral fusion), same PPG reference windows (seeded `run_ubfc_all`).

| ROI set | Front end | Fusion | Pooled MAE | RMSE | r | SUC@5 | Coverage |
|---|---|---|---|---|---|---|---|
| 3 box fractions (forehead, cheeks) | Viola-Jones + KLT | top-3 | 2.51 | 6.71 | 0.929 | 92.7 % | 100 % |
| **10 anatomical patches (Li et al. 2026)** | **MediaPipe 478 landmarks, 1-Euro** | **top-5** | **1.37** | **2.21** | **0.992** | **97.3 %** | 100 % |
| 10 anatomical patches + Kalman (npix) | same | top-5 | 1.42 | 2.34 | 0.991 | 97.1 % | 100 % |

Per subject (box3 → anat10): subject 15 13.78 → 1.81; subject 14 2.57 → 2.05; subject 12 1.14 → 0.98; all others within ±0.1. `outputs/ablation_roi/per_subject.csv`.
Note: the anatomical front end also removes the Viola-Jones 1 Hz re-detection steps (docs/FAILURES.md), so the gain combines patch placement and front-end stability. Kalman gives no extra gain on clean UBFC video (as before). Row 3 (canonical-frame patches from 5 landmarks, README 4.2a) is not run: RetinaFace returns no landmarks (docs/API_NOTES.md).
