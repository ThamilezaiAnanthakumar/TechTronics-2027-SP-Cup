# Phase 0 reproduction log (README Section 15.3)

All runs MATLAB R2026b Update 1 unless stated. Raw outputs under `D:\SPCup Data\work\` (not in git); pooled tables copied to `outputs/repro/`.

## UBFC2, Muftee's pipeline (Viola-Jones + KLT, POS/combined), 42 subjects, 2379 windows
| Run | Code | Pooled MAE | RMSE | r | Median per subject |
|---|---|---|---|---|---|
| Original (22 Nov, R2024b) | uncommitted, = f9d4155 | 2.165 | 6.491 | 0.936 | 1.19 |
| Unseeded re-run | f9d4155 | 2.187 | 6.572 | 0.935 | 1.19 |
| **Seeded (new reference)** | **318426d** | **2.166** | **6.489** | **0.936** | **1.192** |

The unseeded pipeline was not repeatable: `estgeotform2d` (MSAC) draws random samples and nothing set the RNG. Two identical R2024b runs of subject 27 differed in 107/1431 frames (up to 4 px). Since 318426d each video starts from `rng(0,'twister')`; two extractions under different caller RNG states are bit-identical. All 12 method x ROI pairs of the seeded run are within 0.1 bpm of the original. Worst subjects unchanged: 15 (24.81), 49 (5.50), 27 (5.34), 41 (5.12).

## Face module (Sulojan, RetinaFace), `run_pipeline(DebugVideos=false)`, code 921910d (unseeded, as-is)
Held-out subjects 3, 4, 12, 26, 31, 34 — **all accuracy numbers reproduce exactly** (MAE/RMSE/r/SUC@5, target found, ID switches):
Baseline 1.28, QVGA 1.42, MultiFace 1.28, Masked 1.45, Stress 1.48 bpm.
All 8 subjects: identical except Stress (README 1.51 / RMSE 2.86 → re-run 1.44 / 2.63), attributed to the same unseeded `estgeotform2d` in `FaceTracker.m`.
ms/frame (21–42) is not comparable: four MATLAB jobs shared 4 cores. Re-measure on an idle machine.

## AiPEX, all 88 videos (aggregates only; per-video results of AiPEX-test not inspected)
**Muftee `run_aipex` @ f9d4155 — reproduces exactly** (max |MAE difference| over all method x ROI x subset rows: 5e-13 bpm; 88 videos, 82.9 min, serial).
Video-level HR = peak of the averaged window spectra. Best: POS / right cheek MAE 18.06, RMSE 25.52, 21.6 % within 5 bpm (India 20.10, Sierra Leone 16.01). CHROM / combined 18.12; POS / combined 21.07 (India 21.12, Sierra Leone 21.01).

**Thamil `aipex_hr` (code = f629042)**: pending.
