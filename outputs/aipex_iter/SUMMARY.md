# AiPEX iteration — summary (README Section 8.5), 26 Nov 2026

**Protocol.** AiPEX-dev only (62 videos: 31 India, 31 Sierra Leone); 5-fold CV stratified by country (seed 2027); nested CV for multi-valued hypotheses; keep rule = CV MAE better, paired-bootstrap 95 % CI of the gain excludes 0 (5,000 resamples), UBFC dev pooled MAE not worse by > 0.1 bpm, India–Sierra Leone gap not wider by > 0.5 bpm. AiPEX-test (26) untouched. 19 experiment rows in `log.csv` (E000–E018, incl. 2 invalid runs that are kept and labelled).

**Stopped:** stopping rule — ladder exhausted except item 14 (BKF, awaiting operator approval: no reference code available, would be our own implementation) and item 15 (reporting rule, not a pipeline change); 5 consecutive hypotheses without a significant gain (E012–E017), E018 not testable.

## Best configuration (dev CV MAE 8.39 bpm, 95 % CI [6.41, 10.45])
Pre-cropped input adapter (no face detection) · skin mask (face_config.json) · pixel-count-weighted Kalman denoising (Thamil) · POS 1.6 s · smoothness-priors detrend λ = 100 · **Butterworth band-pass order 2**, 0.7–3.0 Hz · 10 s windows, 1 s step · **per-region spectra, SNR-weighted spectral fusion of the 3 crops (top-3)** · video HR = peak of the averaged fused spectrum.
India 8.22 / Sierra Leone 8.43 (gap 0.21, was 5.90) · within 5 bpm 44 % · bias +1.88 bpm.

## Waterfall (kept changes)
| Step | Change | Dev CV MAE | Gain | Gain 95 % CI | UBFC dev pooled MAE |
|---|---|---|---|---|---|
| E000 | baseline (integrated pipeline: skin, pixel-weighted RGB of 3 crops, Kalman, POS, order 4) | 14.29 | – | – | 3.495 |
| E003 | SNR-weighted spectral fusion of the 3 crops (top-3) | 10.99 | −3.30 | [0.01, 6.74] | 2.453 |
| E010 | band-pass order 2 (was 4) | **8.39** | −2.60 | [1.14, 4.60] | 2.509 |

## Rejected / not significant
E001 timing (files are exactly 15 fps; no constant scale factor) · E002 all pixels (15.07) · E004 median of window peaks (12.80, n.s.; bias +12 confirms noise-driven window peaks) · E005 window 15/20/30 s (worse) · E006 harmonic summation (worse) · E007 CHROM (worse) · E008 POS window 1.2/2.0 s (worse) · E009 detrend λ 50/300 (n.s.) · E011 Kalman off (worse by 3.3 → Kalman stays) · E012 band 0.7–2.5 Hz (n.s.) · E013 top-k 1/2 (n.s.) · E014 Viterbi video estimator (7.74, n.s.) · E015 trimmed mean (8.09, n.s.) · E016 texture mask + trimmed mean (7.89, n.s.) · E017 texture + Viterbi (7.61, n.s.) · E018 motion suppression (not testable: crop phase correlation unreliable).

## Where the remaining error is
38/62 dev videos within ±10 % of the reference. Misses are mostly over-estimates (ratio 1.2–1.6) on the shorter recordings (MAE 11.2 in the shortest tercile vs 5.8–7.7). No half/double-rate locks.

## Status vs targets
Must (< 5.8, beat BKF): **not met** (8.39). Several non-significant changes point the same way (E014–E017: 7.6–8.1); with 62 videos the CI half-width is ~2 bpm, so gains below ~1.5 bpm cannot be confirmed on dev alone.

## Next (needs operator)
- BKF (Prakash & Tucker 2018) as an own implementation (ladder 14).
- Freeze at 28 Nov 18:00, then one run on AiPEX-test with the E010 configuration.
