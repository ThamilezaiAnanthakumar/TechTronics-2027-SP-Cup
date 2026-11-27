# Python pipeline: C2 skin-tone evaluation

Runs the **same HR pipeline on every dataset** and reports the error per skin-tone group (ITA).
This is a Python port of the MATLAB face module and baseline HR, so it runs without MATLAB R2025a+.

## Requirements

Python 3.11 with `opencv-python`, `mediapipe`, `scikit-image`, `scipy`, `pandas`, `openpyxl`.

Datasets are read from the folder that contains the repo (`D:/SPCUP`), as downloaded:
`UBFC-rPPG/`, `MPU rppg/`, `CLBP-400_Samples/`, `rppg_biases-main/`, `SRMC_N/`.
Change it with `--data-root` or the `FACEVITALS_DATA` environment variable.
UBFC subjects in `data/raw/UBFC/DATASET_2/` are also picked up.

## Run

```bash
# C2 evaluation: 5 independent processes (start each one separately), then merge
python src/python/run_c2.py --shard 1/5      # ... up to --shard 5/5
python src/python/run_c2.py --analyze-only   # merge shard tables + statistics

# quick test
python src/python/run_c2.py --datasets UBFC --max-frames 600

# skin-tone survey only (fast, includes SRMC-N, which has no ground truth), with thumbnails
python src/python/ita_survey.py
```

MediaPipe on Windows fails when one process spawns several workers that load it at once, so
parallel runs use independent processes (`--shard`), not a process pool. Traces are cached in
`outputs/traces_py/`; `--rerun` ignores the cache.

## What it does

| Step | Same as MATLAB | Difference |
|---|---|---|
| Face box | | MediaPipe full-range detector (RetinaFace needs R2025a+); box extended 15 % upwards to cover the forehead; largest face only (no C3 multi-face tracking, no KLT); exponential box smoothing instead of One-Euro |
| ROI | Face-box fractions from `config/face_config.json` | |
| Skin mask | YCbCr box with MATLAB `rgb2ycbcr` coefficients, same thresholds | |
| Occlusion | `rgb_face` = pixel-weighted mean of non-occluded regions | Absolute skin-ratio rule only |
| HR | `sanity_hr.m`, `ppg_hr.m`, `spectral_hr.m` ported line by line: 30 Hz resampling on real timestamps, POS, Butterworth 0.7–3 Hz, 10 s windows every 1 s, periodogram peak | |
| ITA | | `rgb2lab` (sRGB, D65) on the same skin pixels as HR, every 30th frame; median L\*, b\* per region; region L\* > 15 below the median is dropped as hair/shadow |

### Background-reference rejection (`--method pos_bg`)

Light flicker, camera exposure oscillation and coloured lighting (e.g. an RGB keyboard) modulate
the whole scene. When the modulation has a colour component, POS does not remove it and its
frequency can win the spectral peak (CLBP Subject068: a constant 2.0 Hz = 120 bpm rhythm in face,
wall and hijab; MPU: ~55–59 bpm keyboard lighting in face, shirt and curtain).

1. **Background trace** (`face.background_rgb`): per frame, mean RGB of everything outside the face
   box enlarged 1.6×, with skin pixels (hands, neck: they carry a pulse) and clipped pixels removed,
   computed on a ¼-size copy of the frame. Saved as `rgb_bg` with `bg_npix` and `face_box`.
2. **Rejection** (`rppg.sanity_hr_bg`, `mode="regress"`): face and background go through the same
   resampling, POS and band-pass. In each 10 s window the face pulse signal is least-squares fitted
   by the background signals (background POS + its 3 normalised, band-passed channels) and the fit
   is subtracted before the periodogram. The background has no pulse, so only shared illumination
   is removed.

`mode="spectral"` (divide the face spectrum by the background spectrum) is kept for comparison:
10 s windows make peaks ±10 bpm wide and the flanks of a strong flicker peak survive.
Every subject row has both results: `mae`, `suc5_pct`, ... (baseline) and `bg_mae`, `bg_suc5_pct`, ...

```bash
python src/python/run_c2.py --analyze-only --method compare   # baseline vs background reference
python src/python/run_c2.py --analyze-only --method pos_bg    # skin-tone statistics with the fix
```

AiPEX clips are crops without background, so `pos_bg` equals the baseline there.

### SNR-weighted region fusion (`--method pos_snr`, `pos_snr_bg`)

The baseline averages the regions' pixels (`rgb_face`), so a large noisy region outweighs a small
clean one. `rppg.sanity_hr_snr` instead runs POS on every region separately and, in each 10 s window:

1. computes each region's periodogram and its SNR (de Haan & Jeanne 2013): power within ±0.1 Hz of
   the in-band peak and ±0.2 Hz of its first harmonic, divided by the rest of the 0.7–3 Hz band;
2. normalises each spectrum to unit in-band power and weights it by SNR^`snr_power` (1.0);
3. sums the weighted spectra; HR = peak of the sum.

Weights are recomputed per window, so a region that goes bad for a few seconds (shadow, hand,
mask edge) only loses weight for those seconds. `pos_snr_bg` first regresses the background signals
out of every region (as in `pos_bg`); it equals `pos_snr` on AiPEX. Rows contain `snr_mae`,
`snrbg_mae`, ... and `snr_w_<region>`: the mean share of each region in the fused spectrum.

**Control `pos_eq`:** the same per-region POS and spectrum fusion with equal weights (SNR⁰ = 1).
`pos_eq` vs `pos` measures the gain from fusing region *spectra* instead of averaging *pixels*;
`pos_snr` vs `pos_eq` measures what the SNR weights add on top. Columns: `eq_mae`, `eq_suc5_pct`, ...

Limitation: SNR is measured around each region's own peak, so a region locked onto a strong
artefact can look "clean". The background regression in `pos_snr_bg` reduces that risk.

Reference HR:
- **UBFC:** from the contact PPG, same windows and method as the estimate. Subject MAE = mean over its windows.
- **MPU:** the oximeter's HR column, averaged inside each 10 s window. HR computed from the exported
  PPG column disagreed with the device HR by 11–45 bpm on recordings 8, 9 and 10, so it is not used.
- **AiPEX, CLBP:** one value per recording (AiPEX: Masimo mean; CLBP: spot reading). Estimate = median of the window HRs.
- **SRMC-N:** none. ITA only.

AiPEX clips are pre-cropped regions (no full face), so the skin mask is applied to the whole crop
and face detection is skipped. That is a different path from the other datasets, which is why the
regression below includes a dataset term.

## Statistics (`config/c2_config.json`)

- **Groups:** six ITA classes (Very light > 55°, Light 41–55°, Intermediate 28–41°, Tan 10–28°,
  Brown −30–10°, Dark < −30°, roughly Fitzpatrick I–VI) and three merged groups (I–II, III–IV, V–VI)
  because some classes have very few subjects.
- **Acceptance:** a group passes if |group MAE − population MAE| ≤ `tau_bpm` (2.0 bpm). τ was fixed
  before any result was computed.
- **95 % CI:** bootstrap over subjects.
- **Regression:** subject MAE ~ ITA + dataset, so a harder dataset is not mistaken for a skin-tone effect.
- **Within-dataset:** Spearman correlation between ITA and MAE in each dataset with ≥ 5 subjects.

Outputs in `outputs/tables/`: `c2_subjects.csv`, `c2_windows.csv`, `c2_groups.csv`,
`c2_regression.csv`, `c2_within_dataset.csv`.

## Results: AiPEX rppg_biases (C2)

88 recordings (India 44, Sierra Leone 44). Each recording is three pre-cropped ROI clips
(forehead 60×30, cheeks 25×25 px, upscaled to 200×200), 15 fps, compressed. Reference: one
Masimo mean HR per recording; estimate = median of the 10 s window HRs. AiPEX has no
background, so `pos_bg` = `pos` and `pos_snr_bg` = `pos_snr` here.

### Overall

| Method | MAE | Median | Within 5 bpm | Bias |
|---|---|---|---|---|
| `pos` baseline (pixels averaged) | 16.01 | 16.50 | 13.6 % | +15.8 |
| `pos_eq` region spectra, equal weights | 14.06 | – | 19.3 % | – |
| **`pos_snr` region spectra, SNR weights** | **8.20** | **5.77** | **43.2 %** | **+5.7** |

- `pos_snr` vs `pos`: 72 recordings better, 10 worse (> 1 bpm); Wilcoxon p < 0.001.
- `pos_snr` vs `pos_eq`: 69 better, 15 worse; p < 0.001. The gain comes from the **SNR weights**,
  not only from fusing spectra. Weights average ≈ ⅓ per region over a recording but differ per
  window: each window trusts the region that shows a clear peak in that window.
- AiPEX paper (Dasari et al. 2021) on the same videos: POS 14.0, CHROM 11.6, BKF 5.8 MAE.

### By country

| Country | n | `pos` MAE | `pos_eq` MAE | `pos_snr` MAE | `pos_snr` median | `pos_snr` within 5 bpm | `pos_snr` bias |
|---|---|---|---|---|---|---|---|
| India | 44 | 15.76 | 12.89 | **7.87** | 5.65 | 45.5 % | +4.5 |
| Sierra Leone | 44 | 16.25 | 15.22 | **8.54** | 6.03 | 40.9 % | +7.0 |

### By skin tone (ITA group), with the C2 acceptance test

Population MAE over the 85 recordings with an ITA value; a group passes if
|group MAE − population MAE| ≤ τ = 2 bpm (fixed in `config/c2_config.json` before any result).

| ITA group | n | `pos` MAE (dev.) | `pos_snr` MAE (dev.) |
|---|---|---|---|
| I–II (ITA > 41°) | 15 | 17.71 (+1.65) pass | 9.33 (+1.21) pass |
| III–IV (10–41°) | 14 | 21.72 (+5.66) **fail** | 9.09 (+0.97) pass |
| V–VI (≤ 10°) | 56 | 14.20 (−1.86) pass | **7.56 (−0.56) pass** |
| Population | 85 | 16.06 | 8.12 |

Per country (`pos` → `pos_snr` MAE): India I–II (n = 2) 8.12 → 2.15, III–IV (8) 23.95 → 9.38,
V–VI (32) 14.29 → 7.46; Sierra Leone I–II (13) 19.19 → 10.43, III–IV (6) 18.75 → 8.71,
V–VI (24) 14.09 → 7.69.

Only recordings with consistent ITA and no white-balance warning (27): V–VI (n = 21)
10.07 → 5.59 MAE, 57 % within 5 bpm (I–II and III–IV have 3 each).

### Conclusion

With SNR-weighted region fusion, every ITA group is within ±1.2 bpm of the population MAE and
the darkest group (V–VI, 56 recordings) has the lowest error; India and Sierra Leone are within
0.7 bpm of each other. **No evidence of a performance loss for darker skin.** Caveats:
- Absolute error is still high (8 bpm, 43 % within 5 bpm): tiny, compressed, 15 fps crops limit
  every skin tone.
- ITA from raw video depends on exposure and white balance. Several Sierra Leone recordings read
  "Very light" because of colour casts (b\* ≈ 1–2), so the light groups are partly mislabelled
  lower-quality recordings; the dark group is the reliable part.
- One mean HR per recording: only the recording-level estimate is validated, not tracking.

## Results: other datasets (mean subject MAE, bpm)

| Dataset | n | `pos` | `pos_bg` | `pos_eq` | `pos_snr` | `pos_snr_bg` |
|---|---|---|---|---|---|---|
| UBFC | 12 | 2.20 | 1.62 | 1.94 | 2.10 | 1.76 |
| CLBP | 5 | 10.30 | 1.86 | 10.30 | 10.34 | 2.43 |

- Python port vs MATLAB (`pos`, UBFC subjects 1/3/4): 1.53 / 3.23 / 2.61 vs 1.47 / 2.84 / 2.38.
- UBFC subject48: 12 windows locked at ~50 bpm (lighting); 8.64 → 1.39 with `pos_bg`.
- CLBP Subject068: a constant 2.0 Hz (120 bpm) lighting rhythm in face, wall and hijab; 42.97 →
  1.00 with `pos_bg`. SNR weighting alone cannot fix it (all regions share the flicker).
- MPU-rPPG (5): ~20 bpm for every method and every ITA. Subjects type/game at a desk; an RGB
  keyboard lights the face (~55–59 bpm rhythm in face, shirt and curtain); `Output.mp4` is
  1.5 Mbps. Not used for skin-tone conclusions. Reference = oximeter HR column (HR from the
  exported PPG column disagreed by 11–45 bpm on recordings 8, 9, 10).
- ITA in MPU is biased low (underexposure, warm cast: b\* ≈ 22), in UBFC high (b\* ≈ 6).

## Known limitations

- ITA depends on camera exposure and white balance. Inside one dataset it ranks subjects reliably;
  across datasets the absolute class can be off (UBFC b\* ≈ 6 pushes everyone to "Very light",
  MPU b\* ≈ 22 pushes East-Asian subjects to "Brown"). `wb_warning` flags b\* < 3.
- Few full-face subjects with dark skin. Most of the V–VI evidence comes from AiPEX.
