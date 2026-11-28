# Python pipeline: C2 skin-tone evaluation

Runs the **same HR pipeline on every dataset** and reports the error per skin-tone group (ITA).
The face module is a line-by-line port of the MATLAB one (`src/face`), so it runs without
MATLAB R2025a+. HR methods compared (`--method`):

| Method | Steps |
|---|---|
| `pos` | baseline: skin pixels of all regions averaged → POS → spectral peak (`sanity_hr.m`) |
| `pos_bg` | `pos` with background-reference regression (lighting flicker) |
| `pos_kf` | Kalman denoising of the RGB trace before POS (team method, `kalman_denoise.m`) |
| `pos_eq` | POS per region, spectra fused with equal weights (control) |
| `pos_snr` | POS per region, spectra fused with per-window SNR weights |
| `pos_snr_bg` | `pos_snr` with background regression per region |
| `pos_kf_snr` | Kalman per region → POS per region → SNR fusion (frozen CP3 core) |
| `pos_kf_snr_bg` | `pos_kf_snr` with background regression |

Tested on SRMC-N in the test scripts (not yet a `--method` of `run_c2.py`), see
[SRMC-N tests](#srmc-n-tests-real-world-clinic-videos):

| Method | Steps |
|---|---|
| **`pos` + background + SNR-gated motion** | `pos_bg` in every window; in windows with pulse SNR < 0 dB the head-motion signals of the face box are regressed out as well. **Best on SRMC-N** |
| SNR-gated motion (`posmg`) | `pos` + the motion regression in low-SNR windows only |
| Viterbi tracking (`*_vit`) | port of the team's `viterbiHR.m` on the window spectra |
| vitiligo-aware (AiPEX) | detect depigmented skin, split each region into pigmented / depigmented traces, Kalman + SNR fusion |

## Requirements

Python 3.11 with `opencv-python`, `torch` + `torchvision` (CPU is enough), `mediapipe`,
`scikit-image`, `scipy`, `pandas`, `openpyxl`.

RetinaFace weights (`mobilenet0.25_Final.pth`, 1.8 MB, WIDER FACE) are downloaded on first use to
`models/retinaface/` (gitignored), checked against a fixed SHA-256 and loaded with
`torch.load(weights_only=True)`.

Datasets are read from the folder that contains the repo (`D:/SPCUP`), as downloaded:
`UBFC-rPPG/`, `MPU rppg/`, `CLBP-400_Samples/`, `rppg_biases-main/`, `SRMC_N/`.
Change it with `--data-root` or the `FACEVITALS_DATA` environment variable.
UBFC subjects in `data/raw/UBFC/DATASET_2/` are also picked up.

## Run

```bash
# one dataset at a time (one process; safest on a laptop with ~12 GB RAM)
python -u src/python/run_c2.py --datasets AiPEX       # ~12 min, traces cached afterwards
python -u src/python/run_c2.py --datasets UBFC        # ~10 min with the tracker
python -u src/python/run_c2.py --datasets CLBP        # 4K video, ~1 h with the tracker

# merge all dataset tables and report
python src/python/run_c2.py --analyze-only --method compare        # all methods side by side
python src/python/run_c2.py --analyze-only --method pos_kf_snr_bg  # C2 tables for one method

# parallel runs: independent processes, one per shard, then --analyze-only
python src/python/run_c2.py --shard 1/4      # ... up to --shard 4/4

# quick test
python src/python/run_c2.py --datasets UBFC --subjects subject1 --max-frames 600

# skin-tone survey only (fast), with thumbnails
python src/python/ita_survey.py

# SRMC-N: track and evaluate a subset by metadata (traces cached for the test scripts below)
python -u src/python/run_c2.py --datasets SRMC --where protocol=Resting                 # Normal + Abnormal
python -u src/python/run_c2.py --datasets SRMC --where group=Abnormal protocol=Resting  # ~1 min per clip

# SRMC-N / UBFC tests on the cached traces (minutes, no video is reprocessed)
python src/python/motion_test.py                     # motion regression variants, motion vs error
python src/python/viterbi_test.py [--group Abnormal] # Viterbi tracking + SNR-gated motion
python src/python/harmonic_test.py                  # harmonic check
python src/python/snrmedian_test.py                  # SNR-weighted median aggregation
python src/python/artefact_test.py                   # sources of the low-frequency artefact
python src/python/vitiligo_test.py --subjects India_video9  # synthetic vitiligo on AiPEX (~7 min)
```

`--where KEY=VALUE` filters on subject metadata (SRMC: `group`, `protocol`, `lighting`,
`skin_tone`, ...); the table name gets `_key-value`. With more than 3 `--subjects` the name gets
`_<n>subjects`.

MediaPipe on Windows fails when one process spawns several workers that load it at once, so
parallel runs use independent processes (`--shard`), not a process pool. Traces are cached in
`outputs/traces_py/`; `--rerun` ignores the cache.

## What it does

### Face module (`--face-module tracker`, default)

A line-by-line port of the MATLAB face module (`src/face/`), same algorithm and the same
parameters (`face_params.m` defaults + `config/face_config.json`):

| MATLAB | Python | Note |
|---|---|---|
| `faceDetector("small-network")` (RetinaFace) | `facevitals/retinaface.py`: RetinaFace MobileNet-0.25 | Same detector and backbone, public WIDER FACE weights (MathWorks' trained model cannot be loaded in Python, so boxes can differ slightly). Threshold 0.5, MinSize 10 px, CPU |
| `FaceTracker.m` | `facevitals/tracker.py` `FaceTracker` | Detection every `det.every` frames (`everyLost` when lost), search window after KLT failure, target choice (largest-central / `trk.targetBox`), re-acquisition rules, re-anchoring at IoU < 0.75 |
| `configureKalmanFilter("ConstantVelocity", …)` | `KalmanCV` | State [x vx y vy], P0 = diag(200, 50), Q = diag(100, 25), R = 100 |
| `assignDetectionsToTracks` | `assign_detections_to_tracks` | Hungarian on 1 − IoU, padded with cost of non-assignment 0.5, gate at IoU < 0.1 |
| `detectMinEigenFeatures` + `vision.PointTracker` | `cv2.goodFeaturesToTrack` + `cv2.calcOpticalFlowPyrLK` | Min-eigenvalue corners (300, quality 0.01, inset box), 31×31 window, 3 levels, forward-backward error ≤ 2 px |
| `estgeotform2d("similarity")` | `cv2.estimateAffinePartial2D` (RANSAC, 3 px) | Scale sanity check 0.5–2, point refresh below 50 % |
| `OneEuroFilter.m`, `roi_polygons.m`, `apply_tform.m`, `box_iou.m`, `clip_box.m` | same names in `tracker.py` | Coordinates 0-based |
| `process_video.m`, `roi_skin_stats.m`, `skin_mask.m` | `face.tracked_face_traces`, `face.roi_skin_stats`, `face.skin_mask` | One-Euro on all polygon corners + box; YCbCr skin mask with MATLAB `rgb2ycbcr` coefficients; occlusion = absolute AND relative rule with hysteresis on the EMA skin ratio, or < 10 skin px; NaN frames when lost |
| `sanity_hr.m`, `ppg_hr.m`, `spectral_hr.m` | `facevitals/rppg.py` | 30 Hz resampling on real timestamps, POS, Butterworth 0.7–3 Hz, 10 s windows every 1 s, periodogram peak |

Check on UBFC subject1 (full video): MATLAB README 1.47 bpm, this port 1.41 bpm; the detector ran
on 155 of 1547 frames (every 10th), one face, no false occlusion. Speed ≈ 30 ms/frame at 640×480
on CPU (plus a one-time ~10 s torch import).

Extra outputs for the C2 evaluation (not in MATLAB): the background trace (`rgb_bg`, below) and
ITA samples: `rgb2lab` (sRGB, D65) on the same skin pixels as HR, every 30th frame; median L\*, b\*
per region; a region with L\* > 15 below the median is dropped as hair/shadow.

`--face-module simple` keeps the earlier development version (MediaPipe largest face, box
extended 15 % upwards, exponential smoothing, absolute occlusion rule only). It has its own trace
cache (`outputs/traces_py/` vs `outputs/traces_py/tracker/`) and its tables carry a `_simple`
suffix, so the two are never merged. AiPEX clips are crops, so face detection is skipped and both
modules give identical AiPEX results.

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

### Kalman denoising (`pos_kf`) and the combination (`pos_kf_snr`, `pos_kf_snr_bg`)

`rppg.kalman_denoise` is a port of `src/signal/kalman_denoise.m` from the team branch
`thamilezai_dev` (checked against the MATLAB function on a real AiPEX trace: max difference
3e-14, same NaN handling). Per RGB channel, a random-walk level is tracked with measurement noise
R0·median(npix)/npix(k) (fewer skin pixels → trusted less); the steady-state filter is a
first-order low-pass with −3 dB at 4 Hz, above the HR band. It works **before POS** (cleaner
input, so POS's σ₁/σ₂ weighting is not dominated by frame noise); SNR fusion works **after POS**
(chooses between regions), so they can be stacked:

- `pos_kf`: denoise `rgb_face` (weighted by its non-occluded skin pixels), then baseline POS.
- `pos_kf_snr`: denoise each region (weighted by its own skin pixels), then per-region POS + SNR fusion.
- `pos_kf_snr_bg`: as `pos_kf_snr` with the background regression.

AiPEX crops carry no pixel counts, so there the filter is unweighted (a fixed exponential
smoother). Recording-level datasets also store `*_mae_meanagg`: the MAE when the recording HR is
the **mean** of the window HRs, as in `aipex_hr.m`, next to our **median**.

Limitation: SNR is measured around each region's own peak, so a region locked onto a strong
artefact can look "clean". The background regression in `pos_snr_bg` reduces that risk.

Reference HR:
- **UBFC:** from the contact PPG, same windows and method as the estimate. Subject MAE = mean over its windows.
- **MPU:** the oximeter's HR column, averaged inside each 10 s window. HR computed from the exported
  PPG column disagreed with the device HR by 11–45 bpm on recordings 8, 9 and 10, so it is not used.
- **AiPEX, CLBP:** one value per recording (AiPEX: Masimo mean; CLBP: spot reading). Estimate = median of the window HRs.
- **SRMC-N:** the PR column of `SRMC_N/PatientDetails_Normal.xlsx` (one pulse-oximeter reading per
  recording, sheets "Normal" and "Abnormal"). Estimate = median of the window HRs. The loader reads
  only ID, GENDER, AGE, SKIN TONE, PROTOCOL and PR; the sheets also contain names, which are never read.

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
Masimo mean HR per recording; estimate = **median** of the 10 s window HRs (the *mean* column
reproduces `aipex_hr.m` on `thamilezai_dev`). AiPEX has no background and no face detection, so
`*_bg` = the same method without background, and both face modules give identical results.

### Overall

| Method | MAE (median of windows) | MAE (mean of windows) | Median MAE | Within 5 bpm | Bias |
|---|---|---|---|---|---|
| `pos` baseline (pixels averaged) | 16.01 | 18.56 | 16.50 | 13.6 % | +15.8 |
| `pos_eq` region spectra, equal weights | 14.06 | 16.49 | 13.30 | 19.3 % | – |
| `pos_kf` Kalman denoising (team) | 9.90 | 13.11 | 9.24 | 25.0 % | +9.0 |
| `pos_snr` region spectra, SNR weights | 8.20 | 11.80 | 5.77 | 43.2 % | +5.7 |
| **`pos_kf_snr` Kalman + SNR fusion** | **5.76** | **7.45** | **4.80** | **53.4 %** | **+0.9** |

- **Reproduces the team MATLAB result:** with the mean of windows, `pos` 18.56 and `pos_kf` 13.11
  match `aipex_hr.m` (18.6 → 13.1).
- **The combination beats each part** (> 1 bpm, Wilcoxon): vs `pos_kf` 62 better / 19 worse,
  vs `pos_snr` 56 / 25, vs `pos` 72 / 13; all p < 0.001.
- **Why they stack:** Kalman removes fast frame noise *before* POS (POS's σ₁/σ₂ weighting is not
  dominated by noise); SNR fusion picks the clearest region *after* POS, per window.
- `pos_snr` vs `pos_eq`: 69 better / 15 worse, p < 0.001. The fusion gain comes from the **SNR
  weights**, not only from fusing spectra (weights average ≈ ⅓ per region over a recording but
  differ per window).
- AiPEX paper (Dasari et al. 2021) on the same videos: POS 14.0, CHROM 11.6, BKF 5.8 MAE
  (48 % within 5 bpm).

### By country (MAE, median of windows)

| Country | n | `pos` | `pos_eq` | `pos_kf` | `pos_snr` | **`pos_kf_snr`** | `pos_kf_snr` median / within 5 / bias |
|---|---|---|---|---|---|---|---|
| India | 44 | 15.76 | 12.89 | 9.65 | 7.87 | **6.17** | 4.80 / 54.5 % / +0.4 |
| Sierra Leone | 44 | 16.25 | 15.22 | 10.14 | 8.54 | **5.35** | 4.74 / 52.3 % / +1.3 |

### By skin tone (ITA group), with the C2 acceptance test

Population MAE over the 85 recordings with an ITA value; a group passes if
|group MAE − population MAE| ≤ τ = 2 bpm (fixed in `config/c2_config.json` before any result).
MAE (deviation from population):

| ITA group | n | `pos` | `pos_kf` | `pos_snr` | **`pos_kf_snr`** (95 % CI) |
|---|---|---|---|---|---|
| I–II (ITA > 41°) | 15 | 17.71 (+1.65) | 10.80 (+0.95) | 9.33 (+1.21) | **5.37 (−0.29)** (3.3–7.5) |
| III–IV (10–41°) | 14 | 21.72 (**+5.66 fail**) | 12.15 (**+2.30 fail**) | 9.09 (+0.97) | **4.66 (−1.00)** (2.8–6.8) |
| V–VI (≤ 10°) | 56 | 14.20 (−1.86) | 9.02 (−0.83) | 7.56 (−0.57) | **5.98 (+0.33)** (4.8–7.2) |
| Population | 85 | 16.06 | 9.85 | 8.12 | **5.66** |
| All groups pass? | | no | no | yes | **yes** |

- `pos_kf_snr`: every group within ±1.0 bpm of the population MAE, overlapping 95 % CIs, and no
  trend with skin tone (Spearman ITA vs MAE ρ = −0.03, p = 0.82; baseline ρ = +0.32, p = 0.003).
- Per country, `pos` → `pos_kf_snr`: India I–II (n = 2) 8.12 → 3.53, III–IV (8) 23.95 → 4.78,
  V–VI (32) 14.29 → 6.25; Sierra Leone I–II (13) 19.19 → 5.65, III–IV (6) 18.75 → 4.50,
  V–VI (24) 14.09 → 5.62.
- Only recordings with consistent ITA and no white-balance warning (27): V–VI (n = 21) `pos` 10.07,
  `pos_snr` 5.59, `pos_kf_snr` 6.16 MAE (I–II and III–IV have 3 each). On this subset the
  combination is slightly behind SNR fusion alone.

### Conclusion

**Kalman denoising → per-region POS → SNR fusion** lowers AiPEX MAE from 16.0 to 5.8 bpm (53 %
within 5 bpm, bias +0.9), in both countries, and every ITA group is within ±1 bpm of the
population MAE. **No evidence of a performance loss for darker skin.** Caveats:
- With the combination the darkest group is no longer the lowest-error group (+0.3 bpm); the
  difference is well inside τ and the confidence intervals overlap.
- ITA from raw video depends on exposure and white balance. Several Sierra Leone recordings read
  "Very light" because of colour casts (b\* ≈ 1–2), so the light groups are partly mislabelled
  lower-quality recordings; the dark group (56) is the reliable part.
- AiPEX crops have no pixel counts and no background: the Kalman filter is unweighted there (a fixed
  low-pass) and the background reference is not tested.
- One mean HR per recording: only the recording-level estimate is validated, not tracking.

## Results: other datasets (mean subject MAE, bpm; `--face-module simple`, to be rerun with the tracker)

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

UBFC with the tracker (33 subjects, window MAE vs PPG): `pos` 1.23, `pos_bg` 1.28, `pos_kf` 1.25,
`pos_snr` 1.26, `pos_kf_snr` 1.28, `pos_kf_snr_bg` 1.31. Every subject is below 3 bpm (worst:
subject3 2.99, subject4 2.79, subject33 2.63).

## SRMC-N tests (real-world clinic videos)

Branch `nad_SRMC__tests`. SRMC-N now has a reference (PR reading per recording). All results
below use the **Resting** clips (one clip per person, 20–30 s, normal room lighting):

| Group | Clips with an estimate | Rater skin tone | Notes |
|---|---|---|---|
| Normal (`S_N*`) | 76 (78 listed; S_N43 Dim clips have no PR) | Dark 36, Fair 25, Medium Fair 15 | many clips re-encoded to 480p (~1.6 Mbps) |
| Abnormal (`S_AB*`) | 98 | Medium Fair 62, Dark 32, Fair 4 | patients with abnormal vitals; mostly 1080p–2160p originals |

The Task clips (Bright / Med / Dim lighting) are not evaluated yet. **Tracker setting:**
`config/c2_config.json` has `tracker_max_width: 1280`, so 1440p and 2160p clips were downscaled to
1280 px width before tracking (all SRMC runs).

Errors are |median of window HRs − PR| in bpm, averaged over clips ("MAE"). p values: Wilcoxon
signed-rank vs `pos` (paired), Kruskal–Wallis across rater skin tones.

### Methods added

**1. Motion regression** (`rppg.motion_regressors`, `refs=` argument of `rppg.sanity_hr_snr`).
The tracker already keeps the ROIs on the face (RetinaFace + KLT optical flow + similarity
transform), but head movement still changes the light on the skin. Per frame, four motion signals
come from the face box: centre x and y in face widths, log face width (towards / away from the
camera) and the skin-pixel count. They are band-passed like the pulse and, in each 10 s window,
least-squares fitted to the pulse signal and subtracted (same regression as the background
reference; both can be used together). `rppg.motion_score` = RMS of the band-passed box motion per
window.

**2. SNR gate.** Regressing motion in every window hurts still people: the head moves slightly
with every heartbeat (ballistocardiography), so the motion signals contain the pulse (UBFC
1.23 → 1.68). The gate regresses motion only in windows whose pulse SNR (de Haan & Jeanne, plain
`pos`) is below **0 dB**; other windows keep the plain estimate. On SRMC 70–77 % of windows are
below 0 dB, so in practice the gate mostly protects the cleaner windows.

**3. Final SRMC method: `pos` + background + SNR-gated motion.** Every window: background
regression. Windows with SNR < 0 dB: background + motion regression. Recording HR = median of
windows.

**4. Viterbi tracking** (`rppg.viterbi_hr`): port of the team's `viterbiHR.m` (grid 42–180 bpm in
0.5 bpm, max −3 / +6 bpm/s, λ = 0.1, parabolic refinement); identical to MATLAB R2024b to 1.4e-14.
`sanity_hr_snr(..., return_spectra=True)` returns the per-window fused spectra it needs.

**5. Test scripts** (all on cached traces): `motion_test.py`, `viterbi_test.py`,
`harmonic_test.py`, `snrmedian_test.py`, `artefact_test.py`, `vitiligo_test.py` (AiPEX).

### Baseline on SRMC Normal Resting (76 clips)

| Method (`run_c2.py`) | MAE |
|---|---|
| `pos` | **18.47** (r = −0.01, bias −14) |
| `pos_bg` | 16.71 |
| `pos_kf` | 18.16 |
| `pos_eq` | 19.23 |
| `pos_snr` | 19.94 |
| `pos_kf_snr` (CP3 core) | 21.21 (worse than `pos`, p < 0.001) |
| `pos_kf_snr_bg` | 20.64 |

- **Video quality dominates.** HD originals 11.03 vs re-encoded clips 20.62 (p = 0.003); 1080p
  8.99, 480p 20.97. Dark skin is the worst group within each resolution group (HD: Dark 14.1 vs
  Medium Fair 8.1; re-encoded: 23.8 vs ~18).
- Estimates drift low (47–55 bpm), median pulse SNR −2.3 dB. 11 of 76 clips are within 5 bpm.
- **ITA vs rater labels:** ITA from video is camera-biased. The same rater label differs by 25–60°
  between 480p and HD clips; rater label vs ITA Spearman −0.42. The rater labels are used for the
  skin-tone breakdown.
- **Motion:** error grows with clip motion (Spearman ρ = 0.39; motion terciles still 11.4 /
  medium 18.6 / moving 25.5). Dark-skin clips moved most (median motion 0.006 vs Fair 0.004,
  Medium Fair 0.003).

### Motion handling (SRMC Normal Resting, UBFC as control)

| Method | SRMC Normal | UBFC |
|---|---|---|
| `pos` | 18.47 | 1.23 |
| motion regression, every window | **15.00** (39 better / 22 worse, p = 0.009) | 1.68 ✗ |
| box signals only (no skin count) | 15.23 | 1.58 |
| skin-count signal only | 17.66 | 1.22 |
| regression only where motion > UBFC 95th pct (0.014) | 18.41 | 1.23 |
| reject moving windows (median of still windows) | 18.14 (no real gain) | 1.23 |
| **SNR gate (< 0 dB)** | **15.66** (p = 0.025) | 1.43 |
| `pos` + Viterbi | 19.51 ✗ (18 better / 33 worse, p = 0.024) | 1.26 |
| `pos_kf_snr` + Viterbi | 21.83 | 1.28 |
| SNR gate + Viterbi | 18.07 (n.s.) | 1.26 |

- The UBFC-based motion threshold (95th percentile of UBFC windows) is too high for SRMC: almost
  no SRMC window passes it, so that gate does nothing.
- **Viterbi does not help SRMC:** the wrong peaks are persistent (most windows), not short bursts,
  so Viterbi follows them smoothly and also smooths away the gated motion gains. Agrees with the
  team's E014/E020 (Viterbi off in CP3).
- The motion gate helps Dark-skin clips most (Normal: Dark 21.37 → 15.96, Fair 17.66 → 16.94,
  Medium Fair 12.88 → 12.77), because those clips moved most: part of the Dark-skin gap on SRMC
  was a motion difference.

### Main result: Normal (gate chosen here) and Abnormal (independent test)

The gate threshold and the motion signals were chosen on the Normal group; the Abnormal clips were
tracked afterwards and evaluated unchanged.

| Method | Normal (76) | Abnormal (98) | All (174) | within 5 bpm (all) | UBFC |
|---|---|---|---|---|---|
| `pos` | 18.47 | 16.27 | 17.23 | 43 | 1.23 |
| `pos_kf_snr` | 21.21 ✗ (p = 0.001) | 21.80 ✗ (p < 0.001) | 21.55 ✗ | 33 | 1.28 |
| `pos` + background | 16.71 (p = 0.006) | 13.74 (p = 0.005) | 15.04 (p < 0.001) | 52 | 1.28 |
| `pos` + SNR-gated motion | 15.66 (p = 0.025) | 15.65 (p = 0.92) | 15.65 (p = 0.11) | 49 | 1.43 |
| **`pos` + background + SNR-gated motion** | **15.16** (p = 0.009) | **13.05** (p = 0.004) | **13.97** (p < 0.001) | **61** | **1.29** |

By rater skin tone (all 174 clips):

| Method | Dark (68) | Medium Fair (77) | Fair (29) | Kruskal p |
|---|---|---|---|---|
| `pos` | 19.8 | 14.9 | 17.3 | 0.051 |
| `pos_kf_snr` | 21.7 | 21.5 | 21.2 | 0.84 (equally bad) |
| `pos` + background | 16.7 | 12.7 | 17.5 | **0.016** |
| `pos` + SNR-gated motion | 16.1 | 14.4 | 17.8 | 0.20 |
| **`pos` + background + SNR-gated motion** | **14.5** | **13.2** | **14.6** | **0.63** |

- **Background + gated motion is the best method on both groups** and the most even across skin
  tones (within 1.4 bpm). The two references fix different clips (lighting / camera vs head
  movement) and cover each other's failures.
- Motion regression alone does nothing on Abnormal (p = 0.92): those clips move slightly less
  (median motion 0.004 vs 0.005, p = 0.24); Abnormal Dark clips move less than Normal Dark ones.
- **`pos_kf_snr` (frozen CP3 core) is significantly worse than `pos` on SRMC in both groups.** See
  the SNR analysis below for the reason.
- Caveats: the preference for "background + motion" over "background only" was settled after the
  Abnormal results (it was also best on Normal); a clean confirmation needs unused data (SRMC Task
  clips or the official test set). Abnormal has only 4 Fair clips.

**Age (C6).** No age effect on SRMC.

| Age | Normal: n / `pos` / `pos_kf_snr` / bg + gated motion | Abnormal: n / `pos` / bg + gated motion |
|---|---|---|
| ≤ 29 | 36 / 18.4 / 21.0 / 14.4 | 11 / 22.6 / 19.2 |
| 30–44 | 9 / 14.2 / 18.0 / 12.9 | 13 / 17.2 / 10.8 |
| 45–59 | 11 / 19.9 / 23.0 / 17.8 | 34 / 13.8 / 13.4 |
| 60+ | 20 / 19.7 / 22.2 / 16.1 | 40 / 16.3 / 11.8 |

Normal: Kruskal p = 0.57 (`pos`), 0.85 (`pos_kf_snr`); Spearman age vs error ρ = 0.05–0.07. Age
and skin tone are confounded in Normal (60+ mostly Dark, ≤ 29 mostly Fair).

**ROI cleaning (C6, `--face-module tracker_clean`, Caroppo et al. 2024)** on the 13 Normal clips
with `pos` error < 5 bpm: no effect (`pos` 3.56 → 3.56; `pos_kf_snr` 4.77 → 5.87). It removes
4–12 % of the skin pixels whatever the age.

**Computation.** Per 30 s clip: `pos` 21 ms, + background 22 ms, + background + gated motion
77 ms (both regressions are computed; computing the motion one only in gated windows would halve
it). The motion signals are free (face box). The background trace costs 12.8 ms per 1080p frame and
2.1 ms per 480p frame (on a ¼-size copy; ⅛ would cut it ~4×, untested), about 10 % of the tracking
time on HD. Tracking itself is ~60–120 s per HD clip on CPU.

### Failure analysis (bg + gated motion, 174 clips)

61 clips < 5 bpm, 47 clips 5–15 bpm, 66 clips ≥ 15 bpm (`outputs/tables/srmc_failure_diagnostics.csv`).

| Factor | good (< 5) | bad (≥ 15) | Spearman with error |
|---|---|---|---|
| reference HR is the top of the averaged spectrum (1 = yes) | 1.00 | 0.51 | −0.70 |
| windows within 5 bpm of PR | 45 % | 5 % | −0.66 |
| PR (true HR) | 78 | 90 | +0.34 |
| file bitrate | 12.0 Mbps | 2.8 Mbps | −0.18 |
| face width (px) | 408 | 349 | −0.20 |
| region disagreement (bpm) | 13 | 22 | +0.21 |
| motion | 0.004 | 0.005 | +0.16 |
| age, rater skin tone | – | – | no effect |

- 12 of the 18 clips with PR > 100 fail; in 70 % of the bad clips the estimate is too low.
- 31 of the 66 bad clips are 480p re-encoded (13 of 69 480p clips are good).

| Cause (automatic) | Clips | Notes |
|---|---|---|
| no clear pulse at the reference | 26 | mostly 480p, median PR 95: a C4 limit, the pulse is not in the video |
| reference peak present but a stronger wrong peak wins | 25 | low-frequency artefact (below) |
| pulse found part of the time, median picks the wrong value | 10 | e.g. S_AB15, S_AB16 (nasal tubes), S_AB22 (vitiligo-like spots) |
| stable estimate, reference absent | 3 | S_N36 (74 vs PR 105), S_N82, S_N99: **possible reference mismatch** (PR read at another time); flag, do not tune for them |
| about half the reference | 2 | S_N96, S_N101 |

Case notes:
- **S_N8:** correct (89 vs 88); the gate leaves it alone. **S_N84:** motion + strong backlight;
  stays at ~49 vs PR 94 with every method. **S_N36:** clean and stable at ~75 vs PR 105.
- **S_AB15, S_AB16 (tubes on the face):** the occlusion check reports 100 % skin and 0 %
  occlusion: thin, clear or skin-coloured tubes pass the YCbCr mask and stay in the ROIs (C7 gap).
- **S_AB21:** clean, steadily rising 91 → 99 bpm vs PR 78: likely a reference mismatch.
- **S_AB22 (vitiligo-like spots):** correct 116–118 bpm in early windows, then ~57 (about half);
  the right cheek disagrees with the other regions. The spots are small.
- **S_AB13:** stuck at 52 with `pos`; both the background (89.2) and the motion regression (88.0)
  fix it alone (PR 90): moving head + exposure change seen by both references.

### Ideas tested that did not help

| Idea | All 174 SRMC | UBFC | Verdict |
|---|---|---|---|
| final method (median of windows) | **13.97**, 61 within 5 bpm | 1.29 | – |
| harmonic check, score = P(f) + α·P(2f), 0.7–6 Hz spectrum, α 0.25 | 14.32, 63 within 5 | 1.26 | n.s. |
| harmonic check α 0.5 / 1.0 | 15.36 / 21.58 | 1.27 / 4.38 | worse |
| SNR-weighted median (SNR / SNR²) | 16.73 / 18.73 | 1.74 / 2.01 | worse (p < 0.001) |
| median of the 50 % highest-SNR windows | 17.92 | 1.96 | worse |
| mean of windows | 12.64, **51** within 5 | – | lower mean, fewer accurate clips |
| peak of the averaged spectrum (`pipeline` branch video level) | 17.79 (`pos`: 20.31) | – | worse (p < 0.001) |
| Viterbi | see above | | worse |

- **Harmonic check:** compression removes the weak second harmonic, noise above 3 Hz gives random
  bonuses, and a candidate at HR/2 collects the real pulse as its "harmonic" (pulls estimates to
  half). Agrees with the team's E006.
- **Why SNR weighting fails on SRMC:** windows locked on the low artefact are the *cleanest* by SNR
  (median −1.2 dB) vs correct windows (−1.6 dB); windows above the PR are noisy (−4.2 dB). The
  highest-SNR quarter of windows has median HR 62 bpm. SNR cannot tell a clean pulse from a clean
  artefact. **This is also why `pos_kf_snr` (SNR region fusion) loses to `pos` on SRMC** while it
  wins on AiPEX crops, which have no such artefact. The motion gate uses SNR the other way round
  (acts on noisy windows) and is not affected.

### The low-frequency artefact (~55–65 bpm)

In 123 of 172 clips the strongest non-pulse peak of the raw `pos` spectrum beats the pulse.
Tested sources (`artefact_test.py`, `outputs/tables/artefact_srmc_resting.csv`):

| Candidate | Test | Result |
|---|---|---|
| breathing harmonics (2×, 3×, 4× breathing rate) | breathing rate from slow vertical head motion (median 10/min) | 7 % match = chance (shuffled 7 %) |
| compression keyframes | periodicity of frame-to-frame background brightness jumps | no consistent rhythm, 6 % match |
| band edge (42 bpm) | artefact frequencies | only 12 % below 50 bpm |
| a specific head movement | motion power at the artefact vs at another clip's artefact frequency | barely higher; higher still at lower control frequencies |
| a specific light flicker | same for the background | weak, not specific |
| fixed frequency | spread of artefact frequencies | 45–130 bpm, median ~60 |

Noise power per band (pulse ±8 bpm excluded, 1 = flat):

| bpm | 45–55 | 55–65 | 65–75 | 75–90 | 90–110 | 110–140 | 140–180 |
|---|---|---|---|---|---|---|---|
| face | 1.6 | **1.9** | 1.6 | 1.2 | 0.9 | 0.65 | 0.28 |
| background | 2.0 | **2.2** | 1.6 | 1.2 | 0.9 | 0.73 | 0.29 |

**Conclusion:** the artefact is broadband low-frequency ("1/f") noise of the video (slow sway,
exposure and lighting drift, compression), shaped by the 0.7–3 Hz band-pass into a hump at
55–65 bpm. The background has the same shape, so it is camera / scene noise, not skin. A weak pulse
loses to the largest random bump in the hump. Dominance grows with PR (ρ = +0.41), smaller faces
(−0.29) and lower bitrate (−0.26); it happens in 84 % of 480p vs 56 % of 1080p clips.
**Next step (untested):** whiten the window spectra before peak picking, by the clip's own
background spectrum or by a typical noise shape learned on Normal and tested on Abnormal.

### Vitiligo (C2)

**AiPEX synthetic test** (`vitiligo_test.py`, one subject so far: India_video9, Masimo 80.52).
Depigmented patches are synthesised on real pigmented crops: a smooth random patch field (σ 18 px,
soft edge σ 3 px) at nested coverages 0 / 10 / 25 / 50 % of the skin, moved with the skin by phase
correlation (leaky 0.98), and a linear-light gain exp(k·a + g), a = (0.43, 0.58, 1.0), fitted per
crop to L\* 66, b\* 10 (ITA ≈ 58°; a melanin-only gain turned patches lavender). The
**vitiligo-aware** method detects depigmented pixels (ITA > 20th percentile + 50°, flagged above
3 % of the skin; clipped pixels ≥ 240 count for neither class), splits each crop into pigmented
and depigmented traces (masks eroded 5 px) and fuses them with Kalman + SNR fusion.

| Coverage | 0 % | 10 % | 25 % | 50 % |
|---|---|---|---|---|
| `pos_kf_snr` error | 0.10 | 3.18 | 2.96 | 3.18 |
| vitiligo-aware error | 0.10 | 2.52 | 2.74 | 2.30 |
| detection | 0 / 3 crops | 3 / 3 | 3 / 3 | 3 / 3 |

The aware method helps but is still +2.2–2.6 bpm over clean (fails τ = 2 narrowly); the
depigmented traces get almost no weight because the bright patches clip. One subject only.

**On SRMC full faces the detector does not work yet:** run on face crops it flagged all 5 tested
clips (S_AB22 with spots, and S_AB12, S_AB15, S_AB16, S_AB21 without) with 22–68 % of the skin as
depigmented: hair, eyebrows and shadows set the 20th-percentile reference. Needed: run it only in
the ROI skin, take the reference from the median skin, require a minimum spot size and stability
over time, then validate on S_AB22 vs clips without vitiligo.

### Comparison with the `pipeline` branch (CP3 freeze, 27 Nov)

`config/round1_freeze.json` turns off the background patches (U2), Viterbi (E014/E020) and motion
suppression (E018, "not testable on AiPEX"), and takes the video-level HR as the peak of the
averaged spectrum. On SRMC: background and gated motion both help (above), Viterbi does not
(agrees), the averaged-spectrum video level is worse than the median of windows (17.79 vs 13.97),
and the `pos_kf_snr` core is worse than plain `pos`.

### Constraints touched

| Constraint | Evidence here |
|---|---|
| C1 low light | background reference targets real flicker / exposure changes; it cannot help the official simulated C1 test (independent per-pixel noise). SRMC Resting has no Dim clips (Task clips do) |
| C2 skin tone | final method: Dark 14.5 / Medium Fair 13.2 / Fair 14.6 (p = 0.63); part of the `pos` gap was a motion difference; vitiligo prototype on AiPEX |
| C4 low resolution / compression | 480p re-encoded clips are the main failure (26 clips without a visible pulse); report as a limit and output "unreliable" on low SNR |
| C5 on-device | background costs ~13 ms per 1080p frame |
| C6 aging | no age effect on SRMC; ROI cleaning no effect |
| C7 occlusion | facial tubes are not detected by the skin-ratio rule |

### Outputs (local, `outputs/` is gitignored)

`outputs/tables/`: `c2_subjects_shard1of1_SRMC_protocol-Resting.csv`,
`c2_subjects_shard1of1_SRMC_group-Abnormal_protocol-Resting.csv`,
`c2_subjects_shard1of1_SRMC_clean_13subjects.csv`, `motion_*.csv`, `viterbi_*.csv`,
`srmc_normal_resting_methods.csv`, `srmc_abnormal_resting_methods.csv`,
`srmc_failure_diagnostics.csv`, `srmc_aggregation_test.csv`, `harmonic_*.csv`, `snrmedian_*.csv`,
`artefact_*.csv`, `vitiligo_India_video9*.{csv,json}`. A partial `c2_subjects_shard1of1_SRMC.csv`
from a stopped run (includes Task clips) should be ignored.

### Next steps

1. Add `pos` + background + gated motion as a `--method` of `run_c2.py` (C2 tables).
2. Spectral whitening against the 55–65 bpm noise hump (choose on Normal, test on Abnormal).
3. "Unreliable" flag for low-SNR recordings (C4: degrade gracefully).
4. Evaluate the SRMC Task clips (Bright / Med / Dim: C1 on real video, unused data).
5. Vitiligo detector for full faces; more AiPEX vitiligo subjects.
6. Tube / thin-occluder detection (C7).

## Known limitations

- ITA depends on camera exposure and white balance. Inside one dataset it ranks subjects reliably;
  across datasets the absolute class can be off (UBFC b\* ≈ 6 pushes everyone to "Very light",
  MPU b\* ≈ 22 pushes East-Asian subjects to "Brown"). `wb_warning` flags b\* < 3.
- Few full-face subjects with dark skin in UBFC / CLBP / MPU. SRMC-N adds 68 Dark-rated clips, but
  with one PR reading per clip (possibly taken at another moment than the video).
- SRMC skin tone is a rater label (Fair / Medium Fair / Dark), not Fitzpatrick; ITA from SRMC video
  depends on the camera (480p vs HD).
