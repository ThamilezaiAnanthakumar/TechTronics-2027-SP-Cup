# SNR-weighted region fusion (teammate's method) — import, reproduction, validity checks

README_UPDATE_03 U11 and U12.1–12.3. Code: `contrib/snr_fusion/` (unmodified, see its `SOURCE.md`).
Credit: **Mohamed Irshad Nadha**, branch `c2-skin-tone`, commits `3ff7a15` and `425fdc1` (27 Nov 2026).
Analysis helper (reads their per-video table, recomputes no HR): `tools/snr_fusion_analysis.py`.
Per-video CSVs: `outputs/snr_fusion/`.

## 1. The method, exactly as implemented (AiPEX path)

Source: `facevitals/face.py::crop_traces`, `facevitals/rppg.py::{sanity_hr, sanity_hr_snr, region_snr}`,
`run_c2.py::evaluate_subject` (the `""`, `eq_` and `snr_` methods).

**Input traces (shared by baseline and fusion).** Each AiPEX recording = three crop videos
(`forehead`, `leftcheek`, `rightcheek`), read with OpenCV, timestamps `CAP_PROP_POS_MSEC`
(falls back to k/fps if not increasing). Skin mask on the whole crop: YCbCr (MATLAB `rgb2ycbcr`
coefficients) Cb ∈ [77, 135], Cr ∈ [128, 173], Y ≥ 20, max(R,G,B) < 250
(`config/face_config.json` of their branch). Per frame and crop: plain mean RGB of the skin
pixels (no trimming, no specular removal). The three crops are truncated to the shortest one.
`rgb_face` = pixel-count-weighted mean RGB over the three crops. No Kalman filter in `pos` / `pos_snr`
(the `pos_kf*` variants exist but are not the claimed result).

**Pulse signal (per trace).** Linear interpolation onto a uniform 30 Hz grid (AiPEX is 15 fps, so
×2 upsampling); POS (Wang 2017), 1.6 s window, overlap-add; Butterworth `butter(3, [0.7, 3.0] Hz)`
band-pass (6th-order overall), zero-phase `filtfilt`. No detrending, no normalisation.

**Windows.** 10 s (300 samples at 30 Hz), step 1 s, centres from t₀+5 s to t_end−5 s; a window
needs ≥ 90 % of its samples. Periodogram with symmetric Hann window, nfft = 8192 at 30 Hz
(0.0037 Hz = 0.22 bpm bins), band 0.7–3.0 Hz, **no** sub-bin (parabolic) interpolation.

**Baseline `pos`.** The above on `rgb_face`; window HR = periodogram peak in band.

**SNR fusion `pos_snr`** (`sanity_hr_snr`, `power = 1.0`):
- Signals fused: the POS pulse signals of the **three crops separately** (R1 forehead, R2 left
  cheek, R3 right cheek), all POS. A crop valid in < 50 % of frames is dropped for the whole video;
  in a window, a crop is skipped if < 90 % of that window's grid points fall in its valid frames.
- Per window and crop: periodogram P(f); normalised to unit power inside 0.7–3.0 Hz.
- **SNR** (`region_snr`, "de Haan & Jeanne 2013"), computed inside the 0.7–3.0 Hz band only:
  f₀ = in-band peak; signal = bins with |f − f₀| ≤ **0.1 Hz** (±6 bpm) **or** |f − 2f₀| ≤ **0.2 Hz**
  (first harmonic **included**, ±12 bpm, only the part inside the band, i.e. only when 2f₀ − 0.2 ≤ 3 Hz);
  noise = all other in-band bins. **Linear ratio** (not dB), so always ≥ 0.
- **Weight** = SNR^p with p = `snr_power` = **1.0** (i.e. weight = linear SNR). No softmax, no
  top-k (all valid crops used), no subtraction of 1, no clipping: negative SNR cannot occur in
  linear units; a crop with SNR < 1 (dB < 0) still gets weight = SNR > 0.
- **Fusion on spectra**: F(f) = Σ_r SNR_r · P_r(f)/ΣP_r (not normalised by ΣSNR — irrelevant for the
  arg-max). Window HR = 60 · argmax F.
- Weights are recomputed in every window.

**Control `pos_eq`:** identical, weight = SNR⁰ = 1.

**Video-level estimator** (all three methods): **median** of the window HRs (`np.nanmedian`).
Error = estimate − reference; reference = `Heart_Rate(BPM)` of `Ground_Truth/<Country>_GT.csv`
(one Masimo mean per recording). Per-video "MAE" = |error|; "within 5 bpm" = |error| < 5.

**Every parameter:** window 10 s, step 1 s, resample 30 Hz, POS 1.6 s, band 0.7–3.0 Hz, Butterworth
order 3 (filtfilt), Hann periodogram, nfft 8192, window coverage 0.9, crop coverage 0.5 (video) /
0.9 (window), SNR signal half-widths 0.1 Hz (f₀) and 0.2 Hz (2f₀), weight exponent 1.0, video
estimator median. Skin box as above. `c2_config.json`: `window_s` 10, `step_s` 1, `snr_power` 1.0
(`tau_bpm` 2.0, `bootstrap` 2000, `seed` 0 only affect the ITA statistics).

## 2. Was any parameter chosen after looking at AiPEX results? (U12.1)

**Case A — no AiPEX-specific tuning found.** Evidence:
- Every fusion parameter has exactly one value in the code history, and the fusion code is
  identical in both commits (`git diff 3ff7a15 425fdc1 -- src/python/facevitals/rppg.py` only
  adds `kalman_denoise`). `snr_power = 1.0` is in `c2_config.json` from the first commit; there is no
  sweep code, no alternative value in comments, and no logged alternative result.
- The SNR band widths (±0.1 Hz at f₀, ±0.2 Hz at 2f₀) are attributed to de Haan & Jeanne (2013) in
  the code comments. Window 10 s / step 1 s, POS 1.6 s, 0.7–3 Hz, Butterworth order 3, nfft 8192 and
  the median video estimate are the baseline port of our MATLAB `sanity_hr.m` / `spectral_hr.m`,
  i.e. fixed before the fusion existed.
- The AiPEX results table is already in the first commit (`3ff7a15`), together with the code that
  produces it; `pos_eq` (equal weights) is reported as a control, not as a tuned alternative.
- Limit of this evidence: exploratory runs that were never committed cannot be seen. The only free
  choice beyond literature/baseline is p = 1 (weight = linear SNR), the natural default.
- Not part of the claim: commit 2 adds `pos_kf_snr` (Kalman + SNR fusion), which our reproduction
  gives **5.76** MAE on all 88 (mean-of-windows estimator 7.45). It was added after `pos_snr` had been
  seen on AiPEX, is not in their README table, and must go through the keep/reject rule before use.

Consequences (U12.1, Case A): the result stands as "full AiPEX set, no AiPEX-specific tuning";
AiPEX-test is **no longer held out for this method** (the 26 test videos are among the 88 it was
reported on). The split stays for future methods. Split-wise numbers are in Section 5.

## 3. Reproduction (U11.2)

Run on 27 Nov 2026 with their code unmodified (copy of `contrib/snr_fusion` plus their
`config/face_config.json` in `D:\SPCup Data\work\snr_repro\code`), Python 3.11.3, numpy 1.26.4,
scipy 1.17.1, pandas 2.2.0, opencv 4.11.0, scikit-image 0.26.0, mediapipe 1.2.3:

```
python src/python/run_c2.py --datasets AiPEX --data-root "D:/SPCup Data/dataset/AiPEX rppg_biases" --shard i/5   # i = 1..5
python src/python/run_c2.py --analyze-only --method compare     # and --method pos_snr, --method pos
python tools/snr_fusion_analysis.py "D:/SPCup Data/work/snr_repro/code/outputs/tables/c2_subjects.csv"
```

88/88 recordings processed, no errors; every method gives an estimate for every recording.
**The claim reproduces exactly** (all values to the printed precision):

| Quantity | Claimed (README_UPDATE_03 §0 / their README) | Reproduced |
|---|---|---|
| `pos` mean / median AE / within 5 / bias | 16.0 / 16.5 / 14 % (16.01 / 16.50 / 13.6 % / +15.8) | 16.01 / 16.50 / 13.64 % / +15.83 |
| `pos_eq` mean / within 5 | 14.06 / 19.3 % | 14.06 / 19.32 % |
| `pos_snr` mean / median AE / within 5 / bias | 8.2 / 5.8 / 43 % (8.20 / 5.77 / 43.2 % / +5.7) | 8.20 / 5.77 / 43.18 % / +5.72 |
| `pos_snr` vs `pos`, better / worse by > 1 bpm | 72 / 10, p < 0.001 | 72 / 10; Wilcoxon p = 5.9e-13 |
| `pos_snr` vs `pos_eq`, better / worse > 1 | 69 / 15 | 69 / 15 |
| India `pos` / `pos_eq` / `pos_snr` (n 44) | 15.76 / 12.89 / 7.87 | 15.76 / 12.89 / 7.87 |
| Sierra Leone (n 44) | 16.25 / 15.22 / 8.54 | 16.25 / 15.22 / 8.54 |
| ITA population (n 85) `pos` / `pos_snr` | 16.06 / 8.12 | 16.06 / 8.12 |
| I–II n = 15 `pos` / `pos_snr` (dev.) | 17.71 / 9.33 (+1.21) | 17.71 / 9.33 (+1.21) |
| III–IV n = 14 | 21.72 / 9.09 (+0.97) | 21.72 / 9.09 (+0.97) |
| V–VI n = 56 | 14.20 / 7.56 (−0.56) | 14.20 / 7.56 (−0.57; their −0.56 is rounding of 7.56 − 8.12) |

**Labelling error in README_UPDATE_03 §0:** the overall table (16.0 → 8.2, 72/10) is over
**all 88** recordings, not 85. Only the ITA-group table uses 85 (recordings with an ITA value);
on those 85: `pos` 16.06, `pos_snr` 8.12, median 5.89, within 5 42.4 %, better/worse > 1 bpm 70/9.

## 4. Comparison with our implementation (U11.4)

Ours = `tools/aipex_iter.m` (`videoEstimate`, `traceSpectrum`, `snrDb`) with the frozen CP2 options
of `config/round1_freeze.json` / experiment E019: `fuse='spectral'`, `topK=5`, `filterOrder=2`,
`pixels='trim'`, `videoEst='avgfused'`, Kalman on, POS, `windowSec=10`.

**They are not equivalent.** Same idea (per-crop POS spectra, per-window SNR-weighted spectral
fusion of the three AiPEX crops, unit-power normalisation, f₀ ± 0.1 Hz signal band with harmonic,
rest of 0.7–3 Hz as noise), but every one of the following differs:

| Item | Teammate (`pos_snr`) | Ours (CP2 / E019) |
|---|---|---|
| Pixels per crop | plain mean of skin pixels | skin + specular removal (Y > 0.95·255 or channel ≥ 250) + 10/10 % trimmed mean, ≥ 20 px |
| Kalman pre-filter | none | `kalman_denoise`, R_k = R0·median(npix)/npix_k, 4 Hz |
| Sampling | resampled to 30 Hz (linear interp. on timestamps) | native rate (15 fps), no resampling |
| Detrending | none | smoothness priors λ = 100 |
| Band-pass | Butterworth order 3 (`butter(3,…)` → 6th-order band-pass), filtfilt | order 2 (4th-order band-pass), zero-phase SOS filtfilt |
| Normalisation of pulse | none | z-score |
| Spectrum | `scipy.signal.periodogram`, Hann, nfft 8192 @ 30 Hz | FFT, Hann, nfft 16384 @ 15 Hz |
| Harmonic signal band | ±0.2 Hz around 2f₀ | ±0.1 Hz (6/60 Hz) around 2f₀ |
| SNR units | linear | dB, converted back to linear for weights |
| Weight | SNR_lin (p = 1) | max(SNR_lin − 1, 0); all zero → equal weights |
| Selection | all valid crops | top-k = 5 (all 3 crops) |
| Fused spectrum scale | Σ w·P (unnormalised) | Σ w·P / Σ w (same arg-max) |
| Crop validity gates | < 50 % valid frames → dropped; window needs ≥ 90 % coverage | crop needs ≥ windowSec·fs finite frames; gaps filled linearly (`fillmissing`) |
| Window HR | arg-max bin, no interpolation | — (not used by `avgfused`) |
| **Video estimator** | **median of the per-window fused-spectrum peaks** | **peak (parabolic interpolation) of the mean of the per-window fused spectra** |
| Implementation | Python 3.11 / SciPy | MATLAB |

The two differences most likely to matter for the video-level number are the video estimator
(median of window HRs vs averaged fused spectrum) and the weight rule (SNR vs SNR − 1 clipped at 0).
A numerical side-by-side requires running the teammate's rule inside `aipex_iter.m`
(`fuse.method = "snr_teammate"`, U11.4); that was **not** done here (our code and frozen config are
untouched in this task).

## 5. Validity checks (U12)

### 12.1 Videos used
All 88 AiPEX recordings (India 44, Sierra Leone 44) = all 62 AiPEX-dev + all 26 AiPEX-test ids of
`shared/splits/`. Case A (Section 2).

### 12.2 The "missing" three
Nothing was excluded from the HR table. The 3 recordings missing from the ITA table have no ITA
value because **no crop passed the ITA colour gates** (≥ 3 sampled frames with skin ≥ 30 % of the
crop and median L\* ≥ 25; good samples forehead/left/right: 0/0/0, 0/0/1, 0/0/0) — a skin-tone
measurement limit, not an HR failure and not a data problem. HR is estimated for all three:

| Recording | Split | Ref | `pos` AE | `pos_snr` AE | windows |
|---|---|---|---|---|---|
| India_video1 | dev | 85.4 | 22.58 | 24.89 | 22 |
| India_video42 | dev | 76.5 | 5.90 | 3.33 | 281 |
| Sierra_Leone_video36 | test | 78.9 | 14.86 | 3.32 | 360 |

So the method has **100 % coverage (88/88)**; no fallback estimate is needed, and the corrected-set
table is the 88-recording table of Section 3 (`pos_snr` 8.20). For the C2 statistics the three stay
"ITA unknown" (they cannot be put in a group).

### 12.3 Like-for-like baseline
`pos` and `pos_snr` in their code share loader, timestamps, skin mask, 30 Hz resampling, POS,
band-pass, windows, reference and the median video estimator; they differ only in fusion (pixel
average of the 3 crops → one POS vs per-crop POS spectra fused by SNR). Paired over videos
(Δ = `pos_snr` − `pos`; bootstrap 5,000 resamples over videos, seed 2027; sign test = two-sided
binomial on all non-tied pairs, no ties):

| Set | n | `pos` | `pos_snr` | Δ MAE [95 % CI] | better/worse | sign p | > 1 bpm b/w (sign p) | Wilcoxon p |
|---|---|---|---|---|---|---|---|---|
| all | 88 | 16.01 | 8.20 | −7.80 [−9.31, −6.32] | 74/14 | 5.1e-11 | 72/10 (1.0e-12) | 5.9e-13 |
| ITA-valid | 85 | 16.06 | 8.12 | −7.94 [−9.44, −6.41] | 72/13 | 4.6e-11 | 70/9 (7.8e-13) | 1.1e-12 |
| AiPEX-dev | 62 | 16.15 | 7.91 | −8.24 [−10.07, −6.46] | 52/10 | 5.7e-8 | 51/7 (2.4e-9) | 6.7e-10 |
| AiPEX-test | 26 | 15.66 | 8.90 | −6.76 [−9.22, −4.10] | 22/4 | 5.3e-4 | 21/3 (2.8e-4) | 5.2e-5 |

SNR weights vs equal weights (`pos_snr` − `pos_eq`, all 88): −5.85 [−7.24, −4.53], 72/16, sign
p = 1.2e-9; spectral fusion alone (`pos_eq` − `pos`): −1.95 [−2.91, −0.94]. Most of the gain is from
the SNR weights. Full table incl. per-country: `outputs/snr_fusion/paired_stats.csv`.

**Why their baseline is 16.0 and earlier POS runs were 18.1 / 18.6:** mostly the video estimator.
Their code also stores the mean of the window HRs (as in our `aipex_hr.m`); with it `pos` = **18.56**
and `pos_snr` = **11.80** on the 88. Median vs mean of window HRs is worth 2.5–3.6 bpm on AiPEX, so
numbers from different pipelines are only comparable at equal video estimator.

### Split-wise numbers (for comparison with our dev-CV)

| Set | `pos` MAE / median / ≤ 5 | `pos_eq` MAE | `pos_snr` MAE / median / ≤ 5 / bias |
|---|---|---|---|
| AiPEX-dev (62) | 16.15 / 16.57 / 12.9 % | 14.09 | **7.91** / 5.90 / 43.5 % / +5.30 |
| AiPEX-test (26) | 15.66 / 15.76 / 15.4 % | 13.98 | **8.90** / 5.53 / 42.3 % / +6.72 |
| all (88) | 16.01 / 16.50 / 13.6 % | 14.06 | **8.20** / 5.77 / 43.2 % / +5.72 |

Our frozen CP2 (E019): AiPEX-dev 5-fold CV MAE **8.76** [7.01, 10.83]. Their `pos_snr` on the same 62
dev ids: **7.91**. Not strictly like-for-like: theirs has no fitted parameters (Case A, so no CV
needed), ours is the CV estimate of a configuration selected on dev; same reference values.
Nothing in our pipeline or frozen config was changed.

Files (force-added, `outputs/` is git-ignored): `outputs/snr_fusion/per_video.csv` (88 rows: ids,
split, ITA, reference, estimates/errors of `pos` / `pos_eq` / `pos_snr` with median and mean video
estimators, SNR weight shares), `summary.csv`, `paired_stats.csv`, `ita_groups.csv`. Raw run:
`D:\SPCup Data\work\snr_repro\` (shard logs, `analyze_*.txt`, `code/outputs/tables/c2_subjects.csv`,
cached traces).
