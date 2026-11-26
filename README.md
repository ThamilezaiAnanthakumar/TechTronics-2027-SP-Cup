# TechTronics FaceVitals — IEEE SP Cup 2027 (rPPG)

**Build brief for Claude Code.** This file is the single source of truth for what we are building, why, in which order, and how we know each piece works. Read it fully before touching code. When this file and the code disagree, raise it — do not silently "fix" either.

---

## 0. How Claude Code must work on this repo

1. **Work phase by phase** (Section 16). Start with Section 15 (integrate the team's existing code). Do not start a later phase until the current phase's *Definition of Done* passes.
2. **Never fabricate or hand-edit results.** Every number in `outputs/` and in the report must come from a script that can be re-run. If a run fails, report the failure.
3. **Never tune on test data.** Hyper-parameters may only be chosen on the *dev* split (Section 8.2). The test splits are frozen. Any change after the freeze date must be logged in `docs/CHANGELOG_PARAMS.md` with the reason.
4. **Golden vectors are sacred.** `tests/golden/` holds reference inputs/outputs produced by the MATLAB reference implementation. Ports (Python, TypeScript, C) must match them within tolerance. Do not regenerate golden files unless the human explicitly asks.
5. **MATLAB is the reference implementation** for the competition (sponsor = MathWorks; `process_round1.m` is a required deliverable). The shared C++ core `fvcore` (used by web, Android and Pi) must reproduce it, proven by trace-level golden vectors (Section 11).
6. **Keep the existing contributors' work.** Sulojan (face module), Thamil (Kalman denoising) and Muftee (signal chain) code is reused and extended. Do not rewrite a working module without a measured reason.
7. Before each commit: run `runtests('tests')` in MATLAB (and `pytest`, `npm test` where relevant). Commit messages: `[module] what changed — measured effect`.
8. When unsure about a MATLAB API's behaviour (toolbox version, codegen support, landmark output of a detector), **write a 10-line probe script and check**, then record the answer in `docs/API_NOTES.md`.

---

## 1. Competition requirements (what will be judged)

### 1.1 Dates
| Milestone | Date |
|---|---|
| **Round 1 submission (HR)** | **30 Nov 2026** |
| Round 1 results | 05 Dec 2026 |
| Round 2 submission (HR + SpO2) | 16 Jan 2027 |
| Finalists announced | 09 Feb 2027 |
| Final at ICASSP | 16–21 May 2027 |

### 1.2 Round 1 / Task 1 required content
- **Technical report** in IEEE conference format covering:
  - the pipeline (face detection and ROI selection, signal extraction, filtering, HR estimation);
  - metrics: MAE (bpm), RMSE (bpm), Pearson r, Bland–Altman limits of agreement;
  - **at least two constraints C1–C7 addressed and justified**. Addressing more is rewarded, so we address all seven.
- Judged on innovation, technical feasibility and clarity of documentation. "Least deviation from standard medical equipment" gets priority.
- HR from **controlled and uncontrolled** conditions.

### 1.3 Required files (Round 1)
```
Round1_Baseline/
  Round1_Baseline_None.mat        # facial video + reference HR at baseline
  process_round1.m                # pipeline entry point (MATLAB)
  target_video.mp4                # raw facial video
  reference_signal.csv            # synchronized ground-truth HR
  processed_output.csv            # our estimated HR
  (+ one RoundX_Condition_Constraint.mat per constraint we demonstrate, e.g.
     Round1_LowLight_C1.mat, Round1_LowResolution_C4.mat, Round1_LowBandwidth_C5.mat, ...)
```
Each `.mat` must contain exactly these variables:
| Variable | Content (our convention) |
|---|---|
| `facial_video` | RGB frames, `uint8 [H W 3 N]` (save with `-v7.3`) |
| `roi_data` | struct: per frame, patch polygons/masks, patch names, validity, weights |
| `reference_signal` | struct: `t` (s), `ppg` (if available), `hr_bpm`, `source` |
| `processed_signal` | struct: `t`, fused `bvp`, `hr_bpm` per second, `confidence`, `status` |
| `metrics` | struct: MAE, RMSE, r, BA bias / LoA, SUC@3, SUC@5, coverage |
| `params` | struct: resolution, fps, lux, distance_cm, fitzpatrick, ita_deg, age, constraints_applied, pipeline_version, git_hash |

Naming: `RoundX_Condition_Constraint.mat` (Condition ∈ Baseline, LowLight, LowResolution, LowBandwidth, MultiFace, Aging, Masked, SkinTone). Dataset files: `Subject{ID}_{SkinTone}_{Condition}_{Parameter}.ext`.

### 1.4 The seven constraints (quantified)
| ID | Spec | Official MATLAB simulation |
|---|---|---|
| C1 | Illuminance ≤ 50 lux | `im2double` × gain; `imnoise` Poisson and/or Gaussian |
| C2 | Fitzpatrick I–VI, vitiligo | `rgb2lab` → ITA, per-stratum accuracy vs population mean, pre-specified tolerance |
| C3 | Track designated target, reject bystanders | `faceDetector`, `vision.KalmanFilter`, `assignDetectionsToTracks`, `vision.PointTracker` |
| C4 | ≤ 640×480, stress ≤ 320×240, baseline 1280×720 | `imresize` bicubic |
| C5 | ≤ 500 kbps, ≤ 10 fps, offline/on-device | frame decimation + MPEG-4 re-encode |
| C6 | Ages 60–70+, report subgroup separately | `imfilter` edge kernels (synthetic only; real data primary) |
| C7 | ≥ 40 % lower face occluded | `insertShape` filled polygon on landmarks; must re-weight to forehead/periocular, never fail silently |

### 1.5 Open questions to send to the organiser NOW (cc `sp-competitions@listserv.ieee.org`, required)
1. Are "Task 1" and "Round 1" the same submission? (The document describes both.)
2. Task 1 asks for 1280×720 baseline. UBFC is 640×480. Is a VGA public baseline acceptable, or must the baseline `.mat` be HD?
3. `facial_video` inside the `.mat` is ~1–2 GB per minute of VGA. Size limit? One subject or all?
4. Are simulated constraint conditions (official recipes) accepted as constraint evidence in Round 1?
5. The directory list shows `Round4_FullSystem_Clinical` but there are three rounds — typo?
6. Is a Python `process_round1.py` acceptable alongside the MATLAB one?
7. **Sri Ramachandra data was shared without ground truth.** The file names follow the protocol of Arulselvi et al. (2023): rest, walk, stairs and squats, recorded with a patient monitoring system. Can you share the **patient monitoring system HR and SpO2 readings for each subject and each task** (with timestamps if available)? If they are withheld as a hidden test set, in what format should blind predictions be submitted? Do subject metadata (age, skin type, lux) exist?
8. Does the data agreement allow showing Sri Ramachandra faces at all — in the report, figures or demo video?
9. Some Sri Ramachandra subjects appear to be minors. Does the data agreement cover minors, and are there any extra conditions on how their data may be used?

---

## 2. Where we are (snapshot 26 Nov 2026)

| Item | Status |
|---|---|
| Face module (Sulojan, `face_detect`) | RetinaFace every 10 frames, Kalman track per face + KLT, 1-Euro ROI smoothing, 3 box-fraction ROIs, YCbCr skin box, occlusion hysteresis. 100 % target tracking on multi-face, QVGA, mask clips. Needs MATLAB ≥ R2025a. |
| Kalman denoise (Thamil, `thamilezai_dev`) | Per-channel random-walk, `R_k ∝ 1/npix`, −3 dB at 4 Hz, causal. AiPEX 18.6 → 13.1 bpm. |
| Signal chain (Muftee, `muftee_dev`, **local only**) | GREEN/CHROM/POS, smoothness-priors detrend λ=100, BPF 0.7–3 Hz, 10 s windows 1 s step, FFT 2^14 + parabolic, video-level HR, NaN-aware metrics, loaders, browser demo with MATLAB parity test. |
| Integration (`integrated_pipeline`) | `run_ubfc2.m`, cached face traces, 24-variant grid, synthetic test passes. |
| UBFC2 (42) | POS/combined: pooled MAE 2.17, RMSE 6.49, r 0.936, LoA −13.7/+11.4; median per-subject 1.19. Outliers: **15 (24.8), 49, 27, 41 (5.1–5.5)**. |
| AiPEX (88) | 13.1–18.6 bpm vs published BKF 5.8, CHROM 11.6, POS 14.0. India ≈ Sierra Leone → not a skin-tone problem. |
| UBFC1 | 6 videos, no summary. |
| Sri Ramachandra (organiser) | **On disk, no ground truth.** Not usable for accuracy; usable for the GT-free analyses in Section 6.1. |
| Not tested | C1 low light, C5 low fps/bitrate, C6 aging, C2 ITA stratification, vitiligo, Pi timing. |

### 2.1 Housekeeping (Phase 0)
- Push `muftee_dev`; merge `face_detect`, `thamilezai_dev`, `muftee_dev`, `integrated_pipeline` into `dev`; resolve the two `run_pipeline.m` files (keep `src/`, delete root, update callers); `main` = release branch.
- Install MATLAB R2026b + RetinaFace support package (complimentary competition licence). If unavailable in time, run with cached face traces and Viola-Jones fallback, and say so in the report.

### 2.2 Diagnosis tasks before any new algorithm work (highest value per hour)
**Run these checks on AiPEX-dev only (Section 8.5); they are the first rungs of the iteration loop.** **AiPEX gap (4–7 bpm worse than the *same* published methods) is almost certainly a pipeline problem, not physiology.** Check, in this order, with one script `tools/diagnose_aipex.m`:
1. **Frame timing.** Compare `VideoReader.FrameRate`, `NumFrames/Duration`, and per-frame timestamps (`readFrame` + `CurrentTime`). The source phone recorded ≈25 fps on average. Assuming 30 fps scales HR by 1.2 (≈15 bpm at 75 bpm). Plot estimated vs reference HR: if points lie on a line with slope ≠ 1, it is timing.
2. **Skin masking of pre-cropped ROIs.** AiPEX videos are already cropped (forehead 60×30, cheeks 25×25). Face detection and the YCbCr box are wrong here. Use all pixels (or a permissive mask) and log the skin-pixel ratio per video.
3. **ROI fusion.** Load R1, R2 and R3 as three patches of the same subject and fuse them with the Section 4 fusion.
4. **Reference definition.** The reference is a single mean HR per video. Compare `videoHR` (averaged spectra) against the median of confident 1 s estimates, and against the Viterbi path mean.
5. **Harmonics / sub-harmonics.** Histogram the estimate/reference ratios; peaks at 0.5 or 2 indicate the need for harmonic summation (Section 4.8).

**UBFC2 outliers (15, 49, 27, 41).** For each, plot the spectrogram, the per-ROI SNR and the face-box motion. Label the failure type (motion / harmonic lock / low SNR / tracking) in `docs/FAILURES.md`. Section 4.8 tracking and Section 4.6 quality weighting are designed for these.

---

## 3. System overview — "FV-TT" pipeline

**Name for the report:** *Quality-gated anatomical multi-patch rPPG with physiology-constrained spectral tracking.*

```
Video ─► M0 Timestamped reader ─► M1 Target face detect+track (+5 landmarks, motion signals)
      ─► M2 Anatomical patch grid (canonical face frame)
      ─► M3 Pixel selection per patch: adaptive skin model, pigmentation sub-clusters,
             wrinkle/texture mask, specular mask, occlusion
      ─► M4 Per-patch RGB traces + npix + validity ─► Kalman denoise (1/npix) ─► uniform resample
      ─► M5 Per-patch POS (CHROM alt.) ─► detrend + BPF
      ─► M6 Patch quality: SNR, spectral coherence, coverage ─► weights
      ─► M7 Top-k SNR-weighted spectral fusion
      ─► M8 Harmonic-summed spectrogram + motion suppression + Viterbi HR tracking
             (offline: forward–backward; online: fixed-lag)
      ─► M9 HR(t) at 1 Hz + confidence + status codes ─► M10 Metrics / M13 Packager
```

### 3.1 Design rationale (cite in the report)
- **Anatomical patches, not box fractions.** Li et al. (npj Cardiovasc Health 2026), using 15 MediaPipe-defined ROIs on BUAA-MIHR and MMPD, found that the glabella, nasal dorsum and both malar regions are the most robust under low and varied light. The whole forehead was only mediocre, and the submalar regions were worse than the malar ones. The ranking was consistent across CHROM, POS, OMIT and LGI (Kendall W ≥ 0.86).
- **Selective fusion beats single ROI and whole face.** Li 2026 found an inverted-U: acceptance peaked with the top 4–5 regions and declined toward the holistic face. Bondarenko et al. (npj Digit Med 2025) report diminishing returns with many patches. Hence a top-k SNR-weighted fusion rather than averaging everything.
- **Physiology-constrained tracking.** Tian et al. (npj Digit Med 2026) raised the ±10 bpm acceptance on LGI-PPGI from 46 % to 84 % with a rate-of-change constraint: about 3 bpm/s decrease (from HR-recovery data) and about 6 bpm/s increase. We embed the same prior as Viterbi transition costs over the whole spectrogram, which also recovers from bad starting points. Tian noted bad starting points as the weakness of their causal rule.
- **ROI cleaning for aging skin.** Caroppo et al. (MTAP 2024) removed wrinkles and moles from the cheek ROI in elderly subjects. This cut MAE from 3.75 to 2.20 bpm at 0.5 m.
- **Classical core.** Model/time complexity is one of the six final criteria. A Raspberry Pi target favours POS/CHROM-class methods (MathWorks training material says the same). Learned rPPG models also generalise poorly across users and datasets (Dasari 2021: DeepPhys −14 bpm bias out of distribution; Pham 2026: identity leakage in PPG models).
- **Correct statistics.** Altman & Bland (1983): correlation is not agreement, so lead with bias and limits of agreement. Several windows per subject means repeated-measures Bland–Altman.

---

## 4. Algorithm specification (MATLAB reference)

All parameters live in `config/default.json` and are read by `fv.config.load()`. Values below are **initial**. Tune only on dev.

### 4.0 M0 — Timestamped I/O
- `fv.io.readVideo(path)` returns frames (or a frame iterator) **and per-frame timestamps** `t` (s) from `CurrentTime`. Never assume a constant fps.
- Loaders (each returns a common `Record` struct: `id, videoPath, t_ref, ppg_ref, hr_ref, meta`):
  - `loadUBFC2` (ground_truth.txt: line 1 PPG, line 2 HR, line 3 time in seconds);
  - `loadUBFC1` (gtdump.xmp: ms, HR, SpO2, PPG);
  - `loadAiPEX` (pre-cropped R1/R2/R3 videos, one mean HR per video, country);
  - `loadSriRamachandra` (inspect the format first; write `docs/DATASET_SRIRAM.md`);
  - `loadGeneric` (mp4 + csv).
- `meta` must carry: resolution, fps_nominal, fps_measured, lux (if known), skin type (if known), age (if known), country.

### 4.1 M1 — Target detection and tracking (C3)
- Reuse Sulojan's module. Detector every N=10 frames (adaptive: N=5 if track quality drops), KLT every frame, Kalman per track, Hungarian association (`assignDetectionsToTracks`) with cost = 1 − IoU + λ·(normalised centre distance).
- **Designated-target rule** (configurable): (a) the user clicks a face in UI or live mode; (b) the largest face in the first 2 s in offline mode; (c) an explicit `target_bbox` in config. Re-acquire only near the last target position. Add an **appearance check**: a colour histogram of the target's central face area. A candidate whose Bhattacharyya distance exceeds the threshold is rejected, so a bystander crossing the target position is not adopted.
- Output per frame: target bbox, **5 landmarks** (eyes, nose tip, mouth corners) if the detector provides them (probe RetinaFace/`faceDetector` output; record in `API_NOTES.md`), track quality ∈ [0,1], and **motion signals** (bbox centre x, y and scale, KLT similarity parameters) for M8.
- Optional dense landmarks: MediaPipe Face Mesh via MATLAB–Python interop (`pyrun`). If available, use the Li 2026 polygons (Section 4.2b) directly.

### 4.2 M2 — Anatomical patch grid
**(a) Canonical-frame patches (default, codegen-friendly, needs only 5 landmarks).**
- Origin = midpoint of the eye centres. x-axis along the eye line (toward the subject's left eye = +x in image coordinates after rotation). Unit = inter-ocular distance (IOD). y points down.
- A similarity transform from the landmarks maps canonical rectangles to image polygons every frame, then 1-Euro smoothing is applied to polygon vertices (reuse Sulojan's filter, min cutoff 1 Hz, β 0.02).
- **Initial patch table** (centre x, centre y, width, height in IOD). These are initial values and must be verified visually with `tools/overlay_patches.m` on 10 subjects:

| Patch | cx | cy | w | h | Priority group |
|---|---|---|---|---|---|
| glabella | 0.00 | −0.15 | 0.30 | 0.22 | A |
| nasal_dorsum | 0.00 | 0.22 | 0.18 | 0.28 | A |
| malar_L | +0.55 | 0.45 | 0.32 | 0.28 | A |
| malar_R | −0.55 | 0.45 | 0.32 | 0.28 | A |
| forehead_medial | 0.00 | −0.55 | 0.50 | 0.28 | B |
| forehead_lat_L | +0.45 | −0.55 | 0.32 | 0.28 | B |
| forehead_lat_R | −0.45 | −0.55 | 0.32 | 0.28 | B |
| philtrum | 0.00 | 0.85 | 0.30 | 0.12 | C (masked-out first under C7) |
| submalar_L | +0.50 | 0.85 | 0.28 | 0.25 | C |
| submalar_R | −0.50 | 0.85 | 0.28 | 0.25 | C |

- Each patch is subdivided 2×2 into **sub-patches** (40 total) for quality weighting; sub-patches are the fusion units.
- Clamp patches to the face box. A patch is invalid if more than 30 % of it falls outside the frame.

**(b) MediaPipe polygons (web, mobile, Pi, and MATLAB-with-Python).** These are the 468-landmark indices from Li et al. 2026, Table 2. Use exactly as published:
```
MF  medial forehead      10,109,108,151,337,338
G   glabella             151,108,107,55,8,285,336,337
LLF left lat. forehead   67,103,104,105,66,107,108,109
RLF right lat. forehead  297,338,337,336,296,334,333,332
ND  nasal dorsum         8,55,193,122,196,3,195,248,419,351,417,285
LM  left malar           126,100,118,117,116,123,147,187,205,203,129,209
RM  right malar          355,429,358,423,425,411,376,352,345,346,347,329
P   philtrum             2,97,98,203,186,185,40,39,37,0,267,269,270,409,410,423,327,326
LSM left submalar        203,205,187,147,177,215,138,172,136,135,212,186
RSM right submalar       423,410,432,364,365,397,367,435,401,376,411,425
```
The patch layer exposes one interface (`getPatches(frame, faceState) -> {name, mask, group}`), so the downstream code is identical for (a) and (b).

### 4.3 M3 — Pixel selection inside each patch (C2, C6, C7)
For each sub-patch and frame, build a boolean mask = skin ∧ ¬texture ∧ ¬specular ∧ ¬occluded.
1. **Adaptive skin model (C2).** In the first 3 s, collect pixels from the glabella and malar patches. Fit a 2-D Gaussian in (Cb, Cr) (or normalised rg). Pixels with Mahalanobis distance < τ (initial 3.0) are skin. Refit slowly (EMA, 30 s time constant). Keep the old YCbCr box only as a sanity bound (widened, as Sulojan did for AiPEX). This avoids a fixed box that is biased toward one skin tone.
2. **Pigmentation sub-clusters (C2 vitiligo).** In each sub-patch, if the skin pixels' L* distribution is bimodal (Ashman's D > 2 or a 2-component GMM with separation > ΔL threshold), split the pixels into two clusters by L*. Each cluster becomes its own trace. POS normalises each trace by its own temporal mean, so depigmented and pigmented skin are not mixed into one normalisation. This is our explicit vitiligo handling.
3. **Texture / wrinkle mask (C6).** Gradient magnitude after Gaussian smoothing (σ = 1 px at VGA, scaled with IOD). Exclude the top q % (initial 15 %) of pixels by gradient, plus pixels with high temporal variance relative to the patch median (micro-shadows that move with expression). Then take a **trimmed mean** (10 %/10 %) of the remaining pixels instead of a plain mean.
4. **Specular mask.** Exclude pixels with Y > 0.95·max or saturated in any channel.
5. **Occlusion (C7).** Sub-patch skin ratio EMA with hysteresis (reuse Sulojan's thresholds on 0.40, off 0.50, plus the forehead-ratio test). A patch is occluded when the ratio is below the threshold. A mask classifier is optional; the skin-ratio test is the primary signal. When ≥ 40 % of group-C plus lower malar coverage is occluded, set `status=MASKED` and the fusion uses groups A/B only.
6. Output per sub-patch per frame: mean RGB (trimmed), `npix`, `valid` (track quality > 0, not occluded, npix ≥ npix_min, where npix_min scales with resolution — initial 20 px at QVGA).

### 4.4 M4 — Traces, denoising, resampling (C1, C4, C5)
- Blank invalid samples (NaN). Thamil's Kalman per channel with `R_k = R0·median(npix)/npix_k`. **Q must be computed from the actual sample rate** so the −3 dB point stays at min(4 Hz, 0.8·fs/2). At 10 fps this is 4 Hz versus a Nyquist of 5 Hz; check the derivation still holds.
- Resample to a uniform grid of `fs_proc = min(30, fs_measured)` Hz using the real timestamps (linear interpolation; gaps longer than 0.5 s stay NaN and are reflected in the validity mask).
- **Adaptive spatial pooling (C1, C4).** If a sub-patch's median npix falls below a threshold, or its SNR (M6) is below a threshold for > 5 s, merge it with its 2×2 siblings into the parent patch. Low light and low resolution then automatically trade spatial resolution for SNR (shot-noise variance falls as 1/npix).

### 4.5 M5 — Per-patch rPPG
- **POS** (Wang 2017) with window 1.6 s (in samples: `round(1.6*fs_proc)`), overlap-add. CHROM as an alternative (config switch). GREEN is dropped (11–15 bpm on UBFC2). OMIT/LGI are optional for the ablation table only.
- Then smoothness-priors detrend (λ = 100, scaled to fs), zero-phase Butterworth band-pass, order 4, **0.7–3.0 Hz** (42–180 bpm). Keep an option to widen to 0.65–4.0 Hz for exercise data; the competition band is 0.7–3 Hz. Then z-score.

### 4.6 M6 — Patch quality
For each sub-patch and analysis window (10 s, 1 s step):
- **SNR** (de Haan & Jeanne 2013 style): power within ±6 bpm of the window's dominant in-band peak f0 and of 2·f0 (if below Nyquist), divided by the remaining in-band power, in dB.
- **Coherence**: agreement of the patch's peak with the robust consensus of all valid patches (median of peaks); 1 if within 3 bpm, decaying otherwise.
- **Coverage**: fraction of valid samples in the window (≥ 0.8 required, as now).
- Weight `w = coverage · max(SNR_lin − 1, 0)^p · coherence` (p initial 1). Zero if coverage < 0.8.

### 4.7 M7 — Top-k spectral fusion
- Per window, compute each patch's power spectrum `P_i(f)` (Hann window, zero-pad to 2^14 points over the processing band), normalise to unit in-band energy, keep the **top-k by weight** (initial k = 5; ablate k ∈ {1, 3, 5, 8, all}), and fuse: `P_fused = Σ w_i P_i / Σ w_i`.
- Spectral rather than waveform fusion avoids phase cancellation between patches. A waveform version, `bvp_fused`, is also produced (SNR-weighted sum after sign/lag alignment by cross-correlation with the best patch) for the `processed_signal` output and for later SpO2/BP work.

### 4.8 M8 — Spectral HR tracking (the main fix for outlier subjects)
1. **Harmonic summation.** `H(f) = P_fused(f) + α·P_fused(2f)` with α = 0.5, using the 2f term only where 2f < 0.9·fs/2. This penalises the sub-harmonic locks and resolves half/double ambiguities.
2. **Motion suppression (optional, ablated).** Compute the in-band spectrum of the M1 motion signals. Where it holds > β of the normalised motion power, attenuate `H` by `(1 − γ·M(f))` (γ initial 0.7). Never zero the spectrum completely.
3. **Viterbi.** States = HR grid 42–180 bpm at 0.5 bpm. Emission = `log(H_t(f) + ε)`, normalised per window. Transition from h to h′ over Δt = 1 s: allowed only if `−3 ≤ (h′ − h)/Δt ≤ +6` bpm/s (Tian 2026 physiology); within the allowed range the cost is `λ·|h′ − h|` (λ initial 0.1 per bpm).
   - **Offline mode:** full Viterbi (or forward–backward for per-window posteriors, which become the confidence).
   - **Online mode:** fixed-lag Viterbi with lag L = 3 s, so outputs are delayed by L. During the first 10 s, output "warming up".
4. **Refinement.** Parabolic interpolation of `H_t` around the path's bin.
5. **Confidence** ∈ [0,1] = posterior mass within ±3 bpm of the path (forward–backward) or the normalised peak prominence (online). **Status codes:** `OK`, `WARMUP`, `NO_FACE`, `MASKED` (still estimating from A/B), `LOW_SNR`, `LOW_COVERAGE`. HR is NaN when not OK/MASKED. C7 requires that the system never fails silently.

### 4.9 M9 — Outputs
- Per second: `t, hr_bpm, confidence, status, n_patches_used, top_patches`.
- Video-level HR (for AiPEX and one-value references) = confidence-weighted mean of the path over OK windows. Also report the median.
- `processed_output.csv` columns: `time_s,hr_bpm,confidence,status`.

### 4.10 Reference HR (must not share the tracker)
The reference HR comes from the contact PPG through **plain** detrend → BPF → per-window FFT peak (or peak-to-peak IBI median), with **no Viterbi and no fusion**, on the same window grid. Where the dataset gives oximeter HR (UBFC HR line, AiPEX mean HR), report against both. Never pass the reference through our tracker. That would make our errors look smaller than they are.

---

## 5. Constraint plan and acceptance tests

Implement degradations in `src/+fv/+degrade/`. Each takes a video path plus parameters and writes a new mp4 plus a JSON sidecar with the exact parameters. **Use the official MATLAB functions named in the spec** so judges can reproduce them.

| ID | Generator | Parameters swept | Mechanism in pipeline | Acceptance test (report) |
|---|---|---|---|---|
| C1 | `lowLight`: `im2double` → ×g → noise → `uint8` | g = target_lux / baseline_lux (assume baseline 300 lux unless measured) for 50, 25, 10 lux; noise: Poisson, Gaussian (var 1e-4…1e-3), combined | Adaptive pooling, 1/npix Kalman, trimmed means, top-k fusion, Viterbi | MAE vs lux curve; ≤ 50 lux point explicitly reported; degradation relative to baseline |
| C2 | `skinTone` stratification (no synthesis) + `vitiligo` synthetic | ITA per subject from glabella+malar median L*, b* (`rgb2lab`): ITA = atan2(L*−50, b*)·180/π → 6 classes | Adaptive skin model, pigmentation sub-clusters, POS normalisation | Per-stratum MAE; max abs deviation from population mean ≤ tolerance **pre-specified now** (1.0 bpm or 25 % of population MAE, whichever is larger); vitiligo: MAE with vs without blotches |
| C3 | `multiFace`: paste a second, **real, pulsing** face (another UBFC subject with a different HR) beside/behind the target; variants: bystander enters mid-clip, crosses target | 1–3 bystanders | Designated target + Hungarian + appearance gate | Target-found %, ID switches, MAE; **plus the bystander test**: our HR must follow target reference, and its correlation with the bystander's reference must be low |
| C4 | `lowRes`: `imresize` bicubic | 640×480, 320×240, 160×120 (stress beyond spec) | npix_min scaling, adaptive pooling | MAE vs resolution; graceful = no NaN-only outputs at QVGA |
| C5 | `lowBandwidth`: decimate to 10 / 8 / 6 fps, re-encode MPEG-4 at ≤ 500 kbps | fps × bitrate grid (500, 250 kbps) | Timestamp-aware resampling, fs-aware Kalman/BPF, harmonic term auto-disabled near Nyquist, larger patches (compression macroblocks) | MAE vs fps/bitrate; measured actual bitrate logged (`ffprobe`); on-device mode (no network) demonstrated in web app |
| C6 | `aging`: `imfilter` with edge-emphasising kernels (e.g. `fspecial('laplacian')`/Sobel) applied as localised multiplicative micro-shadow texture on cheeks/forehead | strength 0.05–0.2 | Texture mask, trimmed mean, malar/glabella priority | MAE with vs without; **real elderly subgroup reported separately** (Sri Ramachandra only if a reference becomes available; otherwise GT-free self-consistency on elderly videos, clearly labelled) |
| C7 | `mask`: `insertShape` filled polygon from below the nose to the chin, landmark-aligned, ≥ 40 % of the lower face; variants: surgical (light blue), cloth (dark) | coverage 40–60 % | Occlusion detection → `MASKED` status, re-weight to glabella/forehead/nasal dorsum | MAE masked vs unmasked; % windows correctly flagged MASKED; weight shift plot |

**Implementation notes**
- `imnoise(I,'poisson')` on **double** images treats values as Poisson means scaled by 1e12, which adds almost no noise. For a photon-count model, apply it to a `uint8`/`uint16` image after attenuation, or use `poissrnd(I*Nphot)/Nphot` with an explicit photon budget `Nphot` tied to lux. Document the choice in `API_NOTES.md` after a probe.
- 8-bit quantisation after attenuation is part of the realism. The pulse is ~0.1–1 % of intensity, so at 10 lux it falls below one grey level per pixel and only spatial averaging (with noise acting as dither) recovers it. Say this in the report; it explains why pooling matters.
- `VideoWriter` 'MPEG-4' exposes `Quality`, not an exact bitrate. If the measured bitrate exceeds 500 kbps, re-encode with ffmpeg `-b:v 500k -maxrate 500k -bufsize 1000k`, and record the command and measured bitrate.
- Any real low-light dataset beats simulation. Request **BUAA-MIHR** (illuminance levels down to 1 lux, measured at the face) and **MMPD** (Fitzpatrick III–VI, LED/incandescent/natural light, smartphone) today for Round 2.

---

## 6. Datasets and roles

| Dataset | Role | Notes |
|---|---|---|
| UBFC-rPPG DATASET_2 (42) | Dev + test (split below) | 640×480, contact PPG |
| UBFC-rPPG DATASET_1 (8) | Dev | Finish the batch run |
| AiPEX rppg_biases (88) | Dev 70 % (iteration loop, Section 8.5) + held-out test 30 %; C2 evidence (India vs Sierra Leone) | Pre-cropped ROIs, mean HR per video |
| Sri Ramachandra (organiser) | **No ground truth.** GT-free analyses only (Section 6.1); blind predictions file | Never used for tuning or for accuracy claims |
| Synthetic (known HR) | Unit/integration tests | Already exists |
| Degraded copies of UBFC2 test split | C1, C3, C4, C5, C6, C7 | Generated by `fv.degrade.*` |
| BUAA-MIHR, MMPD, PURE | Round 2 (request access now) | Real low light, darker skin, motion |

---

### 6.1 Using the Sri Ramachandra videos without ground truth
The organiser's data cannot support accuracy claims as it stands. It is still valuable, because it is the data closest to what the judges care about. Owner: Thamil.

**Known facts (26 Nov 2026; reported by the operator, still to be confirmed by `fv.io.loadSriRamachandra` / probe scripts — see `docs/DATASET_SRIRAM.md`):**
- Sample `S_N103_Resting.mp4`: 848×480, ~29.6 fps (steady timestamps), H.264 ~1.4 Mbps, ~32 s, mono AAC audio.
- No ground truth in this file: no sidecar or metadata readings, no monitor or oximeter in frame, and no periodic beeps in the audio (spectrogram). The audio-beep and monitor-OCR routes in Step 1 do not apply to it; still check the other files quickly.
- File names match the organiser's protocol (Arulselvi et al. 2023: rest, walk, stairs, squats, with patient-monitor readings), so the organiser question (Section 1.5, Q7) asks specifically for the patient monitoring system HR and SpO2 per subject and task.
- Recorded outdoors and backlit, so the face is underexposed: treat it as real-world C1 evidence for the GT-free analyses.
- A quick fixed-box POS check gave forehead / left-cheek / right-cheek medians of ~104 / 89 / 64 bpm, region-to-region MAE 24–35 bpm, SNR about −4 dB. Plain POS is unreliable here; use these videos as stress tests for the confidence/status logic.
- **The subject appears to be a minor.** Never put Sri Ramachandra faces in figures, the report, demos or committed files. The data stays out of git (`.gitignore`). Section 1.5, Q8–Q9 ask whether minors are covered and whether faces may be shown at all.

**Step 1 — look for a reference before giving up (half a day).**
- Email the organiser (Section 1.5, question 7). The reference may be withheld deliberately as a hidden test set.
- Inspect every file: metadata sheets, filenames, a separate CSV/XLS folder, the README.
- **Audio track:** many pulse oximeters and monitors beep once per beat. If the videos have audio, a beep-onset detector (band-pass around the beep tone, envelope, peak picking) gives beat-to-beat HR. Validate it on 2–3 videos by counting beeps by hand.
- **Monitor in frame:** if a patient monitor or oximeter display is visible, its HR digits can be read (manual annotation every 5 s is enough, or OCR of a fixed screen ROI). Validate the OCR against manual reads.
- A reference recovered like this is a *derived* reference. It must be described exactly as such in the report, with its own validation, and reported in a separate table from UBFC/AiPEX.

**Step 2 — if no reference exists, run GT-free analyses (all clearly labelled "no ground truth"):**
1. **Blind predictions:** `processed_output.csv` per video in the competition format, ready to submit if the organiser scores it.
2. **Internal agreement:**
   - independent patch groups (glabella+forehead vs left malar vs right malar) estimate HR separately; report their pairwise MAE and limits of agreement;
   - method agreement (POS vs CHROM vs OMIT) on the same traces.

   Agreement is necessary for accuracy but not sufficient; say so.
3. **Signal quality distribution:** SNR, confidence, coverage and status codes per video. Compare these distributions with UBFC2 and AiPEX to show where the hospital data sits in difficulty.
4. **Self-referenced robustness:** for videos where the clean estimate is high-confidence (confidence ≥ 0.8 over ≥ 90 % of windows), apply C1, C4, C5, C6 and C7 degradations. Report the deviation of the degraded estimate from the clean estimate. This isolates the *added* error caused by each constraint without needing a reference. Label it "self-referenced degradation error", never "MAE vs ground truth".
5. **Demographics:** ITA distribution of the cohort (C2), and age subgroup if metadata exists (C6).
6. **Failure gallery:** spectrograms and patch-weight maps of the hardest videos, if the data agreement allows showing faces (Section 1.5, question 8).
7. **Never** tune parameters on this data.

## 7. Metrics (M10)

`fv.metrics.evaluate(est, ref)` returns:
- **MAE, RMSE, Pearson r, MAPE** (pooled over windows *and* mean-over-subjects; report both);
- **Bland–Altman**: bias and 95 % LoA. Because there are multiple windows per subject, use the **repeated-measures LoA** (Bland & Altman 2007, variance components between/within subject) as primary, and per-subject means as secondary;
- **SUC@3, SUC@5**, and **CTA acceptance** (±10 bpm), so results are comparable with Tian 2026 and Li 2026;
- **coverage** (% of windows with an estimate) — never hide NaNs;
- **95 % CIs by subject-level bootstrap** (2,000 resamples);
- **per-stratum tables**: ITA class, country, sex, age band (≥ 60);
- **runtime**: ms/frame per module, total real-time factor.

Window protocol (state it in every table caption): primary = 10 s windows, 1 s step. Secondary = 30 s windows and whole-video, for comparison with published UBFC numbers.

---

## 8. Experiments

### 8.1 Runner
`run_experiment('config/exp_<name>.json')` loads datasets, applies an optional degradation, runs the pipeline with **per-subject caching** (face stage cached keyed by video hash + face config hash; signal stage cached keyed by trace hash + signal config hash), and writes `outputs/<exp>/<timestamp>/` (per-subject CSVs, `summary.json`, `metrics.mat`, figures, `run.log` with git hash and config).

### 8.2 Split and freeze
- **Dev:** UBFC1 (all) + UBFC2 subjects {1, 3, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16} (includes outlier 15 so we can fix it).
- **AiPEX:** dev (70 %, iterated with 5-fold CV) and a frozen held-out test (30 %, stratified by country) — Section 8.5.
- **Test:** the other 30 UBFC2 subjects + AiPEX-test. Sri Ramachandra: GT-free analyses only, never used for tuning (Section 6.1).
- **Freeze parameters on 28 Nov 2026 18:00.** After the freeze, test runs only.
- State in the report that the UBFC2 baseline numbers in Section 2 predate this split.

### 8.3 Required ablations (one table each)
1. ROI set: box fractions (current) vs anatomical canonical vs MediaPipe polygons.
2. Fusion: mean-all vs top-k (k = 1, 3, 5, 8) vs SNR-weighted all.
3. Tracking: argmax vs harmonic argmax vs Viterbi vs Viterbi + motion suppression; online fixed-lag vs offline.
4. Pixel selection: plain box vs adaptive skin vs + texture mask vs + pigmentation clusters.
5. Denoising: none vs Kalman.
6. Method: POS vs CHROM (vs OMIT/LGI if time allows).

### 8.4 Targets (aims, not promises; report what we actually get)
| Metric | Current | Target |
|---|---|---|
| UBFC2 test pooled MAE (10 s) | 2.17 (all 42) | < 1.2 bpm, LoA within ±5 bpm |
| UBFC2 worst subject | 24.8 | < 5 bpm |
| AiPEX MAE (dev CV, then held-out test) | 13.1 | must < 5.8 (beat BKF); very good ≤ 4.0; stretch ≤ 3.0 (Section 8.5) |
| C1 at 50 lux (simulated) | — | < 2× baseline MAE |
| C4 at QVGA | 1.42 (face module) | < 1.5× baseline |
| C5 at 10 fps / 500 kbps | — | < 3 bpm |
| C7 masked | 1.45 | < 2 bpm, ≥ 95 % windows flagged |
| C2 per-ITA deviation | — | within pre-specified tolerance |

### 8.5 AiPEX iteration loop (iterate until results are very good — without fooling ourselves)
AiPEX is where we are weakest (13.1 bpm vs 5.8 bpm for the best published method). It is also our strongest fairness evidence (India vs Sierra Leone, darker skin). Claude Code must **iterate the pipeline on AiPEX in a structured loop** until the targets below are met or the stopping rule fires.

Tuning repeatedly on the same 88 videos and then reporting on them would overfit, and expert judges will notice. The loop is therefore run under a held-out protocol.

**Split (do this first, then never change it):**
- Confirm that each AiPEX video is a distinct subject (check the README, IDs and file names; record the answer in `docs/DATASET_AIPEX.md`). If any subject has several videos, split by subject.
- Hold out **AiPEX-test = 30 % of videos**, stratified by country (13 India + 13 Sierra Leone; adjust if counts differ), chosen with a fixed seed (`rng(2027)`). Write the IDs to `shared/splits/aipex_test.txt`. **Nobody looks at per-video results on AiPEX-test until the final run.**
- **AiPEX-dev = the remaining 70 %** (≈ 62 videos). All iteration happens here, evaluated by **5-fold cross-validation stratified by country**. Any parameter selection happens on the training folds only and is scored on the held-out fold (nested if a search is involved). With no learned model, most changes are fixed pipeline choices, and for those plain 5-fold CV over dev is enough.

**Targets (on AiPEX-dev CV, video-level MAE vs the Masimo mean HR):**
| Level | MAE | Other conditions |
|---|---|---|
| Must | < 5.8 bpm (beats published BKF) | India–Sierra Leone MAE gap ≤ 1.5 bpm |
| Very good | ≤ 4.0 bpm | ≥ 80 % of videos within 5 bpm, bias within ±1 bpm |
| Stretch | ≤ 3.0 bpm | no video > 15 bpm |

The Masimo reference itself carries about ±3 bpm accuracy, and it is a 120 s mean of a varying HR. Below about 3 bpm, gains may not be measurable, and claims must not exceed what the reference supports.

**Experiment log — mandatory.** `outputs/aipex_iter/LOG.md` plus `log.csv`, one row per experiment:
`exp_id, date, git_hash, hypothesis, change, config_diff, cv_mae, cv_mae_ci95, delta_vs_best, p_paired_bootstrap, india_mae, sl_mae, within5_pct, bias, ubfc_dev_mae, decision (keep/reject), notes`.
Every run, including failures, gets a row. The total number of experiments goes into the report.

**One loop iteration:**
1. **State one hypothesis** and the expected effect *before* running (e.g. "Using measured timestamps instead of nominal fps will remove a multiplicative bias").
2. **Change one thing** (a config flag, or a code change behind a flag so the old path stays available).
3. **Run** 5-fold CV on AiPEX-dev, plus a regression check on the UBFC dev split (cached traces, under a minute).
4. **Diagnose:**
   - scatter of estimate vs reference, with slope/intercept;
   - histogram of estimate/reference ratios (look for peaks at 0.5, 2 and 1.2);
   - per-country MAE;
   - the 10 worst videos with their spectrograms and patch SNRs, saved to `outputs/aipex_iter/<exp_id>/`.
5. **Decide:**
   - **Keep** only if CV MAE improves *and* a paired bootstrap over videos (5,000 resamples) gives a 95 % CI of the improvement that excludes 0 (for small changes, p < 0.05). It must also not worsen UBFC dev pooled MAE by more than 0.1 bpm, and must not widen the India–Sierra Leone gap by more than 0.5 bpm.
   - Otherwise **reject** and revert the flag.
6. Write the row, then pick the next hypothesis from the ladder (or from what the diagnosis showed).

**Hypothesis ladder (expected value order; skip items already disproven):**

*Input correctness (Section 2.2, likely the biggest gains):*
1. **Timing.** Use measured per-frame timestamps or the true average fps. Test nominal vs measured. A slope of about 1.2 in the scatter is the signature.
2. **Pre-cropped input.** Bypass face detection; skin mask off or permissive; log the skin ratio.
3. **Patch fusion.** Combine R1 forehead + R2 + R3 cheeks as three patches, using top-k spectral fusion with SNR weights (k ∈ {1, 2, 3}).
4. **Reference alignment.** Compare video-level estimators: averaged normalised spectra vs median of window HRs vs confidence-weighted Viterbi path mean vs trimmed mean. Use the full overlapping duration the reference averages over (match the 120 s).

*Spectral estimation:*

5. **Window length.** Try 10, 15, 20 and 30 s. Longer windows help a one-value-per-video reference.
6. **Harmonic summation** (α ∈ {0, 0.25, 0.5}); a sub-harmonic penalty.
7. **Viterbi tracking** with the physiological limits; λ ∈ {0.05, 0.1, 0.2}.
8. **Band.** The default is 0.7–3.0 Hz. Check the AiPEX reference HR range first; only narrow the band if *every* reference HR fits with ≥ 10 bpm margin, and justify it physiologically (resting, seated adults), never by fitting.

*Signal extraction:*

9. **Method.** POS vs CHROM vs OMIT vs LGI on these traces; POS window 1.2/1.6/2.0 s.
10. **Detrending λ** ∈ {50, 100, 300}; band-pass order 2 vs 4.
11. **Kalman denoising** (Thamil's, the current best single gain) combined with the new chain; R0/Q variants.
12. **Pixel selection inside the crops.** Trimmed mean, texture mask, and pigmentation sub-clusters. These matter for darker skin, where shot noise and a low green-channel signal dominate.
13. **Motion.** The crops may jitter. Estimate inter-frame shift by phase correlation inside each crop, then apply spectral motion suppression (Section 4.8) or motion-compensated pixel sampling.
14. **BKF.** Implement the bounded Kalman filter of Prakash & Tucker (2018), the best published method on this dataset, either as a method option or by borrowing its motion-bounding idea. Use the authors' public code if the licence allows; cite it.
15. **Robust per-video rejection.** If confidence is very low, report "low quality" (the coverage metric) rather than a guess. This is allowed only if reported transparently: give MAE on all videos *and* MAE on confident videos *with* the coverage %. Never drop videos silently.

**Search budget and stopping rule:**
- Grid searches are limited to ≤ 50 configurations per hypothesis group. Prefer one-factor-at-a-time steps with the keep/reject rule over large blind sweeps.
- **Stop** when the "Very good" target is met on dev CV, **or** the ladder is exhausted and 3 consecutive hypotheses give no significant gain, **or** the time box ends (**28 Nov 14:00**, leaving time for the freeze at 18:00).
- When stopping, write `outputs/aipex_iter/SUMMARY.md`: best config, the cumulative improvement waterfall (each kept change and its gain), what failed and why.

**Final evaluation (once, after the freeze):**
- Lock the config (`round1-freeze` tag), run once on **AiPEX-test**, and report test MAE, RMSE, r, Bland–Altman (one value per video, so standard BA is fine), within-5 bpm %, and per-country results with bootstrap CIs. Report the dev-CV numbers next to them.
- If test results are much worse than dev CV, report that honestly and discuss it — do not go back and re-tune.
- **One configuration for all datasets.** The only AiPEX-specific setting allowed is the input adapter (pre-cropped ROIs → patches, no face detection), because that reflects how the data was supplied, not tuning. Any other dataset-specific parameter must be declared in the report as such.

**Report figure:** a "waterfall" chart from 13.1 bpm down to the final dev-CV MAE, one bar per kept change. Add the held-out test result and the published methods (BKF 5.8, CHROM 11.6, POS 14.0, Spherical Mean 15.1, DeepPhys 15.5) for context.

---

## 9. Repository layout (target)

```
/config/                  default.json, exp_*.json
/src/+fv/
  +io/                    readVideo.m, loadUBFC1.m, loadUBFC2.m, loadAiPEX.m, loadSriRamachandra.m, loadGeneric.m
  +face/                  detectFaces.m, TargetTracker.m (class), landmarks.m, motionSignals.m
  +roi/                   canonicalPatches.m, mediapipePatches.m, getPatches.m, overlay.m
  +pix/                   SkinModel.m (class), pigmentClusters.m, textureMask.m, specularMask.m, occlusion.m
  +trace/                 extractTraces.m, kalmanDenoise.m (Thamil), resampleUniform.m, adaptivePooling.m
  +rppg/                  pos.m, chrom.m, omit.m, lgi.m, preprocess.m (detrend+BPF)
  +quality/               snr.m, coherence.m, weights.m
  +fuse/                  spectralFusion.m, waveformFusion.m
  +track/                 harmonicSum.m, motionSuppress.m, viterbiHR.m, fixedLagViterbi.m, tianRule.m
  +metrics/               evaluate.m, blandAltmanRM.m, bootstrapCI.m, stratify.m, ita.m
  +degrade/               lowLight.m, lowRes.m, lowBandwidth.m, multiFace.m, aging.m, mask.m, vitiligo.m
  +pipeline/              runOffline.m, OnlineProcessor.m (class, frame-by-frame, stateful)
  +submit/                buildMat.m, writeCsv.m, packageRound.m
/process_round1.m         competition entry point (thin wrapper over fv.pipeline.runOffline)
/run_experiment.m
/tools/                   diagnose_aipex.m, overlay_patches.m, plot_subject.m, make_report_figures.m
/tests/                   MATLAB unit + integration tests; golden/
/shared/                  patches.json, traceframe.schema.json, config defaults, golden/ (single source for all platforms)
/contrib/                 team contributions, unmodified, with SOURCE.md (Section 15)
/core/                    fvcore C++17 library + bindings (Section 12)
/python/fvfront/          reference front end + fvtool CLI (Section 13)
/web/                     TypeScript PWA (Section 14.1)
/mobile/android/          Android app (Section 14.2)
/pi/                      Raspberry Pi 5 app, fvpi CLI, dashboard (Section 14.3)
/docs/                    API_NOTES.md, FAILURES.md, DATASET_*.md, CHANGELOG_PARAMS.md, report/
/outputs/                 results (git-ignored except summaries)
```

### 9.1 `process_round1.m` contract
```matlab
function out = process_round1(videoPath, varargin)
% PROCESS_ROUND1  FV-TT heart-rate pipeline (IEEE SP Cup 2027, Round 1).
%   out = process_round1('target_video.mp4')
%   out = process_round1(v, 'Reference','reference_signal.csv', 'Mode','offline', ...
%                        'Config','config/default.json', 'OutDir','Round1_Baseline', ...
%                        'Constraint','None', 'Condition','Baseline', 'Params',paramsStruct)
% Writes processed_output.csv and, if a reference is given, RoundX_Condition_Constraint.mat
% with facial_video, roi_data, reference_signal, processed_signal, metrics, params.
```
It must run on a clean machine with only the declared toolboxes (list them in `docs/REQUIREMENTS.md`: MATLAB R2026b, Computer Vision, Image Processing, Signal Processing, Deep Learning (RetinaFace), optionally Python 3.10+ with mediapipe).

---

## 10. Testing

- **Unit tests** per module (synthetic inputs with known answers). Examples:
  - POS on a synthetic pulse plus specular/intensity noise recovers 72 bpm ± 0.3;
  - Viterbi on a synthetic spectrogram with injected 2× harmonic spikes returns the true track;
  - the Kalman −3 dB point equals the spec at 30 and 10 fps;
  - the ITA function matches hand-computed values;
  - each degradation generator writes the right resolution, fps and measured bitrate.
- **Integration test** (existing `test_integration.m`, extended): synthetic face video with a lost-face gap, an occluded cheek, a bystander with a different HR, and low light.
- **Regression test:** 3 UBFC2 dev subjects with cached traces; MAE must not get worse than the stored value by more than 0.1 bpm without a logged reason.
- **Golden vectors** (`tests/golden/`): for 3 subjects, store per-sub-patch RGB traces → POS output → fused spectrum → HR track as `.mat` and `.json`. These drive the fvcore parity tests (Section 12.4). Tolerances: traces 1e-6 relative; HR ±0.1 bpm.

---

## 11. Multi-platform architecture

All four targets — MATLAB, web, Android and Raspberry Pi — share **one algorithm definition** and **one portable signal core**. Only the camera/face/ROI front end differs per platform.

```
                  ┌────────────── platform-specific front end ──────────────┐
 camera/video ──► │ capture + timestamps → face/landmarks → patch masks →    │
                  │ per-patch trimmed-mean RGB, npix, valid ("TraceFrame")  │
                  └───────────────────────────┬──────────────────────────────┘
                                              ▼
                  ┌──────────── fvcore (C++17, shared, Section 12) ──────────┐
                  │ Kalman denoise → resample → POS/CHROM → detrend/BPF →    │
                  │ quality → top-k spectral fusion → harmonic Viterbi →     │
                  │ HR, confidence, status (+ per-patch AC/DC for SpO2)     │
                  └──────────────────────────────────────────────────────────┘
```

| Platform | Front end | Signal core |
|---|---|---|
| MATLAB (reference, competition) | RetinaFace + KLT + canonical patches (or MediaPipe via `pyrun`) | MATLAB `src/+fv` (reference) |
| Desktop Python tool + Raspberry Pi 5 | `python/fvfront` (MediaPipe or YuNet + KLT) | fvcore via pybind11 |
| Web PWA (judge demo) | MediaPipe Tasks Web, TypeScript port of `fvfront` | fvcore → WebAssembly |
| Android | CameraX + MediaPipe Tasks Android, Kotlin port of `fvfront` | fvcore via JNI/NDK |

**Why one hand-written C++ core:**
- There is one implementation to test, so every device produces identical numbers.
- Trace-level golden vectors from MATLAB prove the core matches the reference.

MATLAB Coder–generated C remains an *optional* parallel route on the Pi (Simulink + Raspberry Pi Blockset + PIL) to show the sponsor's workflow in the report. It is not on the critical path.

### 11.1 TraceFrame — the contract between front ends and the core
One record per video frame:
```
TraceFrame {
  double  t_s;              // real capture timestamp, seconds, monotonic (never frame_index/fps)
  uint8   face_status;      // 0 = no face, 1 = tracked, 2 = re-acquired this frame
  float   track_quality;    // 0..1
  float   motion[4];        // dx, dy (normalised by IOD), dscale, droll (rad), vs previous frame
  uint16  n_patches;
  Patch   patches[n_patches];
}
Patch {
  uint16  id;               // stable ID from shared/patches.json
  uint8   group;            // 0=A, 1=B, 2=C priority group
  uint8   cluster;          // 0, or 1/2 when pigmentation sub-clusters are active (Section 4.3)
  float   rgb_mean[3];      // trimmed mean of selected pixels, 0..255, linear sensor RGB
  uint32  npix;             // number of selected pixels
  uint8   valid;            // 1 = usable this frame
  float   skin_ratio;       // selected / total pixels in the patch (occlusion logic)
}
```
- Single sources of truth live in `shared/`:
  - `shared/patches.json` — IDs, names, groups, canonical rectangles (Section 4.2a), MediaPipe polygons (Section 4.2b), 2×2 sub-patch split rule;
  - `shared/traceframe.schema.json`;
  - `shared/config.default.json` — all algorithm parameters; MATLAB and fvcore read the same file;
  - `shared/golden/`.

  No platform may hard-code patch lists or parameters.
- Serialisation:
  - JSON Lines `.tracejsonl` (first line is a header: `{"version":1,"source":...,"resolution":...,"fps_nominal":...,"lux":...,"device":...,"ae_locked":true,...}`);
  - a compact little-endian binary `.trace` with the same header as JSON.

  Readers and writers exist in MATLAB, Python, C++, Kotlin and TS.
- Every platform can **record TraceFrames** (and optionally the raw video). Device runs are compared with MATLAB this way, and `processed_output.csv` is produced from device runs.

### 11.2 Front-end rules (all platforms)
1. **Lock auto-exposure, auto-white-balance and focus** after a 2 s convergence. Auto-adjustments cause step changes far larger than the pulse (≈0.1–1 % of intensity). Log the lock state and values in the trace header.
2. Use the sensor/capture timestamp wherever the API exposes it.
3. Default operating point: 640×480 at 30 fps. Constraint modes: `lowres` 320×240; `lowbw` 10 fps (C4/C5). These are selectable at runtime and written in the header.
4. Target selection (C3): track up to 4 faces. The designated target is the face the user taps or clicks, else the largest face in the first 2 s. Re-acquire only near the last target position, with the colour-histogram appearance gate (Section 4.1).
5. Pixel selection per patch follows Section 4.3 exactly:
   - adaptive (Cb, Cr) Gaussian skin model fitted in the first 3 s on glabella + malar;
   - specular removal;
   - gradient-based texture exclusion (top 15 %);
   - L*-bimodality pigmentation split;
   - trimmed mean 10/10.

   Where full-resolution per-pixel work is too slow on a device, it may run on a decimated patch (every 2nd pixel) *only if* the front-end parity test still passes.
6. **Front-end parity test:** on 50 shared frames from 3 UBFC dev videos, patch masks must reach IoU ≥ 0.8 with the Python reference `fvfront`, and per-patch trace correlation must be ≥ 0.98 over 30 s.

## 12. fvcore — shared C++ signal core

### 12.1 Scope
fvcore implements MATLAB modules M4–M9 (Sections 4.4–4.9) on a stream of TraceFrames:
- Kalman denoising;
- uniform resampling;
- adaptive pooling;
- POS and CHROM;
- smoothness-priors detrending and Butterworth band-pass;
- SNR, coherence and weights;
- top-k spectral fusion;
- harmonic summation, motion suppression, and Viterbi (offline full + online fixed-lag);
- confidence, status codes, video-level HR;
- per-patch AC/DC per channel (for SpO2 in Round 2).

### 12.2 Layout and build
```
core/
  CMakeLists.txt            # C++17, -O2, options: FV_BUILD_TESTS, FV_BUILD_PYTHON, FV_BUILD_WASM, FV_BUILD_JNI
  include/fv/               # public headers
    config.hpp  traceframe.hpp  processor.hpp  result.hpp  fv_c.h (C ABI)
  src/                      # kalman.cpp resample.cpp pos.cpp chrom.cpp detrend.cpp filter.cpp
                            # quality.cpp fusion.cpp viterbi.cpp harmonic.cpp motion.cpp processor.cpp io_trace.cpp
  third_party/              # pocketfft (header-only), nlohmann/json (header-only) — vendored, licences kept
  tests/                    # Catch2: unit + golden parity
  bindings/python/          # pybind11 → module `fvcore`
  bindings/jni/             # JNI glue for Android (com.techtronics.fv.FvCore)
  bindings/wasm/            # Emscripten embind → fvcore.js + fvcore.wasm
```
- No heap allocation per frame after warm-up: ring buffers sized from config (window 10 s + lag 3 s + margin, at max 30 fps, max 80 traces).
- `double` internally; deterministic (no threading inside the core; callers thread around it).
- Band-pass: implement a 4th-order Butterworth as cascaded biquads with coefficients computed at runtime from `fs_proc` (bilinear transform, matching MATLAB `butter`). Offline mode uses zero-phase forward–backward (`filtfilt` equivalent with the same edge padding as MATLAB). Online mode uses a causal filter over the analysis window — document the difference and test both against MATLAB.
- Smoothness-priors detrend: per analysis window, solve `(I + λ² D2ᵀD2) z = x` with a banded (pentadiagonal) solver, O(N). Do not build dense matrices.

### 12.3 API
```cpp
namespace fv {
struct Config { static Config fromJsonFile(const std::string&); static Config fromJsonString(const std::string&); /* fields mirror shared/config.default.json */ };
enum class Mode { Online, Offline };
enum class Status : uint8_t { Warmup, Ok, NoFace, Masked, LowSnr, LowCoverage };
struct HrSample { double t_s; double hr_bpm; double confidence; Status status; int n_patches_used; std::vector<uint16_t> top_patches; };
struct Result { std::vector<HrSample> series; double video_hr_bpm; double video_hr_median_bpm; std::vector<float> bvp; std::vector<double> bvp_t; /* + per-patch AC/DC, diagnostics */ };

class Processor {
 public:
  Processor(const Config&, Mode);
  void push(const TraceFrame&);          // O(n_patches) amortised
  std::vector<HrSample> poll();          // Online: new 1 Hz outputs (delayed by lag)
  Result finalize();                     // Offline: full Viterbi / forward–backward over the whole record
  Diagnostics diagnostics() const;       // last window: per-patch SNR, weights, fused spectrum (for UI)
};
}
```
A C ABI (`fv_c.h`: `fv_create`, `fv_push`, `fv_poll`, `fv_finalize`, `fv_destroy`, plus JSON-string getters for diagnostics) backs JNI, WASM and `ctypes`.

### 12.4 Golden-vector parity (non-negotiable)
- `tools/export_golden.m` (MATLAB, frozen config) writes the following for 3 UBFC dev subjects plus 1 synthetic record into `shared/golden/<id>/`:
  - `input.tracejsonl`;
  - stage outputs: `resampled.json`, `pos.json` (per patch), `quality.json`, `fused_spectrum.json`, `hr_offline.json`, `hr_online.json`.
- Catch2 tests compare each stage, with tolerances:

| Stage | Tolerance |
|---|---|
| Resampled traces | 1e-9 relative |
| POS output | 1e-6 |
| Spectra | 1e-6 relative to peak |
| HR track | ±0.1 bpm per window |
| Status codes | Identical |

- The same golden tests run in CI for: Linux x86-64, Linux aarch64 (the Pi or QEMU), Android NDK (arm64-v8a, via an instrumented test), and WASM (Node).

### 12.5 Performance budget
≤ 1 ms per `push` and ≤ 15 ms per 1 Hz analysis step with 40 patches on a Raspberry Pi 5. A benchmark binary `fvbench` prints both.

## 13. Front-end reference (`python/fvfront`) and desktop tool
- Python 3.11. Dependencies:
  - `mediapipe` (FaceLandmarker task, 478 landmarks, `num_faces=4`);
  - `opencv-python` (YuNet `FaceDetectorYN` fallback, KLT, video I/O);
  - `numpy`.
- Implements Section 11.2:
  - target selection and tracking;
  - patch polygons from `shared/patches.json`;
  - pixel selection (Section 4.3);
  - TraceFrame output;
  - landmark detection every frame on desktop, every 3rd frame on the Pi with KLT propagation of polygon vertices between detections.
- CLI:
  - `fvtool trace --video in.mp4 --out run.tracejsonl [--lowres] [--fps 10]`;
  - `fvtool hr --trace run.tracejsonl --mode offline --out processed_output.csv`;
  - `fvtool eval --est processed_output.csv --ref reference.csv`.

  It uses the same metrics definitions as Section 7, ported, with tests against MATLAB outputs.
- Purpose: the reference front end for the Pi; a cross-check of the MATLAB front end on the same videos; batch processing for the web/mobile parity tests.

## 14. Platform specifications

### 14.1 Web app (PWA, judge demo) — `web/`
- **Stack:** TypeScript + Vite + React. MediaPipe Tasks Vision `FaceLandmarker` runs on the main thread (GPU delegate). Patch rasterisation and pixel selection run in a Web Worker on `ImageData` (OffscreenCanvas). TraceFrames go to fvcore-WASM in the worker.
- **PWA:** a service worker precaches the app, WASM and `.task` model, so it works offline after the first load (C5). No network requests after load; video never leaves the device; a statement on screen says so.
- **Modes:** live webcam; local file (`requestVideoFrameCallback` for true media timestamps); low-bandwidth mode (320×240, 10 fps). Use `MediaStreamTrack.applyConstraints` to request a fixed frame rate and, where supported, manual exposure/white balance (`exposureMode: 'manual'`, `whiteBalanceMode: 'manual'`). Feature-detect, and log whether the lock was achieved.
- **UI:**
  - video with the target box; click to designate the target among several faces (C3);
  - patch overlay coloured by fusion weight, occluded patches greyed out (C7);
  - live BVP waveform; fused spectrum with the Viterbi track;
  - HR, confidence and status badge;
  - ITA readout (C2);
  - lighting warning when the mean face luminance is low (C1);
  - export of `processed_output.csv` and `.tracejsonl`.
- **Tests:** Vitest runs fvcore-WASM on `shared/golden` (must match). Playwright smoke test on a bundled 20 s synthetic clip.
- **iOS:** served as the PWA (Safari, Add to Home Screen) — this is the iOS "app" for this competition.

### 14.2 Android app — `mobile/android/`
- Kotlin, Jetpack Compose, minSdk 26, targetSdk current, arm64-v8a.
- **No `INTERNET` permission in the manifest.** This proves on-device, offline operation (C5). The only permission is CAMERA (plus storage access through the system file picker for exports).
- **Camera:**
  - CameraX `Preview` + `ImageAnalysis` (YUV_420_888, 640×480 or 320×240, `STRATEGY_KEEP_ONLY_LATEST`);
  - `Camera2Interop`: `CONTROL_AE_TARGET_FPS_RANGE = [30,30]` (or [10,10] in `lowbw` mode);
  - after 2 s, set `CONTROL_AE_LOCK = true`, `CONTROL_AWB_LOCK = true`, and lock AF;
  - front camera default, rear selectable;
  - timestamps from `ImageProxy.imageInfo.timestamp` (sensor ns).
- **Front end** (Kotlin port of `fvfront`, same spec):
  - MediaPipe Tasks `FaceLandmarker` in `LIVE_STREAM` mode (`numFaces = 4`, GPU delegate with CPU fallback);
  - patch masks rasterised from the polygons;
  - pixel selection done **directly in YUV**: the skin model lives in (Cb, Cr), which matches YUV; convert only selected pixels to RGB for the trimmed means;
  - per-frame work on a background executor;
  - target designation by tapping a face.
- **Core:** fvcore via JNI (`FvCore.push(frame)`, `poll()`, `diagnostics()`), built with CMake through the Android Gradle plugin.
- **UI:** preview with overlays (target box, weighted patches), HR, confidence, status badge, waveform, spectrum, mode switches (normal / lowres / lowbw), and a session timer.
- **Session recording:**
  - `.tracejsonl` (+ header with device model, resolution, fps, AE/AWB lock values);
  - optional MP4 via `VideoCapture`/MediaRecorder for offline re-processing;
  - `processed_output.csv`;
  - export via the share sheet.
- **Benchmark screen:** ms per frame for landmarks, front end and core; real-time factor; battery drop per minute.
- **Tests:**
  - JVM unit tests for the Kotlin front-end maths;
  - instrumented test that decodes a bundled UBFC dev clip with `MediaCodec`, runs the full pipeline, and compares HR with the golden HR (±1 bpm per window, front-end differences allowed) and the fvcore golden traces (exact tolerances).

### 14.3 Raspberry Pi 5 — `pi/`
- **Hardware:**
  - Raspberry Pi 5 (8 GB), active cooler;
  - Camera Module 3 (CSI) *or* a USB webcam (e.g. Logitech C920);
  - optional 7" display;
  - optional **BH1750 I²C lux sensor** placed next to the face, so every session logs measured illuminance (real C1 evidence instead of only simulation).
- **Software:**
  - Raspberry Pi OS Bookworm 64-bit, Python 3.11;
  - `picamera2` (CSI) or OpenCV V4L2 (USB);
  - `python/fvfront`;
  - fvcore via pybind11 (built natively on the Pi with CMake);
  - FastAPI + WebSocket dashboard.
- **Capture:**
  - Picamera2 `FrameDurationLimits = (33333, 33333)` for 30 fps (100000 for 10 fps);
  - after 2 s of auto mode, read `ExposureTime`, `AnalogueGain` and `ColourGains` from the metadata, then set `AeEnable=False`, `AwbEnable=False` with those values (on USB cameras, use V4L2 `exposure_auto`/`white_balance_temperature_auto` off via `v4l2-ctl`);
  - timestamps from the request metadata `SensorTimestamp`.
- **Process model:**
  - capture thread → bounded queue (drop oldest) → front-end process (MediaPipe every 3rd frame + KLT, or YuNet + KLT fallback when MediaPipe aarch64 wheels are unavailable — verify and record in `docs/API_NOTES.md`) → fvcore in the same process → results queue → dashboard;
  - a watchdog restarts the pipeline on camera errors.
- **CLI (`fvpi`):**
  - `fvpi run --source cam|usb|file.mp4 --mode online|offline --res 640x480|320x240 --fps 30|10 --log runs/`;
  - `fvpi bench` (per-module ms/frame, CPU %, temperature, real-time factor over 5 min);
  - `fvpi replay run.tracejsonl` (deterministic re-run through fvcore).
- **Dashboard:** served on `localhost:8000`. It is also reachable from a phone on the Pi's own Wi-Fi hotspot, with no internet needed. It shows the same views as the web app plus the lux reading.
- **Demo mode:** a systemd service starts on boot in kiosk mode on the display.
- **Targets:**
  - ≥ 15 fps end to end at 640×480 (≥ 10 fps at the C5 operating point);
  - HR every 1 s, first HR ≤ 13 s after the face appears;
  - CPU < 80 %, no thermal throttling over 10 min.
- **Optional sponsor route (only if time allows):** a Simulink model with the Raspberry Pi Blockset camera block and MATLAB Function blocks calling codegen-compatible versions of M4–M9, verified with PIL on the golden traces, with execution time reported next to the fvcore numbers.

### 14.4 Device validation protocol (for the report)
1. **File replay:** each platform processes the same 3 UBFC test videos from file (the Pi via `fvpi run --source file`, Android via the instrumented decode test, web via file mode). Compare HR with MATLAB: per-window difference ≤ 0.5 bpm, or explained by front-end differences measured with the patch-IoU test.
2. **Live:** 5 team volunteers. *Live recordings of people are human-subject data — use only team members who consent, keep them out of the dataset and the report unless IRB covers it, and state this explicitly.* Phone and Pi record simultaneously with a fingertip oximeter as a sanity check. These sessions are demos, not validation results, unless IRB approval exists.
3. **Timing table:** ms/frame per module and real-time factor for each platform, plus model size (MediaPipe task file, fvcore binary size) — this feeds the model/time-complexity criterion.

## 15. Existing contributions and how to integrate them

The team's code already exists. **First action in any Claude Code session: inventory it, do not rewrite it.**

1. Place each contribution unmodified under `contrib/<contributor>/` (e.g. `contrib/sulojan_face_detect/`, `contrib/thamil_kalman/`, `contrib/muftee_signal_chain/`, `contrib/integrated_pipeline/`, `contrib/pi/`, plus any mobile or web code), with a `SOURCE.md` (branch, commit hash, date, author).
2. Write `docs/INVENTORY.md`: for every file, its purpose, entry points, inputs/outputs, dependencies and MATLAB version, the tests it has, and where it maps in the target layout (Section 9).
3. Run each contribution's own tests and scripts as-is. Record what passes or fails. Reproduce the headline numbers from Section 2 (e.g. UBFC2 POS/combined pooled MAE 2.17; face module 1.28 bpm on held-out subjects) before changing anything. If a number does not reproduce, stop and report.
4. Move code into `src/+fv/...` with minimal edits: rename functions to the package namespace, keep the algorithm identical, and keep the author in the file header. Re-run and confirm identical outputs (byte-for-byte on cached traces where possible).
5. The existing browser webcam demo with its MATLAB parity test is the seed for `web/`. The existing Pi code is the seed for `pi/`: reuse its capture/display parts, and adapt it to emit TraceFrames and to call fvcore.
6. Only then extend with the new modules (Sections 4.2–4.8, 12–14).
7. The report credits each contributor's component (Sulojan: face module; Thamil: Kalman denoising and AiPEX evaluation; Muftee: signal chain; the Pi contributor: Pi implementation; others as applicable).

## 16. Execution plan for one operator with Claude Code

Run **three Claude Code sessions in parallel**, each in its own `git worktree` and branch, merging to `dev` daily.

| Session | Branch | Scope |
|---|---|---|
| **A — Reference & evaluation** (critical for Round 1) | `wt/matlab` | Sections 2.2, 4–8, 10, `process_round1.m`, report figures |
| **B — Core, Python front end, Pi** | `wt/core-pi` | Sections 11–13, 14.3 |
| **C — Web and Android** | `wt/apps` | Sections 14.1, 14.2 (starts on the interim POS path; switches to fvcore when B publishes bindings) |

**Interim rule:** until fvcore passes the golden tests, sessions B and C use a minimal POS + FFT implementation in their own language behind the same interface. Then they swap to fvcore with a one-line change.

### Phase 0 — 26 Nov (all sessions)
- [ ] Repo set up; contributions inventoried and reproduced (Section 15); `shared/` created (patches.json, schema, config, golden folder).
- [ ] Organiser questions sent (Section 1.5), including Sri Ramachandra ground truth.
- [ ] A: AiPEX split written (Section 8.5); AiPEX diagnosis (Section 2.2) run on AiPEX-dev only; UBFC outliers labelled.
- [ ] B: fvcore skeleton builds; TraceFrame I/O in C++ and Python; Pi capture with AE/AWB lock emitting TraceFrames.
- [ ] C: web skeleton with MediaPipe + patch overlay; Android CameraX with AE/AWB lock + landmarks.

### Phase 1 — 27–28 Nov
- [ ] A: **AiPEX iteration loop (Section 8.5) until the target or the stopping rule, time box 28 Nov 14:00**; M2–M8 in MATLAB; unit tests; dev ablations; **freeze 28 Nov 18:00**, tag `round1-freeze`; export golden vectors.
- [ ] B: fvcore POS/detrend/BPF/FFT passing golden tests; pybind11; `fvtool`; Pi live HR (interim) with `fvpi bench`.
- [ ] C: web and Android emitting valid TraceFrames with interim HR; file-replay mode.

### Phase 2 — 28–29 Nov
- [ ] A: test-split runs (baseline, C1–C7, AiPEX-test once), stratification, repeated-measures Bland–Altman, bootstrap CIs, Sri Ramachandra GT-free study, all figures.
- [ ] B/C: device file-replay comparison on 3 UBFC test videos (Section 14.4, step 1); timing tables; screenshots.

### Phase 3 — 29–30 Nov — submission
- [ ] `process_round1.m` on a clean checkout; `.mat` files (six variables each); CSVs; video.
- [ ] Report including an "Edge deployment" section: architecture, TraceFrame contract, Pi and Android timing, screenshots, and offline operation (no-internet manifest, PWA).
- [ ] Upload with margin.

### Phase 4 — 1–20 Dec — platform parity
- fvcore complete (Kalman, adaptive pooling, quality, fusion, harmonic Viterbi offline + fixed-lag, status), passing golden tests on x86-64, aarch64, Android and WASM.
- All three apps on fvcore; device validation protocol complete; lux sensor on the Pi.

### Phase 5 — until 16 Jan — Round 2 (HR + SpO2)
- BUAA-MIHR / MMPD; own data only under IRB.
- SpO2 from fvcore per-patch AC/DC ratio-of-ratios, calibrated and validated out of sample.
- Live demos on Pi and phone under ≥ 2 constraints.

### Phase 6 — Feb–May — Round 3
- BP from BVP morphology/timing features; clinical validation plan; demo hardening; presentation.

### 16.1 First prompt for each session (copy into Claude Code)
> Read `README.md` fully. Then do Phase 0 for Session [A|B|C] only. Start with Section 15 (inventory and reproduce the existing contributions) and do not modify any algorithm before the headline numbers reproduce. Show me `docs/INVENTORY.md` and the reproduction results before writing new code. Ask me whenever the README is ambiguous or contradicts the existing code.

## 17. Round 1 report outline (IEEE conference format)
1. **Introduction:** problem; equity and edge motivation; contributions (bullets):
   - anatomical multi-patch selection grounded in Li 2026;
   - pigmentation-aware sub-clusters for vitiligo;
   - texture-masked trimmed pooling for aging skin;
   - top-k SNR spectral fusion;
   - physiology-constrained harmonic Viterbi tracking with offline/online variants;
   - a reproducible C1–C7 stress suite using the official recipes;
   - an on-device web demo.
2. **Related work** (short): CHROM/POS/OMIT/LGI; ROI studies (Kim 2021, Li 2024/2026, Bondarenko 2025); correction (Tian 2026); bias (Dasari 2021); elderly (Caroppo 2024); the organiser group's prior work (Jeyakumar 2020/2022, Arulselvi 2023) acknowledged respectfully.
3. **Method:** one figure of the pipeline plus one subsection per module; equations for POS, SNR, fusion and the Viterbi cost.
4. **Datasets and protocol:** splits, freeze date, window protocol, reference derivation, degradation recipes with exact parameters.
5. **Results:**
   - baseline table (MAE, RMSE, r, bias, LoA, SUC, coverage, CI);
   - Bland–Altman plot;
   - per-subject bar chart;
   - C1–C7 table + curves (MAE vs lux, resolution, fps/bitrate);
   - ITA/country/age stratification;
   - ablations;
   - runtime.
6. **Edge deployment:** architecture (TraceFrame + fvcore), Pi/Android/web timing, offline operation, device-vs-MATLAB agreement.
7. **Discussion:** failure cases (with spectrograms), limitations (simulation vs real low light; no real vitiligo subjects yet; arrhythmia and the Viterbi prior, as Tian notes), Round 2/3 plan.
8. **Ethics and data statement:** datasets used, licences, the organiser agreement, no own human data in Round 1.

---

## 18. Risks and mitigations
| Risk | Mitigation |
|---|---|
| R2026b/RetinaFace not installed in time | Cached face traces + Viola-Jones/`faceDetector` fallback; state it |
| Viterbi over-smooths real fast HR changes / arrhythmia | Hard limits from physiology only; report fixed-lag vs offline; flag in limitations |
| Over-tuning on UBFC2 or AiPEX | Freeze date; AiPEX iterated only on the dev split with CV and a significance rule (Section 8.5); held-out AiPEX-test run once; Sri Ramachandra never used for tuning |
| Sri Ramachandra ground truth never arrives | GT-free analyses (Section 6.1); accuracy claims rest on UBFC/AiPEX only |
| Platform ports drift from the MATLAB reference | Trace-level golden vectors and parity tests in CI (Section 12.4) |
| Phone/Pi auto-exposure and white balance corrupt the pulse signal | Lock AE/AWB after convergence (Sections 11.2, 14.2, 14.3) |
| AiPEX root cause not found in time | Report honestly with the diagnosis attempted; it is still the best C2 evidence available |
| `.mat` size too large | Ask the organiser; store `facial_video` for the baseline subject, store paths + hashes for others if allowed |
| Simulated low light unconvincing | Request BUAA-MIHR now; in Round 2 record real ≤ 50 lux sessions under IRB |
| Muftee's code stays local | Push today; if blocked, zip and add to the repo via another member |

## 19. References (for the report)
- de Haan & Jeanne, IEEE TBME 2013 (CHROM).
- Wang et al., IEEE TBME 2017 (POS).
- Pilz et al., CVPRW 2018 (LGI).
- Álvarez Casado & Bordallo López, IEEE JBHI 2023 (OMIT).
- Tarvainen et al., IEEE TBME 2002 (detrending).
- Bondarenko, Menon & Elgendi, npj Digit Med 8:479 (2025).
- Li, Elgendi, Zhu, Lam & Menon, npj Cardiovasc Health 3:40 (2026).
- Tian, Li, Zhu, Elgendi & Lam, npj Digit Med 9:233 (2026).
- Caroppo et al., Multimed Tools Appl 83 (2024).
- Dasari et al., npj Digit Med 4:91 (2021).
- Pham et al., Proc ACM IMWUT 10(3) Art. 147 (2026).
- Altman & Bland, The Statistician 32 (1983); Bland & Altman, J Biopharm Stat 2007 (repeated measures).
- Kim, Lee & Sohn, Sensors 2021.
- Li, Elgendi & Menon, npj Cardiovasc Health 2024.
- Jeyakumar et al. 2022 (Cyber-Physical Systems ch. 5); Ruba et al., ICADEE 2020; Arulselvi et al., DELCON 2023.
- Datasets: Bobbia et al. 2019 (UBFC-rPPG); Dasari et al. 2021 (AiPEX); Xi et al. 2020 (BUAA-MIHR); Tang et al. 2023 (MMPD).
