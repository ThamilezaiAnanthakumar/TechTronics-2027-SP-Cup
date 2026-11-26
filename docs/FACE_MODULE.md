# TechTronics — IEEE SP Cup 2027 (FaceVitals)

2027 SP Cup, Team TechTronics, Department of Electronic and Telecommunication Engineering, University of Moratuwa, Sri Lanka.

**Challenge:** contactless estimation of Heart Rate, SpO₂ and Blood Pressure from RGB facial video (remote photoplethysmography, rPPG).
**Round 1:** Heart Rate only. Deadline **30 November 2026**.

```
Video ─► Face detection & tracking ─► ROI + skin mask ─► RGB traces ─► rPPG signal ─► HR ─► Metrics
        └──────────── face module (this code, done) ────────────┘   └─ signal processing (open) ─┘
```

---

## 1. Status

**Done: face / ROI module** (`src/face`, `src/dataio`), from video to per-frame RGB traces of the target face.
- [x] Face detection with RetinaFace (`faceDetector`), every N frames.
- [x] Multi-face tracking: Kalman filter per face and Hungarian matching. One designated target, never swapped for a bystander (**C3**).
- [x] Target followed every frame by KLT points and a similarity transform, re-anchored on the detector when it drifts.
- [x] ROI polygons as fractions of the face box, moved with the head. The regions are configurable: default forehead + left/right cheek.
- [x] One-Euro smoothing of the ROI outlines (less jitter).
- [x] YCbCr skin mask, validated on dark skin (AiPEX India / Sierra Leone).
- [x] Per-region occlusion flag from the skin ratio, with an absolute and a relative rule (**C7**). `rgb_face` averages only non-occluded regions.
- [x] Works at VGA and QVGA (**C4**).
- [x] Frame timestamps read from the video (never `k/fps`), and no frame is dropped.
- [x] One `.mat` trace file per video (`-v7`, readable from MATLAB and Python).
- [x] All tunable parameters in `config/face_config.json`.
- [x] CPU detection by default, so results are bit-reproducible.

**Done: evaluation tooling** (`src/degrade`, `src/eval`, `src/tools`)
- [x] Test-clip generator from UBFC: QVGA (C4), MultiFace (C3), Masked (C7), Stress (all three).
- [x] Benchmark: tracking, jitter, occlusion and speed metrics, plus a **baseline HR** (POS) against the contact PPG (MAE, RMSE, r, SUC@3/5, Bland–Altman).
- [x] Debug overlay videos.

**Open (next steps)**
- [ ] Signal-processing module (`src/signal`): a stronger HR method, peak tracking across windows, and use of `quality` / `occluded`.
- [ ] Competition deliverables: `process_round1.m`, `processed_output.csv`, the `.mat` submission structure.
- [ ] C5 (≤ 10 fps, ≤ 500 kbps) clips. The module reads real timestamps, so it should accept them unchanged, but this is untested.
- [ ] Raspberry Pi 5 timing.
- [ ] Organizer dataset (Sri Ramachandra) once available.

---

## 2. Requirements

- MATLAB **R2025a or newer** (developed on R2026b).
- Computer Vision, Image Processing, Deep Learning and Signal Processing Toolboxes.
- Add-on **Computer Vision Toolbox Model for RetinaFace Face Detection** (Home → Add-Ons). It provides `faceDetector`.

## 3. Quick start

From the repo root in MATLAB:
```matlab
addpath(genpath("src"))

% one video -> one trace file
out = process_video("path/to/video.avi", Subject="subject1", Condition="Baseline", Constraint="None");
export_traces(out);                       % -> outputs/traces/UBFC2_subject1_Baseline_None.mat

% full benchmark: test clips + traces + tables + debug videos
T = run_pipeline(Quick=true);             % first 10 s per video (smoke test)
T = run_pipeline();                       % full videos, UBFC subjects [1 3 4 12 17 26 31 34]
T = run_pipeline(Config="my.json", Rerun=true);
```
Python, to read a trace:
```python
from scipy.io import loadmat
S = loadmat("outputs/traces/UBFC2_subject1_Baseline_None.mat", squeeze_me=True, struct_as_record=False)
t, rgb = S["t"], S["rgb_face"]
```

---

## 4. How the face module works

Per frame (`process_video`):
1. **Timestamp.** `t(k) = VideoReader.CurrentTime`, read *before* `readFrame`.
2. **FaceTracker.step** (`src/face/FaceTracker.m`):
   - **KLT** moves the target and gives a 3×3 similarity transform `A` (anchor frame → current frame).
   - Every `det.every` frames, **RetinaFace** detects all faces, and **Hungarian assignment** on `1 − IoU` updates one Kalman track per face.
   - The **target** is chosen once (largest + most central face, or `trk.targetBox`). If the KLT box drifts from the detection (IoU < 0.75), the target is re-anchored.
   - If KLT fails, the tracker searches a window around the last position. If nothing is found, the target is **lost** (`quality = 0`) and is re-acquired only within `trk.reacqDist` face widths of its last position. This is what prevents a jump to a bystander.
3. **ROI polygons** (`roi_polygons`). Each region is a box `[u1 v1 u2 v2]` in face-box fractions of the anchor box, moved by `A`, so it stays on the same skin.
4. **Smoothing.** A One-Euro filter on the polygon corners.
5. **Skin pixels** (`roi_skin_stats`, `skin_mask`). YCbCr box per pixel inside each polygon, giving mean RGB, `npix` and `skin_ratio`.
6. **Occlusion.** A region is occluded when its smoothed skin ratio is **below `occl.onBelow`** *and* **below `occl.relOn` × the reference region's ratio** (default reference: forehead). Both rules use hysteresis.
7. **`rgb_face`.** The pixel-weighted mean of all non-occluded regions.

### Code map
| File | Role |
|---|---|
| `config/face_config.json` | all user-tunable parameters |
| `src/face/face_params.m` | defaults + loads and validates the JSON |
| `src/face/process_video.m` | per-frame loop, builds the output struct |
| `src/face/FaceTracker.m` | detection schedule, Kalman/Hungarian tracks, target choice, KLT |
| `src/face/detect_faces.m` | RetinaFace wrapper (full frame or window) |
| `src/face/roi_polygons.m`, `apply_tform.m` | region geometry |
| `src/face/skin_mask.m`, `roi_skin_stats.m` | skin classification, per-region statistics |
| `src/face/OneEuroFilter.m` | ROI smoothing |
| `src/face/box_iou.m`, `clip_box.m` | box helpers |
| `src/dataio/export_traces.m`, `trace_name.m`, `to_char_struct.m` | save trace files (`-v7`, scipy-readable) |
| `src/dataio/read_video_header.m`, `read_frame_times.m`, `load_ubfc_gt.m` | video and ground-truth loaders |
| `src/degrade/make_test_clips.m`, `clip_conditions.m`, `draw_mask.m`, `scan_target_boxes.m` | C3 / C4 / C7 test clips + reference boxes |
| `src/eval/run_benchmark.m`, `benchmark_metrics.m`, `bland_altman.m` | metrics, tables, figure |
| `src/eval/sanity_hr.m`, `ppg_hr.m`, `spectral_hr.m` | baseline HR (POS) and reference HR from the contact PPG |
| `src/tools/debug_video.m`, `video_info.m`, `repo_root.m` | overlay video, video facts, paths |
| `src/run_pipeline.m` | entry point: clips → traces → benchmark → debug videos |

---

## 5. Configuration (`config/face_config.json`)

Any field can be left out; the defaults from `face_params.m` are used. The exact values used are stored in each trace's `meta.params`.

| Key | Meaning |
|---|---|
| `roi.names` | **regions in use**, e.g. `["forehead","lcheek","rcheek"]` (default) |
| `roi.<name>` | region box as face-box fractions `[u1 v1 u2 v2]` (u: left→right, v: top→bottom). Predefined but unused: `glabella`, `nose`, `lmalar`, `rmalar` |
| `det.model`, `det.threshold` | RetinaFace size and confidence |
| `det.every` | full-frame detection period in frames (KLT tracks in between) |
| `det.device` | `"cpu"` (default, bit-reproducible) or `"gpu"` / `"auto"` (faster, small run-to-run differences) |
| `trk.targetBox` | optional manual target box `[x y w h]` in the first frame |
| `trk.reacqDist` | re-acquisition radius after target loss, in face widths |
| `skin.cb`, `skin.cr`, `skin.yMin`, `skin.satMax` | YCbCr skin thresholds. Widened from the textbook box (Cb 77–127, Cr 133–173) to Cb 77–135, Cr 128–173, yMin 20 after testing on darker skin (AiPEX India / Sierra Leone): accepted skin went from 93 → 97 % (India) and 86 → 97 % (Sierra Leone), and a blue face mask is still rejected |
| `occl.onBelow`, `occl.offAbove` | absolute skin-ratio thresholds for the occlusion flag |
| `occl.reference`, `occl.relOn`, `occl.relOff` | reference region and relative thresholds |
| `smooth.*` | One-Euro filter (`enable`, `minCutoff`, `beta`) |
| `clip.*` | test-clip generation (bystander size, mask shape and colour) |

To add a region, define it under `roi` and list it in `roi.names`. For example,
`"names": ["forehead","lcheek","rcheek","glabella"]` gives an extra output `rgb_glabella` and one more column in `npix` / `occluded`.

---

## 6. Output (input for signal processing)

There is one `.mat` file per video: `outputs/traces/<dataset>_<subject>_<Condition>_<Constraint>.mat`.
N is the number of frames (**no frame is dropped**), and R is the number of regions.

| Variable | Shape | Meaning |
|---|---|---|
| `t` | N×1 | frame time in seconds, **read from the video** (not k/fps) |
| `rgb_<region>` | N×3 | mean R,G,B (0–255) of the skin pixels in that region, target face only (cheeks: viewer's side) |
| `rgb_face` | N×3 | pixel-weighted mean over the **non-occluded** regions (the recommended single trace) |
| `roi_names` | 1×R | region names, which give the column order of `npix`, `skin_ratio`, `occluded` |
| `npix`, `skin_ratio` | N×R | skin pixels used, and the skin share of each region |
| `occluded` | N×R | 1 = region occluded (e.g. mask): down-weight or drop it |
| `quality` | N×1 | 0 = no target (RGB is NaN), 1 = tracked, 2 = confirmed by the detector |
| `n_faces`, `target_id` | N×1 | faces visible, and the track ID of the target |
| `roi_data` | N×1 struct | `face_box` [x y w h], one 4×2 polygon per region, `all_boxes`, `track_ids` |
| `meta` | struct | source, condition, constraint, resolution, fps, timing, and the `params` used |

Rows with `quality = 0` are NaN. Interpolate across them or mask them out before filtering.

---

## 7. Results: face module + baseline signal (POS)

UBFC-rPPG DATASET_2, reference HR from the contact PPG, 10 s windows every 1 s, CPU, config v1.2.
There is **no training step**: all parameters are fixed by hand. Subjects 1 and 17 were looked at while
setting parameters (ROI placement, relative occlusion rule), so they form the **tuning set**. The other
six subjects were never used for any decision and form the **test set**.

### Test set: held-out subjects 3, 4, 12, 26, 31, 34 (318 windows per condition)

| Condition | Target found | ID switches | HR MAE (bpm) | RMSE | r | SUC@3 | SUC@5 | Bland–Altman bias [limits of agreement] | Occlusion accuracy | ms/frame |
|---|---|---|---|---|---|---|---|---|---|---|
| Baseline (640×480) | 100 % | 0 | **1.28** | 2.14 | 0.86 | 92.5 % | 96.8 % | −0.46 [−5.6, +4.7] | 100 % | 8.8 |
| QVGA 320×240 (C4) | 100 % | 0 | **1.42** | 2.36 | 0.84 | 91.2 % | 96.0 % | −0.57 [−6.1, +5.0] | 100 % | 6.2 |
| MultiFace (C3) | 100 % | 0 | **1.28** | 2.15 | 0.86 | 92.5 % | 96.8 % | −0.46 [−5.6, +4.7] | 100 % | 9.1 |
| Masked (C7) | 100 % | 0 | **1.45** | 2.40 | 0.84 | 91.1 % | 95.7 % | −0.60 [−6.3, +5.1] | 100 % | 7.4 |
| Stress (C3 + C4 + C7) | 100 % | 0 | **1.48** | 2.47 | 0.83 | 90.8 % | 95.3 % | −0.62 [−6.4, +5.2] | 100 % | 6.0 |

### All 8 subjects: 1, 3, 4, 12, 17, 26, 31, 34 (tuning + test, 420 windows per condition)

| Condition | Target found | ID switches | HR MAE (bpm) | RMSE | r | SUC@3 | SUC@5 | Bland–Altman bias [limits of agreement] | Occlusion accuracy | ms/frame |
|---|---|---|---|---|---|---|---|---|---|---|
| Baseline (640×480) | 100 % | 0 | **1.42** | 2.80 | 0.78 | 92.6 % | 97.0 % | −0.64 [−7.9, +6.6] | 100 % | 9.0 |
| QVGA 320×240 (C4) | 100 % | 0 | **1.53** | 2.96 | 0.76 | 91.6 % | 96.4 % | −0.72 [−8.2, +6.8] | 100 % | 6.1 |
| MultiFace (C3) | 100 % | 0 | **1.43** | 2.80 | 0.78 | 93.2 % | 97.0 % | −0.64 [−7.9, +6.6] | 100 % | 8.7 |
| Masked (C7) | 100 % | 0 | **1.40** | 2.57 | 0.79 | 92.5 % | 96.6 % | −0.57 [−6.6, +5.4] | 100 % | 7.4 |
| Stress (C3 + C4 + C7) | 100 % | 0 | **1.51** | 2.86 | 0.76 | 91.8 % | 96.0 % | −0.66 [−7.5, +6.2] | 100 % | 5.9 |

**How it was evaluated:**
- **Test clips:**
  - QVGA = bicubic resize.
  - MultiFace = a second real UBFC subject pasted beside the target at 75 % size.
  - Masked = a synthetic light-blue mask polygon over more than 40 % of the lower face.
- **HR:** `rgb_face` is resampled to 30 Hz on the real timestamps, then POS, a 0.7–3 Hz Butterworth filter, 10 s windows every 1 s, and the periodogram peak. The reference uses the same steps on the contact PPG (`ground_truth.txt`, line 1).
- **MAE, RMSE, r:** mean over subjects. **SUC@k:** % of windows with |error| < k bpm. **Bland–Altman:** pooled over all windows of the set.
- **Per-subject Baseline MAE (bpm):** s34 0.30, s26 0.39, s31 0.71, s12 1.04, s1 1.47, s17 2.26, s4 2.38, s3 2.84.
- **The test set scores no worse than the full set** (MAE 1.28 vs 1.42), so the hand tuning did not inflate the results.
- **Tables:** test set in `outputs/tables/heldout/`, all subjects in `outputs/tables/`.
- **ID switches:** frames where the output box overlaps the bystander more than the target. **Occlusion accuracy:** per-region flags against the expected pattern.

**Findings (test set):**
- **C3:** the bystander has no effect (0 switches; HR identical to Baseline).
- **C4:** QVGA costs +0.14 bpm.
- **C7:** occluded cheeks are flagged in 100 % of frames. HR comes from the uncovered regions (+0.17 bpm).
- **All three combined:** +0.20 bpm.
- **Limits of agreement:** about ±5 bpm on the test set. They widen to about ±7 bpm on all 8 subjects because of a few windows where the plain spectral peak locks onto the wrong frequency (mainly subject 17, whose HR varies). That's a signal-processing limitation, not a face-module error.

### Design decisions and lessons
- **CPU detection by default.** On a GPU, RetinaFace output varies slightly between runs, which changed some re-anchor decisions and shifted HR by up to 0.45 bpm. On CPU, runs are bit-identical.
- **Relative occlusion rule.** The absolute rule alone flagged pale cheeks (UBFC subject 17) as occluded. Requiring the region to be well below the forehead fixed it (HR 3.32 → 2.83 bpm) without missing any mask frame.
- **Skin thresholds widened** after the AiPEX dark-skin check (see section 5). This improved HR in every condition (Baseline 1.53 → 1.42 bpm).
- **ROI choice.** Forehead + cheeks by default. Li et al. (npj Cardiovasc. Health 2026) rank the glabella, nose bridge and upper cheeks (malar) highest. These are predefined in the config, but their box fractions are untuned estimates.

### Known limitations
- The ROI fractions were tuned on frontal UBFC faces. Large head rotations are not tested.
- The synthetic mask is a solid light blue. A skin-coloured cloth mask would be harder to detect.
- UBFC contains few dark-skinned subjects. Dark skin was checked only at the pixel level (AiPEX crops), not end-to-end.
- The baseline HR has no peak tracking, so outlier windows widen the limits of agreement.

---

## 8. Repository layout

```
├── config/face_config.json   # face-module parameters
├── src/
│   ├── face/                 # detection, tracking, ROI, skin mask, occlusion (core module)
│   ├── dataio/               # loaders, trace export
│   ├── degrade/              # C3 / C4 / C7 test-clip generation
│   ├── eval/                 # benchmark, baseline HR, Bland–Altman
│   ├── tools/                # debug video, video info
│   ├── signal/               # signal processing (open)
│   └── run_pipeline.m        # entry point
├── data/                     # gitignored: raw/ datasets, degraded/ test clips
├── outputs/                  # gitignored: traces/, tables/, figures/
├── deploy/pi5/  tests/  report/  submission/  docs/
```

## 9. Data

Datasets live in `data/raw/` (gitignored; never commit videos or `.mat` data):
```
data/raw/UBFC/DATASET_2/subjectN/vid.avi
data/raw/UBFC/DATASET_2/subjectN/ground_truth.txt
data/raw/AiPEX/Data/{India,Sierra_Leone}/videoN_{forehead,leftcheek,rightcheek}.avi
```
- **UBFC:** subjects used: 1, 3, 4, 12, 17, 26, 31, 34. Excluded: 27 (no image consent; never show its face) and 11, 18, 20, 24 (pulse-oximeter dropouts). Cite S. Bobbia et al., *Pattern Recognition Letters*, 2019.
- **AiPEX rppg_biases:** pre-cropped ROI clips only (no full faces) plus a mean HR per video. Used to check the skin mask on darker skin. Cite A. Dasari et al., *npj Digital Medicine*, 2021.
- **Organizer dataset:** covered by a data agreement. Keep it on team machines only.
