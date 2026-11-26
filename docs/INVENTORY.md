# Code inventory

Date: 2026-11-26. Inventoried at `dev` = `46790ae` (= `integrated_pipeline` 0792a67 + a docs-only commit that made the spec the root `README.md` and moved the old face-module README to `docs/FACE_MODULE.md`).

| Contributor | Branch | Commit(s) | Date | Scope |
|---|---|---|---|---|
| Sulojan | `face_detect` (= `origin/face_detect`) | 44e89fc, 808b938, 921910d (on top of e6179f9 initial structure) | 2026-11-19 to 11-22 | face/ROI module, dataio, test-clip generation, face benchmark |
| Thamil (Thamilezai Ananthakumar) | `thamilezai_dev` (= `origin/thamilezai_dev`) | f629042 | 2026-11-24 | Kalman denoising, AiPEX baseline-vs-denoise check |
| Muftee | `muftee_dev` (local only) | f9d4155 | 2026-11-26 | signal chain, loaders, run scripts, web demo |
| Integration (Muftee) | `integrated_pipeline` | db91ae5 (merge), 0792a67 | 2026-11-26 | `src/pipeline/*`, `run_ubfc2.m`, `tests/test_integration.m`, edits to `getConfig.m` and `computeMetrics.m` |

Authorship was checked per file with `git log -- <file>`. Sulojan's and Thamil's files on `dev` are byte-identical to their branches (`git diff face_detect dev`, `git diff thamilezai_dev dev`). Muftee's files differ from `muftee_dev` only in `config/getConfig.m` (lines 62-73 added) and `src/eval/computeMetrics.m` (skips NaN pairs; adds SUC3/SUC5/coverage), both changed in 0792a67. The code has three separate front ends (Viola-Jones in `extractRGBTraces`, RetinaFace in `process_video`, and AiPEX crop readers), two configs and two sets of HR, reference and metric functions. Only the integrated path (`run_ubfc2` → `process_video` → `kalman_denoise` → Muftee's chain) joins all three contributions. Nothing has been moved to `src/+fv/` yet, and no code has been run for this inventory.

Abbreviations: **CVT** Computer Vision Toolbox, **IPT** Image Processing Toolbox, **SPT** Signal Processing Toolbox, **Stats** Statistics and Machine Learning Toolbox, **PCT** Parallel Computing Toolbox. **RetinaFace** = `faceDetector`, which needs MATLAB R2025a+ and the support package "Computer Vision Toolbox Model for RetinaFace Face Detection". **TS** = `tests/test_synthetic.m`, **TI** = `tests/test_integration.m`, **PJ** = `deploy/web/tests/parity.mjs` (indirect: compares the JS port against MATLAB output). Other version floors: `arguments` blocks need R2019b, `Name=Value` call syntax R2021a, `estgeotform2d` R2022b, `theme(fig)` R2025a. Sulojan's README states R2025a+ (developed on R2026b).

## 1. Sulojan (`face_detect`) — 36 files

| File | Purpose | Entry point? | Inputs → outputs | Dependencies | Tests | Target (Section 9) |
|---|---|---|---|---|---|---|
| `.gitattributes` | LF line endings; marks binary types | – | – | – | – | keep at root |
| `.gitignore` | ignores data/, outputs/, videos, MATLAB/Python/LaTeX artefacts | – | – | – | – | keep at root (see "Data/" case issue, C17) |
| `docs/FACE_MODULE.md` | face-module README (identical to `face_detect:README.md`; moved in 46790ae) | – | – | – | – | `docs/` (keep) |
| `config/face_config.json` | face-module parameters (det, trk, roi, skin, occl, smooth, clip), v1.2 | – | read by `face_params` | – | – | `config/default.json` (face section) |
| `src/face/face_params.m` | built-in face defaults merged with the JSON; validates ROIs | `p = face_params(configPath)` | JSON path → param struct | `repo_root`; core `jsondecode` | none (TI builds `fv` by hand) | `+fv` config loader for `config/default.json` / `shared/` |
| `src/face/process_video.m` | face-module main loop: video → per-frame ROI RGB of the target face, occlusion, quality, roi_data, timing | `out = process_video(videoPath, Name=Value)` | video path → struct `t, rgb_<roi>, rgb_face, npix, skin_ratio, occluded, quality, n_faces, target_id, roi_data, meta, debug` | `face_params, FaceTracker, OneEuroFilter, roi_polygons, roi_skin_stats, repo_root`; RetinaFace (when no Detector is passed); core `VideoReader`; `git` via `system` | none (TI uses a synthetic struct shaped like its output) | `+fv/+trace/extractTraces.m` (frame loop) + `+fv/+pipeline/OnlineProcessor.m` |
| `src/face/FaceTracker.m` | multi-face tracker (Kalman + Hungarian on 1-IoU) plus target KLT similarity tracking, re-acquisition | class: `FaceTracker(fd,p,[H W])`, `st = step(I,t)` | frame, time → state (quality, boxes, anchorBox, A) | `detect_faces, box_iou, clip_box, apply_tform`; CVT (`configureKalmanFilter`, `assignDetectionsToTracks`, `vision.PointTracker`, `detectMinEigenFeatures`, `estgeotform2d`) | none | `+fv/+face/TargetTracker.m` |
| `src/face/OneEuroFilter.m` | 1-Euro adaptive low-pass for ROI coordinates | class: `update(x,t)`, `reset()` | row vector, t → smoothed vector | none | none | `+fv/+face/` (helper; no named slot) |
| `src/face/detect_faces.m` | RetinaFace detection, whole frame or ROI, optional upscale | `[bboxes,scores] = detect_faces(fd,I,p,roi)` | uint8 frame → Mx4 boxes, scores | `clip_box`; CVT `detect` on RetinaFace; IPT `imresize` | none | `+fv/+face/detectFaces.m` |
| `src/face/roi_polygons.m` | ROI quads from anchor box + KLT transform | `R = roi_polygons(anchorBox,A,p)` | box, 3x3 A → 4x2 polygon per region + box | `apply_tform` | none | `+fv/+roi/canonicalPatches.m` |
| `src/face/roi_skin_stats.m` | skin-pixel RGB sum/count inside one polygon | `[rgbSum,nSkin,nPoly,M,rect] = roi_skin_stats(I,poly,p)` | frame, polygon → sums, counts, mask | `skin_mask`; IPT `poly2mask` | none | `+fv/+pix/SkinModel.m` + `+fv/+trace/extractTraces.m` |
| `src/face/skin_mask.m` | YCbCr box + yMin + saturation skin mask | `M = skin_mask(P,p)` | RGB patch → logical mask | IPT `rgb2ycbcr` | none | `+fv/+pix/SkinModel.m` |
| `src/face/apply_tform.m` | apply 3x3 2-D transform to points | `Q = apply_tform(A,P)` | A, Kx2 → Kx2 | none | none | `+fv/+face/private/` |
| `src/face/box_iou.m` | IoU of one box against many | `iou = box_iou(a,B)` | boxes → Mx1 | none | none | `+fv/+face/private/` |
| `src/face/clip_box.m` | clip [x y w h] boxes to the frame | `c = clip_box(b,frameSize,asInteger)` | boxes → boxes | none | none | `+fv/+face/private/` |
| `src/dataio/load_ubfc_gt.m` | parse UBFC DATASET_2 `ground_truth.txt` | `gt = load_ubfc_gt(gtPath)` | path → `gt.ppg, gt.hr, gt.t` | none | none | `+fv/+io/loadUBFC2.m` (merge with `loadUBFC`, C2) |
| `src/dataio/read_frame_times.m` | decoded frame timestamps from `VideoReader.CurrentTime` | `t = read_frame_times(videoPath,maxFrames)` | path → Nx1 s | core `VideoReader` | none | `+fv/+io/readVideo.m` |
| `src/dataio/read_video_header.m` | header fields of a video | `hdr = read_video_header(videoPath)` | path → struct | core `VideoReader` | none | `+fv/+io/readVideo.m` |
| `src/dataio/export_traces.m` | save `process_video` output as a scipy-readable `.mat` | `outPath = export_traces(out,outDir)` | struct → `<ds>_<subj>_<cond>_<constr>.mat` | `trace_name, to_char_struct` | none | `+fv/+io/` helper (trace cache) |
| `src/dataio/to_char_struct.m` | recursively convert string → char/cellstr | `x = to_char_struct(x)` | any → any | none | none | `+fv/+io/private/` |
| `src/dataio/trace_name.m` | trace file name convention | `name = trace_name(ds,subj,cond,constr)` | strings → name | none | none | `+fv/+io/private/` |
| `src/degrade/clip_conditions.m` | table of test conditions (Baseline, VGA/QVGA, MultiFace, Masked, Stress) | `C = clip_conditions()` | → struct array | none | none | `+fv/+degrade/` (condition registry) |
| `src/degrade/make_test_clips.m` | write C3/C4/C7 test clips + sidecar `_meta.mat` from UBFC2 | `make_test_clips(subjects, Name=Value)` | raw UBFC2 videos → `data/degraded/<cond>/subjectN.avi` + meta | `face_params, clip_conditions, scan_target_boxes, detect_faces, draw_mask, clip_box, box_iou, to_char_struct`; RetinaFace; IPT `imresize, imgaussfilt` | none | `+fv/+degrade/multiFace.m`, `lowRes.m`, `mask.m` |
| `src/degrade/draw_mask.m` | paint a synthetic surgical mask on the lower face | `J = draw_mask(I,box,p)` | frame, box → frame | IPT `poly2mask, imgaussfilt` | none | `+fv/+degrade/mask.m` |
| `src/degrade/scan_target_boxes.m` | reference face box per frame (largest RetinaFace, every N frames, interpolated, cached) | `boxes = scan_target_boxes(videoPath,fd,p,Name=Value)` | video → Nx4 | `detect_faces`; RetinaFace | none | `+fv/+degrade/` (ground-truth boxes for C3) |
| `src/eval/run_benchmark.m` | process every subject x condition clip, tabulate face metrics + Bland-Altman | `T = run_benchmark(subjects, Name=Value)` | clips + GT → `face_benchmark*.csv`, BA csv/png | `process_video, export_traces, trace_name, load_ubfc_gt, benchmark_metrics, bland_altman, face_params`; RetinaFace | none (is itself a benchmark) | `run_experiment.m` (face benchmark mode) |
| `src/eval/benchmark_metrics.m` | per-clip metrics: target found, IoU, ID switches, jitter, occlusion accuracy, ms/frame, sanity HR | `[m,w] = benchmark_metrics(S,clip,gt)` | trace + sidecar + GT → table row, window pairs | `clip_conditions, box_iou, ppg_hr, sanity_hr` | none | `+fv/+metrics/` (face-module metrics; no named slot) |
| `src/eval/bland_altman.m` | pooled Bland-Altman per condition with CIs + figure | `R = bland_altman(W, Name=Value)` | window pairs → table, PNG | core graphics (`theme` → R2025a) | none | `+fv/+metrics/blandAltmanRM.m` |
| `src/eval/sanity_hr.m` | baseline HR: resample 30 Hz → POS (local `pos_rppg`) → butter(3) → `spectral_hr` | `[hr,tc] = sanity_hr(t,rgb,Name=Value)` | t, Nx3 RGB → window HR | `spectral_hr`; SPT `butter, filtfilt` | none | contrib only / drop (duplicate of `rgbToPulse` POS, C1) |
| `src/eval/spectral_hr.m` | sliding-window periodogram peak HR | `[hr,tc] = spectral_hr(x,tu,fs,Name=Value)` | uniform signal → window HR | SPT `periodogram, hann` | none | contrib only / drop (duplicate of `estimateHR`, C10) |
| `src/eval/ppg_hr.m` | reference HR from contact PPG, same windows as `sanity_hr` | `[hr,tc] = ppg_hr(tg,ppg,Name=Value)` | GT t, PPG → window HR | `spectral_hr`; SPT `butter, filtfilt` | none | contrib only / drop (duplicate of `referenceHR`, C9) |
| `src/tools/debug_video.m` | overlay video/GIF of a trace file (faces, ROIs, skin, status); refuses subjects 21/27 | `outPath = debug_video(tracePath, Name=Value)` | trace .mat + video → AVI/GIF | `roi_skin_stats`; CVT `insertShape, insertText` | none | `tools/overlay_patches.m` |
| `src/tools/video_info.m` | print header vs decoded timestamps, GT length check | `video_info(videoPath, Name=Value)` | video (+GT) → console | `read_video_header, read_frame_times, load_ubfc_gt` | is a check script | `tools/` |
| `src/tools/repo_root.m` | repo root path | `root = repo_root()` | → string | none | none | `+fv/private/` |
| `src/run_pipeline.m` | face-module benchmark driver: clips → traces → benchmark → debug videos | `T = run_pipeline(Name=Value)` (Subjects, Config, Quick, MakeClips, Rerun, DebugVideos) | → tables, figures | `face_params, make_test_clips, run_benchmark, clip_conditions, trace_name, debug_video, repo_root`; RetinaFace | `Quick=true` smoke run | contrib only; logic → `run_experiment.m`. **Name clashes with root `run_pipeline.m` (C3)** |

## 2. Thamil (`thamilezai_dev`) — 2 files

| File | Purpose | Entry point? | Inputs → outputs | Dependencies | Tests | Target (Section 9) |
|---|---|---|---|---|---|---|
| `src/signal/kalman_denoise.m` | causal per-channel random-walk Kalman on frame-mean RGB; `R_k = R0·median(npix)/npix_k`; Q set for −3 dB at `Cutoff` | `y = kalman_denoise(x, fs, Npix=, Cutoff=4)` | NxC traces (NaN allowed), fs → NxC | none (core only) | TI (`denoise='kalman'` variant must give MAE < 1 bpm); no direct unit test of the −3 dB point | `+fv/+trace/kalmanDenoise.m` |
| `src/eval/aipex_hr.m` | AiPEX: face_detect skin mask → pooled `rgb_face` → `sanity_hr` with/without `kalman_denoise`; ITA strata; CSVs | `[T,S] = aipex_hr(Name=Value)` (Root, OutDir, TraceDir, Win, Rerun, MaxVideos) | `Data/AiPEX-Lab` clips + GT CSV → per-video and summary CSVs | `face_params, skin_mask, sanity_hr, kalman_denoise, repo_root`; IPT `rgb2lab`; Stats `corr` | is a check script (no asserts) | `tools/diagnose_aipex.m` (+ contrib) |

## 3. Muftee (`muftee_dev`) — 31 files

| File | Purpose | Entry point? | Inputs → outputs | Dependencies | Tests | Target (Section 9) |
|---|---|---|---|---|---|---|
| `config/getConfig.m` | all signal-chain parameters and paths; name/value overrides (edited in 0792a67) | `cfg = getConfig(name,value,...)` | overrides → cfg | none | TS, TI | `config/default.json` + `+fv` loader |
| `run_pipeline.m` (root) | Round 1 baseline on one UBFC subject: load → `extractRGBTraces` → reference → methods x ROIs → CSV/MAT/figures | `out = run_pipeline(varargin)` (getConfig overrides) | subject → `outputs/<subj>/...` | `getConfig, loadUBFC, extractRGBTraces, referenceHR, rgbToPulse, preprocessPulse, estimateHR, computeMetrics, makeFigures` | none | seed for `process_round1.m`. **Name clashes with `src/run_pipeline.m` (C3)** |
| `run_ubfc_all.m` | root `run_pipeline` on every UBFC subject; pooled metrics | `out = run_ubfc_all(varargin)` | → `outputs/UBFC_summary/*.csv` | `getConfig, listUBFCSubjects, run_pipeline (root), computeMetrics` | none | `run_experiment.m` |
| `run_ubfc1.m` | UBFC DATASET_1, including pairing loose videos with GT folders by HR agreement | `out = run_ubfc1(varargin)` | → `outputs/UBFC1*/` | `getConfig, run_pipeline (root), extractRGBTraces, rgbToPulse, preprocessPulse, estimateHR, loadUBFC, referenceHR, computeMetrics` | none | `run_experiment.m` (+ `+fv/+io/loadUBFC1.m` pairing logic) |
| `run_aipex.m` | AiPEX via crop traces; video-level HR (`videoHR`); paper baseline from `Results_Final_Main.csv` | `out = run_aipex(varargin)` | → `outputs/AiPEX/*` | `getConfig, loadAiPEX, extractCropTraces, rgbToPulse, preprocessPulse, estimateHR, videoHR, computeMetrics`; PCT (`parpool`/`parfor` when `useParallel`) | none | `run_experiment.m` / `tools/diagnose_aipex.m` |
| `src/io/loadUBFC.m` | find UBFC subject (DS1 `gtdump.xmp` or DS2 `ground_truth.txt`), parse GT | `[videoPath,gt] = loadUBFC(cfg)` | cfg.dataRoot/subject → path, `gt.ppg, gt.hrSensor, gt.t` | core `readlines, readmatrix` | TI uses a GT struct shaped like its output (not the loader) | `+fv/+io/loadUBFC1.m` + `loadUBFC2.m` |
| `src/io/loadAiPEX.m` | index AiPEX videos + GT HR into a table | `V = loadAiPEX(cfg)` | cfg.aipexRoot → table | core `readtable` | none | `+fv/+io/loadAiPEX.m` |
| `src/io/listUBFCSubjects.m` | subjects with `vid.avi` under dataRoot, sorted | `subjects = listUBFCSubjects(cfg)` | cfg → cellstr | none | none | `+fv/+io/` helper |
| `src/face/extractRGBTraces.m` | Viola-Jones + KLT face box, rectangular ROIs, skin-masked mean RGB, `combined` ROI; cached | `traces = extractRGBTraces(videoPath,cfg)` | video → `traces.fps, t, rgb.(roi), box, faceOK, info` | `degradeFrame, roiBoxes, roiSumRGB`; CVT (`vision.CascadeObjectDetector, vision.PointTracker, detectMinEigenFeatures, estgeotform2d`) | none | contrib only (front end superseded by `process_video`; C6) |
| `src/face/extractCropTraces.m` | mean RGB of pre-cropped ROI videos (AiPEX), same output shape | `traces = extractCropTraces(roiPaths,cfg,cacheFile)` | ROI video paths → traces | `degradeFrame, roiSumRGB` | none | `+fv/+trace/extractTraces.m` (crop mode) |
| `src/face/roiBoxes.m` | ROI rectangles from face box (`[x1 x2 y1 y2]` fractions) | `boxes = roiBoxes(faceBox,cfg,imSize)` | box → `[r1 r2 c1 c2]` per ROI | none | PJ (JS port) | contrib only (web parity) / superseded by `+fv/+roi` |
| `src/face/roiSumRGB.m` | skin-masked RGB sum in a rectangle; falls back to all pixels below `minSkinPixels` | `[s,n] = roiSumRGB(frame,box,cfg)` | frame, rect → 1x3 sum, count | IPT `rgb2ycbcr` | PJ (JS port) | contrib only / superseded by `+fv/+pix/SkinModel.m` |
| `src/signal/rgbToPulse.m` | GREEN / CHROM / POS pulse extraction | `p = rgbToPulse(rgb,fs,method,cfg)` | Nx3 → Nx1 | `bandpassFilter` (CHROM) | TS (all 3 methods ±1 bpm), TI, PJ | `+fv/+rppg/pos.m`, `chrom.m` (GREEN dropped) |
| `src/signal/preprocessPulse.m` | detrend → bandpass → z-score | `pf = preprocessPulse(p,fs,cfg)` | Nx1 → Nx1 | `detrendSmoothnessPriors, bandpassFilter` | TS, TI, PJ | `+fv/+rppg/preprocess.m` |
| `src/signal/bandpassFilter.m` | zero-phase Butterworth (SOS), order `cfg.filterOrder` | `y = bandpassFilter(x,fs,cfg)` | NxM → NxM | SPT `butter, zp2sos, filtfilt` | TS, TI, PJ | `+fv/+rppg/preprocess.m` |
| `src/signal/detrendSmoothnessPriors.m` | Tarvainen smoothness-priors detrend (sparse) | `y = detrendSmoothnessPriors(x,lambda)` | Nx1 → Nx1 | none | TS, TI, PJ | `+fv/+rppg/preprocess.m` |
| `src/signal/estimateHR.m` | sliding-window FFT peak HR (Hann, nfft ≥ 2^14, parabolic interpolation); returns spectra | `[tHR,hr,spec] = estimateHR(signal,fs,cfg)` | Nx1 → window times, HR, spectra | SPT `hann` | TS, TI, PJ | `+fv/+fuse/spectralFusion.m` / `+fv/+track/` (single-patch baseline) |
| `src/signal/referenceHR.m` | contact PPG → video time base → same chain → reference HR per window; mean oximeter HR per window | `[tRef,hrRef,ref] = referenceHR(gt,traces,cfg)` | GT (`.t,.ppg,.hrSensor`), traces → reference | `preprocessPulse, estimateHR` | TI | `+fv/+metrics/` (reference HR; no named slot) |
| `src/signal/videoHR.m` | one HR per video: peak of the averaged normalised window spectra, plus median | `[hrAvgSpec,hrMedian] = videoHR(spec,hrWin)` | spectra → 2 scalars | none | none | `+fv/+fuse/spectralFusion.m` (video-level) |
| `src/degrade/degradeFrame.m` | per-frame constraint hook: resize, low-light gain + noise; no-op by default | `frame = degradeFrame(frame,dcfg)` | frame → frame | IPT `imresize, imnoise` | none | `+fv/+degrade/lowLight.m`, `lowRes.m` |
| `src/eval/computeMetrics.m` | MAE, RMSE, MAPE, r, BA, SUC3/5, coverage; skips NaN pairs (edited in 0792a67) | `m = computeMetrics(hrEst,hrRef)` | paired vectors → struct | none | TI (coverage, nWin asserted) | `+fv/+metrics/evaluate.m` |
| `src/tools/makeFigures.m` | 6 diagnostic PNGs per subject; no ROI overlay for private subjects | `files = makeFigures(traces,R,ref,tRef,hrRef,best,cfg)` | results → PNGs | `degradeFrame, roiBoxes`; CVT `insertShape`; Stats `corr` | none | `tools/plot_subject.m` |
| `tests/test_synthetic.m` | synthetic 72 bpm RGB with drift and noise; each method within ±1 bpm in every window | `test_synthetic()` | → pass / `error` | `getConfig, rgbToPulse, preprocessPulse, estimateHR` | – | `tests/` |
| `deploy/web/README.md` | browser demo docs, parity numbers | – | – | – | – | `web/` docs |
| `deploy/web/index.html` | demo page shell | – | – | `app.js, style.css` | – | seed for `web/` (Section 14.1) |
| `deploy/web/style.css` | demo styles | – | – | – | – | seed for `web/` |
| `deploy/web/app.js` | webcam, MediaPipe Face Landmarker box, ROI sums, live HR UI, reference logging, CSV export | browser module | camera → UI, CSV | `rppg.js`; MediaPipe tasks-vision 0.10.14 from jsDelivr + face model from Google storage (needs internet) | none | seed for `web/` |
| `deploy/web/rppg.js` | JS port of `rgbToPulse`, `preprocessPulse`, `estimateHR`, `roiBoxes`, `roiSumRGB` + `LiveEstimator`; Butterworth/filtfilt matching MATLAB | ES module exports | traces → HR | none | PJ | seed for `web/` (TypeScript) |
| `deploy/web/tests/parity.mjs` | Node parity test: JS vs MATLAB per-window HR (tolerance 0.5 bpm) + live replay | `node deploy/web/tests/parity.mjs [dir]` | `outputs/web_parity/*.json` → exit code | `rppg.js` | – | `web/` tests + `shared/golden/` |
| `deploy/web/tools/exportParityData.m` | export cached `traces.mat` + MATLAB HR to JSON for PJ | `exportParityData(subjects)` | `outputs/<s>/traces.mat` → JSON | `getConfig, rgbToPulse, preprocessPulse, estimateHR` | – | `tools/` / `shared/golden/` generator |
| `deploy/web/tools/analyzeWebSession.m` | run the MATLAB chain on a browser-recorded session; score logged references | `out = analyzeWebSession(tracesCsv,refsCsv,fs)` | CSVs → table | `getConfig, rgbToPulse, preprocessPulse, estimateHR` | none | `tools/` |

## 4. Integration (0792a67, Muftee) — 6 files

| File | Purpose | Entry point? | Inputs → outputs | Dependencies | Tests | Target (Section 9) |
|---|---|---|---|---|---|---|
| `run_ubfc2.m` | integrated pipeline on all UBFC2 subjects; per-subject + summary (mean and pooled) CSVs, face stats, log | `out = run_ubfc2(varargin)` | getConfig overrides → `outputs/UBFC2_integrated/*` | `getConfig, listUBFCSubjects, face_params, loadUBFC, faceTraces, subjectHR, computeMetrics`; RetinaFace on cache miss | none | `run_experiment.m` (UBFC2 config) |
| `src/pipeline/faceTraces.m` | `process_video` with on-disk cache keyed on face params + maxFrames; clear error if RetinaFace is missing | `[fv,fd] = faceTraces(videoPath,subject,cfg,p,fd)` | video → fv struct | `process_video`; RetinaFace | none | `+fv/+pipeline/runOffline.m` (front-end stage) |
| `src/pipeline/tracesForSignal.m` | mask lost/occluded frames → optional Kalman at native fs → resample to `cfg.resampleFs`; validity per ROI | `traces = tracesForSignal(fv,cfg,denoise)` | fv → uniform traces + valid | `kalman_denoise` | TI | `+fv/+trace/resampleUniform.m` |
| `src/pipeline/signalHR.m` | Muftee's chain per method x ROI; windows with valid fraction < `minValidFrac` → NaN | `R = signalHR(traces,hrRef,cfg)` | traces, ref HR → results | `rgbToPulse, preprocessPulse, estimateHR, computeMetrics` | TI | `+fv/+pipeline/runOffline.m` |
| `src/pipeline/subjectHR.m` | one subject: each denoise variant → `signalHR` against `referenceHR` | `[T,R,ref] = subjectHR(fv,gt,cfg)` | fv, GT → table, results | `tracesForSignal, referenceHR, signalHR` | TI | `+fv/+pipeline/runOffline.m` |
| `tests/test_integration.m` | synthetic `process_video`-shaped struct (72 bpm, jittered times, 4 s gap, occluded rcheek) through `subjectHR` | `test_integration()` | → pass / assert error | `getConfig, subjectHR` (and everything below it) | – | `tests/` |

## 5. Shared / docs (not code)

| File | Note |
|---|---|
| `README.md` | Since 46790ae (Muftee, 2026-11-26): the build spec. Earlier history: 8af1bc3 (Thamil, initial commit), then Sulojan's face README, now at `docs/FACE_MODULE.md`. Target: root README. |

Total: 36 + 2 + 31 + 6 = 75 tracked files (excluding `.gitkeep`), counting `README.md` here and not in Sulojan's 36. Sulojan's count includes `docs/FACE_MODULE.md`.

## 6. Conflicts and duplicates

| # | Issue | Evidence | Note / suggested resolution |
|---|---|---|---|
| C1 | **Two POS implementations** (three with JS) | `src/signal/rgbToPulse.m:28-41` vs `src/eval/sanity_hr.m:30-43` (`pos_rppg`); also `deploy/web/rppg.js:184-206` | The algorithm is identical: same projection `[0 1 -1; -2 1 1]`, overlap-add, `round(1.6*fs)`. `sanity_hr` hardcodes 1.6 s and fs = 30 (`:16`, `:32`); `rgbToPulse` uses `cfg.posWindowSec`. Keep one (`+fv/+rppg/pos.m`). |
| C2 | **Two UBFC GT loaders with different field names** | `src/dataio/load_ubfc_gt.m:24-26` (`gt.hr`) vs `src/io/loadUBFC.m:49-52` (`gt.hrSensor`, plus `gt.path`) | `referenceHR.m:20` reads `gt.hrSensor`, so a `load_ubfc_gt` struct breaks it. `load_ubfc_gt` requires exactly 3 non-empty lines (`:13`); `loadUBFC` accepts ≥ 3 (`:35`) and also DATASET_1 `.xmp` (`:55-66`). Merge into `+fv/+io/loadUBFC1/2.m`. |
| C3 | **Two `run_pipeline.m`** with incompatible signatures | root `run_pipeline.m:1` `run_pipeline(varargin)` (getConfig pairs) vs `src/run_pipeline.m:1` `run_pipeline(opts)` (`arguments` Name=Value: Subjects, Config, Quick, ...) | MATLAB runs the **current folder** before the path, and `addpath` *prepends*. (a) `docs/FACE_MODULE.md:53-62` says "from the repo root: `addpath(genpath("src"))`; `run_pipeline(Quick=true)`". That resolves to the root file (Muftee's), which passes `'Quick'` to `getConfig` and fails with `getConfig:unknownField` (`getConfig.m:90-91`). (b) `run_ubfc_all.m:10` and `run_ubfc1.m:19` do `addpath(genpath(src))` and then call `run_pipeline(...)` (`run_ubfc_all.m:18`, `run_ubfc1.m:53`). They only work when the cwd is the repo root. From any other cwd, `src/run_pipeline.m` is first on the path and rejects the positional name/value args. Rename one (e.g. root → `process_round1`, src → `run_face_benchmark`). |
| C4 | **Two configs** | `config/getConfig.m` (signal, Viola-Jones front end, paths) vs `config/face_config.json` + `src/face/face_params.m` (face module) | The two overlap on ROI, skin and detection period with different values (C5, C7). `getConfig.m:65` points at the JSON, so both are live in `run_ubfc2`. Section 9 wants one `config/default.json`. |
| C5 | **Different skin YCbCr ranges** | `getConfig.m:45-46` Cb [77 127], Cr [133 173] (also `rppg.js:389`) vs `face_config.json:28-29` / `face_params.m:55-56` Cb [77 135], Cr [128 173], plus `yMin` 20 and `satMax` 250 | The face side was widened for dark skin (808b938; `docs/FACE_MODULE.md:125`). Low-skin handling also differs: `roiSumRGB.m:21` falls back to *all* ROI pixels below `minSkinPixels` = 50, while `process_video.m:144` marks the region occluded below `occl.minPix` = 10. |
| C6 | **Two face front ends** | `extractRGBTraces.m:61` Viola-Jones (`vision.CascadeObjectDetector`), detect every 30 frames (`getConfig.m:33`) vs `process_video` RetinaFace, `det.every` = 10 (`face_config.json:8`) | `run_pipeline`/`run_ubfc_all`/`run_ubfc1` (Muftee) still use Viola-Jones; `run_ubfc2` uses RetinaFace. Section 2 UBFC2 numbers (POS/combined 2.17) come from the Viola-Jones path. |
| C7 | **Different ROI conventions and geometry** | `getConfig.m:38-41` `[x1 x2 y1 y2]`, forehead `[0.30 0.70 0.08 0.25]`, names `leftCheek/rightCheek` vs `face_config.json:18-20` `[u1 v1 u2 v2]`, forehead `[0.32 0.12 0.68 0.27]` (x 0.32-0.68, y 0.12-0.27), names `lcheek/rcheek`; AiPEX uses `leftcheek/rightcheek` (`aipex_hr.m:36`, `getConfig.m:29`) | Same cheek numbers, different field order. The forehead differs. Three naming schemes. |
| C8 | **`src/io` vs `src/dataio`** | `src/io/{loadUBFC,loadAiPEX,listUBFCSubjects}.m` (Muftee) vs `src/dataio/{load_ubfc_gt,read_frame_times,read_video_header,export_traces,...}.m` (Sulojan) | Both do dataset I/O. Merge into `+fv/+io`. |
| C9 | **Two reference-HR methods** | `src/eval/ppg_hr.m:14-20` (resample PPG to 30 Hz, butter(3), periodogram) vs `src/signal/referenceHR.m:22-31` (interp onto video t, detrend + butter(4) + z-score, `estimateHR`) | Face-module MAE (1.28) and signal-chain MAE (2.17) are measured against different references and windows. They are not directly comparable. |
| C10 | **Two spectral HR estimators** | `src/eval/spectral_hr.m:22-30` (periodogram, nfft ≥ 8192, windows selected by time, no interpolation) vs `src/signal/estimateHR.m:21-47` (FFT, nfft ≥ 2^14, sample-index windows, parabolic interpolation) | Keep one. |
| C11 | **Band-pass order** | `sanity_hr.m:25`, `ppg_hr.m:18` `butter(3, …)` (transfer-function form) vs `getConfig.m:51` `filterOrder = 4` via `bandpassFilter.m:7-9` (SOS) | Spec Section 4.5 says order 4. |
| C12 | **Kalman cutoff clamp differs from spec** | `kalman_denoise.m:27` `min(opts.Cutoff, 0.45*fs)` vs spec Section 4.4 (`C:\SP Cup 2027\README.md:210`) `min(4 Hz, 0.8·fs/2)` = `min(4, 0.4·fs)` | Same result for fs ≥ 10 Hz (4 Hz). Below 10 Hz the code allows a higher cutoff: fs = 8 gives 3.6 vs 3.2 Hz, fs = 5 gives 2.25 vs 2.0 Hz. Also, the −3 dB point is exact only at median npix (`:13-15`). The fs passed in differs by caller: nominal `R.forehead.fps` (`aipex_hr.m:77`) vs measured `1/median(diff(t))` (`tracesForSignal.m:20,52`). |
| C13 | **Frame time base** | `extractRGBTraces.m:125` and `extractCropTraces.m:42` `t = (0:N-1)/fps` vs `process_video.m:77`, `read_frame_times.m:17`, `aipex_hr.m:123` `v.CurrentTime` | Sulojan's `video_info.m:23-26` exists to check exactly this mismatch. Use real timestamps. |
| C14 | **Four metric implementations** | `computeMetrics.m` vs `benchmark_metrics.m:121-132` (`hr_err`) vs `bland_altman.m:29-56` vs `aipex_hr.m:148-159` (`summarize`) | SUC is `<` everywhere except `run_aipex.m:118` (`<= 5`). Consolidate into `+fv/+metrics/evaluate.m`. |
| C15 | **Two AiPEX pipelines** | Thamil `aipex_hr.m` (`skin_mask` with face skin box, `rgb_face` pixel-weighted, `sanity_hr`, video HR = mean of window HRs `:84`) vs Muftee `run_aipex.m` + `loadAiPEX` + `extractCropTraces` (getConfig skin box, `estimateHR`, video HR = `videoHR` averaged spectrum) | Different data roots too: `aipex_hr.m:27` `<repo>/Data/AiPEX-Lab` vs `getConfig.m:26` `D:\SPCup Data\dataset\AiPEX rppg_biases\rppg_biases-main`. The two report different AiPEX numbers. |
| C16 | **Dataset roots** | `getConfig.m:15` hardcoded `D:\SPCup Data\dataset` vs `make_test_clips.m:16` `data/raw/UBFC/DATASET_2` (repo-relative) | Pick one, configurable. |
| C17 | `.gitignore` case | `.gitignore:2` ignores `data/**`; `aipex_hr.m:27` uses `Data/AiPEX-Lab` | On Linux (case-sensitive) AiPEX data under `Data/` is **not** ignored. |
| C18 | **Two degradation paths** | `degradeFrame.m` (in-loop resize, gain + noise) vs `make_test_clips.m` + `clip_conditions.m` (offline clips: QVGA, MultiFace, Masked) | Complementary rather than identical; both feed `+fv/+degrade`. `degradeFrame.m:25` `intmax(cls)` errors for non-integer frames (only uint8 is used now). |
| C19 | Privacy list duplicated | `getConfig.m:23` `privateSubjects = [21 27]` (used by `makeFigures.m:128-131`) vs `debug_video.m:24` hardcoded | One list in config. |
| C20 | Three trace-cache layouts | `outputs/<subject>/traces.mat` (`extractRGBTraces.m:16-17`) vs `outputs/traces/<ds>_<subj>_<cond>_<constr>.mat` (`export_traces.m:14`) vs `outputs/traces/ubfc2_integrated/<subject>.mat` (`faceTraces.m:18`, `getConfig.m:66`) | Unify (TraceFrame / `shared/`). |
| C21 | JS config mirror | `rppg.js:10-21` (`DEFAULT_CFG`), `:384-390` (`ROI_CFG`): "Keep the two in sync" | Manual copy of `getConfig` values. Section 9 `shared/` should be the single source. |

## 7. Tests that exist

| Script | Kind | What it asserts | Needs data? |
|---|---|---|---|
| `tests/test_synthetic.m` | plain function (not `matlab.unittest`); call `test_synthetic()` | 60 s synthetic RGB at 30 fps, 72 bpm pulse, illumination drift, noise σ = 0.05. For GREEN, CHROM and POS: `max |hr − 72| ≤ 1` bpm in **every** 10 s window; otherwise `error('test_synthetic:fail')` (`:28-44`). | no |
| `tests/test_integration.m` | plain function; call `test_integration()` | Synthetic `process_video`-shaped struct through `subjectHR`, for both `none` and `kalman`. POS/face MAE < 1 bpm (`:54`). Coverage between 60 % and 100 %, so gap windows are dropped (`:55-56`). Occluded rcheek has `nWin == 0` (`:58`). Kalman POS/face windows within 5 s of t = 32 s are NaN (`:61-62`). | no |
| `deploy/web/tests/parity.mjs` | Node script, exit code | JS `processTrace` gives the same number of windows as MATLAB (throws otherwise, `:38`). Worst per-window `|HR_JS − HR_MATLAB| ≤ 0.5` bpm, else exit 1 (`:85-87`). Also prints pf correlation and MAE (MATLAB / JS offline / JS live) without asserting them. | yes: `outputs/web_parity/*.json` from `exportParityData.m`, which needs cached UBFC `traces.mat` |
| `src/tools/video_info.m` | check script (prints, no asserts) | Prints decoded frame count vs header (`OK`/`MISMATCH`), timestamp vs `(k-1)/fps` difference, GT length vs frames. | yes (video) |
| `src/run_pipeline.m` (`Quick=true`) + `src/eval/run_benchmark.m` | benchmark / smoke run (no asserts) | Face metrics tables and Bland-Altman; `run_benchmark` only warns on missing clips. | yes (UBFC2 + RetinaFace) |
| `src/eval/aipex_hr.m` | evaluation script (no asserts) | Baseline vs Kalman MAE/RMSE/SUC per country and ITA class. | yes (AiPEX) |
| `src/face/face_params.m` `validate` | input validation | errors if a used region lacks a 1x4 box (`:107-108`); warns if `occl.reference` is not a region. | no |

There are no unit tests for the face module (tracker, ROI, skin, occlusion), the loaders, `kalman_denoise` alone (e.g. its −3 dB point), `degradeFrame` or `videoHR`. Nothing in the repo runs the tests automatically (no `runtests` suite or CI).

## 8. Not present

- **Pi:** `deploy/pi5/` contains only `.gitkeep`. There is no Raspberry Pi code in any branch (`face_detect`, `thamilezai_dev`, `muftee_dev`, `integrated_pipeline`, `main`). Section 15 step 5 assumes "existing Pi code"; it is not in this repo.
- **Android / mobile:** none.
- **Python front end, C++ `fvcore`, `shared/`, `contrib/`, `process_round1.m`, `src/+fv/` packages:** none yet.
- **Submission writer** (`RoundX_Condition_Constraint.mat` with the six required variables): none. The root `run_pipeline.m` writes `processed_output.csv` and `hr_results.mat` with different contents.
