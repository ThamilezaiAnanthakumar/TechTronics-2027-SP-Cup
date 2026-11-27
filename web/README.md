# FaceVitals web app (`web/`)

Browser demo of camera heart rate (rPPG) for IEEE SP Cup 2027 (spec: root
`README.md`, Sections 4.2–4.9, 11, 14.1). Everything runs on the device; no
video leaves the browser and no network requests are made after loading.

## Run

```
cd web
npm ci
npm run fetch-models   # once: copies the MediaPipe WASM and downloads face_landmarker.task into public/
npm run dev            # http://localhost:5173
npm test               # Vitest, synthetic data only
npx tsc --noEmit -p .  # type check
npm run build          # dist/ + precaching service worker (offline PWA)
npm run e2e            # Playwright smoke test against the running dev server (system Chrome, fake camera, synthetic face clip)
```

Camera access needs `localhost` or HTTPS.

## Structure

| path | what |
|---|---|
| `src/pipeline.ts` | main thread: camera / file capture (`requestVideoFrameCallback` timestamps), camera AE/AWB/focus lock, MediaPipe FaceLandmarker (4 faces), target selection with the appearance gate, 1-Euro landmark smoothing |
| `src/front/worker.ts` | Web Worker: pixel selection (`FrontEnd`) and the signal core; `makeCore()` picks the core (CP3 default, interim selectable) and is the one line fvcore-WASM will replace |
| `src/core/cp3Core.ts` | **default core: CP3**, the frozen Round-1 chain (port of `src/pipeline/snrFusionHR.m`, see below) |
| `src/core/viterbiTrack.ts` | CP3 output tracking (default): port of `src/signal/viterbiHR.m` + online fixed-lag Viterbi, \|ΔHR\| ≤ 4.99 bpm per 1 s step |
| `src/core/troikaTrack.ts` | CP3 output tracking (option): TROIKA spectral peak tracking, port of `src/track/troikaTrack.m` |
| `src/front/frontEnd.ts` | README 4.3 pixel selection → TraceFrames (11.1): adaptive (Cb, Cr) skin model, specular and texture masks, trimmed mean, occlusion EMA/hysteresis, 2×2 sub-patches |
| `src/front/appearance.ts` | README 4.1 colour-histogram appearance gate (Bhattacharyya distance) |
| `src/core/interimCore.ts` | interim TS signal core (README 4.4–4.9): Kalman, resampling, POS/CHROM, detrend, BPF, SNR/coherence/coverage weights, top-k spectral fusion, harmonic sum, motion suppression, fixed-lag Viterbi |
| `src/core/motion.ts` | README 4.8.2 spectral motion suppression |
| `src/config.ts` | imports `shared/patches.json` and `shared/config.default.json` (single sources of truth) |
| `src/localConfig.ts` | **web-local tunables pending move to `shared/`** (operator approval; see `NOTES_FOR_OPERATOR.md`) |
| `src/ui/` | React UI: video + patch overlay, HR / confidence / status, BVP, spectrum with Viterbi track, CSV and `.tracejsonl` export |
| `tests/` | Vitest; synthetic signals and rendered synthetic faces only |

## Signal cores

The UI select **Core: CP3 (frozen) / interim** chooses the core before a run; the
active core is shown in the HR card and the session-log note, and written to the
`.tracejsonl` header (`core: cp3-ts | interim-ts`, `core_desc`).

**CP3 (default, `src/core/cp3Core.ts`)** — the frozen Round-1 configuration
(`config/round1_freeze.json` in the main repo), ported line by line from
`src/pipeline/snrFusionHR.m` + `src/signal/kalman_denoise.m`. Per patch (10
anatomical patches, no sub-patches): Kalman denoise on the native frame times
(R_k ∝ 1/npix, −3 dB at min(4 Hz, 0.4·fs)) → linear resample to 30 Hz (edges:
nearest) → POS 1.6 s (std N−1) → Butterworth band-pass **order 3**, 0.7–3 Hz,
zero phase (no detrend, no z-score) → 10 s windows / 1 s step (window needs
≥ 90 % of its samples, a patch ≥ 90 % valid samples in the window and ≥ 50 %
overall) → symmetric Hann periodogram, nfft 8192, unit in-band power → linear
SNR (±0.1 Hz around the peak, ±0.2 Hz around its harmonic, over the rest of the
band) → fused = Σ SNR_i·P_i/ΣP_i(band) → HR = 60·argmax (no interpolation).
The filter is MATLAB `butter(3, band/(fs/2), 'bandpass')` in b/a form (coefficients
within 1e-12 of MATLAB) with `filtfilt`'s own edge handling (odd reflection of
3·(nfilt−1) = 18 samples, `zi`-scaled initial states), so there is no edge-padding
difference. CP3 parameters live in `CP3` in `cp3Core.ts`, not in
`shared/config.default.json` (that file still holds the interim values: order 4,
detrend, nfft 16384, …).

*Online mode*: once per second the core runs the chain on the newest 15 s of
TraceFrames (10 s window + 5 s history so Kalman, POS and the filter settle) and
evaluates the one window [T−10 s, T). No harmonic summation, no motion
suppression. The right edge of the window is always a filter edge online, unlike
offline.
*Confidence* = share of the fused in-band power within ±0.1 Hz of the fused peak,
in [0, 1] (synthetic 72 bpm: 0.54–0.76). *Status* (same codes as interim, never
OK without an HR): WARMUP (< 10 s), NO_FACE (no face in the last 1 s),
LOW_COVERAGE (window or every patch below the 90 % rules), LOW_SNR (no qualifying
patch has linear SNR > 1, i.e. 0 dB, the interim weight floor), MASKED (interim
README 4.3.5 occlusion rule; HR still CP3's), OK.

*Parity with MATLAB* (`tests/cp3Parity.test.ts`; fixtures
`tests/fixtures/cp3_parity_subject{1,3,12}.json` written by
`tools/export_cp3_parity.m` in the main repo: UBFC **dev** subjects 1, 3, 12, first
60 s of the cached anatomical-patch traces, per-frame mean RGB only, rounded to
4 decimals, and `snrFusionHR` run on those same numbers): offline window HRs are
identical — subject 1: 43 windows, subject 3: 50, subject 12: 50; max |diff| 0 bpm,
100 % within 0.5 bpm. Online `Cp3Core` rows vs the MATLAB offline windows (same
centres; `tests/cp3Core.test.ts`): 100 % within 0.5 bpm (max 0.22 / 0.44 / 0.22
bpm); 1 / 6 / 1 windows are LOW_SNR online (no HR) where MATLAB reports one.
Synthetic 72 bpm (jittered timestamps): max error 0.51 bpm at 30 fps, 0.81 bpm at
10 fps.

*Output tracking* (UI select **Tracking: Viterbi ±5 bpm (default) / TROIKA / off**
next to the core select; `.tracejsonl` header `tracker`, `tracker_kind`). The
operator saw a resting ~55 bpm jump to 85 and 65 during motion; the trackers run on
the fused in-band spectrum of each window (the window argmax itself stays CP3) so
that consecutive readings never change by 5 bpm or more:

- **Viterbi (default, `viterbiTrack.ts`)** — port of `viterbiHR.m` with
  MaxDown = MaxUp = 4.99 bpm/s: grid 42–180 bpm in 0.5 bpm, emission
  log(normalised interp1 of the fused spectrum + 1e-12), transition allowed only
  if |h′ − h| ≤ 4.99 bpm × the actual step, cost 0.1/bpm, parabolic refinement.
  Online it is fixed-lag: the row of window T is released at T + 3 s (`finish()`
  flushes); a fixed-lag decision is additionally limited to ±4.99 bpm × step from
  the previous output (the backtracked paths of successive decisions need not
  coincide; 0 times in the motion-burst test). NO_FACE windows are gaps (flat
  emission); more than 10 s without a spectrum flushes and restarts the path.
  Offline parity with MATLAB `viterbiHR` on the synthetic scenarios below: max
  |diff| 1.4e-14 bpm. Main-session measurement (MATLAB, UBFC dev): pooled MAE
  1.38 bpm vs 1.39 argmax.
- **TROIKA (`troikaTrack.ts`)** — port of `troikaTrack.m` (Zhang, Pi & Liu,
  TBME 2015, III-D, physical units: init on the highest peak, ±15 bpm search
  (±19 when lost) plus the doubled range for the harmonic, harmonic pair →
  nearest candidate → hold, |est − prev| ≥ 5 bpm × step → prev ± 2 bpm × step,
  cubic-trend step after 3 unchanged windows, re-init after > 10 s without a
  spectrum); parameters in `TROIKA`. No lag. Parity with MATLAB: 0 bpm, actions
  identical in 380/380 windows. Main-session measurement (MATLAB): it hurts
  accuracy on real data (UBFC dev MAE 1.39 → 2.01, AiPEX dev 5.59 → 18.46) because
  it locks onto a wrong component after a bad start and cannot recover; the
  synthetic `hard` scenario shows the same (initialised on a motion peak at 114
  bpm for a true 70: off by more than 5 bpm for the first 40 windows). Kept as an option only.
- **off** — raw argmax (exact MATLAB `snrFusionHR` window HRs, no lag).

Synthetic test (`tests/troikaTrack.test.ts`, `tests/viterbiTrack.test.ts`, port of
`tests/test_troika_track.m` with the same input: MATLAB `rng(5)` + `rand` are
reproduced bit-exactly by `tests/mt19937.ts`): true HR 55 bpm, then a real rise to
70 at 0.5 bpm/s; every 7th window a motion peak at 85 bpm and every 11th at 65 bpm,
1.5× stronger than the pulse.

| 120 windows, 1 s step | max \|error\| | max \|step\| between readings |
|---|---|---|
| argmax (tracking off) | 30.07 bpm | 30.19 bpm |
| TROIKA | 2.42 bpm | 3.11 bpm |
| Viterbi, online, lag 3 s (default) | 1.25 bpm | 1.87 bpm |

The parity fixture `tests/fixtures/troika_parity.json` (MATLAB outputs only; the
spectra are regenerated in the test) comes from `tools/export_troika_parity.m` in
the main repo, run against `troikaTrack.m` as committed in 8f93c0f.

**Interim (`src/core/interimCore.ts`)** — the README 4.4–4.9 interim core
(POS/CHROM, detrend, order-4 BPF, SNR/coherence/coverage weights, top-k fusion,
harmonic summation, motion suppression, fixed-lag Viterbi, sub-patches), kept
selectable for comparison.

## Implemented (README 14.1 checklist)

- Live webcam, local file, low-bandwidth mode (320×240, 10 fps); real timestamps.
- AE/AWB/focus lock after convergence, with brightness guards (reverts if the lock darkens the image).
- Target: largest face in the first 2 s or click-to-designate; re-acquire only near the last position, with the colour-histogram appearance gate.
- Pixel selection per README 4.3 (except the L* pigmentation split and the temporal-variance mask).
- 2×2 sub-patches as fusion units with README 4.4 adaptive pooling into the parent patch.
- CP3 signal core (default), parity-tested against MATLAB; interim signal core incl. motion suppression (4.8.2) selectable.
- Status codes, never silent; MASKED only for occlusion (low skin ratio), not for patches that are merely small.
- PWA with a precaching service worker; on-screen privacy statement; low-light warning; ITA readout.

## Tests (synthetic, plus CP3 MATLAB-parity fixtures)

| file | what |
|---|---|
| `core.test.ts`, `seedParity.test.ts`, `pos.test.ts`, `kalman.test.ts`, `viterbi.test.ts` | signal core, incl. MASKED only on occlusion |
| `motion.test.ts` | README 4.8.2: periodic sway at 1.6 Hz / 0.5 Hz captures the tracker without suppression, 72 bpm recovered with it; jitter never triggers it |
| `frontEnd.test.ts`, `stability.test.ts`, `faceSynth.ts` | pixel selection, sub-patches, erosion; rendered synthetic face with landmark jitter and drift through 1-Euro + FrontEnd + core |
| `cp3Parity.test.ts`, `cp3Core.test.ts` | CP3: Butterworth coefficients and window HRs vs MATLAB (UBFC dev 1, 3, 12), online core (tracking off) vs MATLAB offline, synthetic 72 bpm at 30 / 10 fps with each tracker, NO_FACE |
| `troikaTrack.test.ts`, `viterbiTrack.test.ts`, `trackScenarios.ts`, `mt19937.ts` | output trackers: 85 / 65 bpm motion-burst scenario, step scaling, gaps / re-init, MATLAB parity (`troika_parity.json`) |
| `subpatch.test.ts` | sub-patch fusion and adaptive pooling |
| `appearance.test.ts` | appearance gate |

Decisions waiting for the operator, and the synthetic evidence behind each web-local
tunable, are in `NOTES_FOR_OPERATOR.md`.

## Known limitations

- CP3 parity covers 3 UBFC dev subjects × 60 s (offline window HRs). The interim core's parity with MATLAB / fvcore is still pending golden vectors (`shared/golden/` does not exist yet).
- Viterbi tracking delays each reading by 3 s; TROIKA has no lag but does not recover from a wrong start (see above).
- CP3 online: the 1 Hz analysis recomputes 15 s × 10 patches each second (8192-point FFTs); measured median 13 ms, max 43 ms per tick in Node on the subject-3 fixture; not yet profiled in the browser worker.
- Not implemented: L* pigmentation sub-clusters (C2 vitiligo), temporal-variance texture mask.
- The Playwright smoke test (`e2e/smoke.spec.ts`) uses the system Chrome (`channel: 'chrome'`) and the already running dev server; it is skipped when no server answers. It checks that the app loads, the camera path runs on a synthetic-face fake camera and shows a status, no request goes to another host, and no page error occurs. It does not check HR (MediaPipe on a drawn face reports LOW_COVERAGE).
- `TraceFrame.subpatches` is a web extension of the 11.1 contract (optional field) pending approval. At IOD ≈ 70 px most sub-patch families are below `npix_min` and pool into their parents, so sub-patch fusion mainly matters for larger faces or locally corrupted patches. The 1 Hz analysis can take ≈140 ms in the worker.
- Frame rejection on large inter-frame motion and patch erosion are implemented but off (no benefit on synthetic data).
- Motion suppression cannot win against a motion peak more than ≈3× the pulse peak in every patch (γ = 0.7 floor).
- Small faces (IOD ≈ 40 px at 640×480) produce mostly LOW_SNR on synthetic data; sub-patches pool, but the patches themselves are at or below `npix_min`.
- All tuning was done on synthetic data; nothing here is validated on real video. Not a medical device.
