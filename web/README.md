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
| `src/front/worker.ts` | Web Worker: pixel selection (`FrontEnd`) and the signal core; `makeCore()` is the one line fvcore-WASM will replace |
| `src/front/frontEnd.ts` | README 4.3 pixel selection → TraceFrames (11.1): adaptive (Cb, Cr) skin model, specular and texture masks, trimmed mean, occlusion EMA/hysteresis, 2×2 sub-patches |
| `src/front/appearance.ts` | README 4.1 colour-histogram appearance gate (Bhattacharyya distance) |
| `src/core/interimCore.ts` | interim TS signal core (README 4.4–4.9): Kalman, resampling, POS/CHROM, detrend, BPF, SNR/coherence/coverage weights, top-k spectral fusion, harmonic sum, motion suppression, fixed-lag Viterbi |
| `src/core/motion.ts` | README 4.8.2 spectral motion suppression |
| `src/config.ts` | imports `shared/patches.json` and `shared/config.default.json` (single sources of truth) |
| `src/localConfig.ts` | **web-local tunables pending move to `shared/`** (operator approval; see `NOTES_FOR_OPERATOR.md`) |
| `src/ui/` | React UI: video + patch overlay, HR / confidence / status, BVP, spectrum with Viterbi track, CSV and `.tracejsonl` export |
| `tests/` | Vitest; synthetic signals and rendered synthetic faces only |

## Implemented (README 14.1 checklist)

- Live webcam, local file, low-bandwidth mode (320×240, 10 fps); real timestamps.
- AE/AWB/focus lock after convergence, with brightness guards (reverts if the lock darkens the image).
- Target: largest face in the first 2 s or click-to-designate; re-acquire only near the last position, with the colour-histogram appearance gate.
- Pixel selection per README 4.3 (except the L* pigmentation split and the temporal-variance mask).
- 2×2 sub-patches as fusion units with README 4.4 adaptive pooling into the parent patch.
- Interim signal core incl. motion suppression (4.8.2) using the TraceFrame motion signals.
- Status codes, never silent; MASKED only for occlusion (low skin ratio), not for patches that are merely small.
- PWA with a precaching service worker; on-screen privacy statement; low-light warning; ITA readout.

## Tests (all synthetic)

| file | what |
|---|---|
| `core.test.ts`, `seedParity.test.ts`, `pos.test.ts`, `kalman.test.ts`, `viterbi.test.ts` | signal core, incl. MASKED only on occlusion |
| `motion.test.ts` | README 4.8.2: periodic sway at 1.6 Hz / 0.5 Hz captures the tracker without suppression, 72 bpm recovered with it; jitter never triggers it |
| `frontEnd.test.ts`, `stability.test.ts`, `faceSynth.ts` | pixel selection, sub-patches, erosion; rendered synthetic face with landmark jitter and drift through 1-Euro + FrontEnd + core |
| `subpatch.test.ts` | sub-patch fusion and adaptive pooling |
| `appearance.test.ts` | appearance gate |

Decisions waiting for the operator, and the synthetic evidence behind each web-local
tunable, are in `NOTES_FOR_OPERATOR.md`.

## Known limitations

- Interim TS core; parity with MATLAB / fvcore is pending golden vectors (`shared/golden/` does not exist yet).
- Not implemented: L* pigmentation sub-clusters (C2 vitiligo), temporal-variance texture mask.
- The Playwright smoke test (`e2e/smoke.spec.ts`) uses the system Chrome (`channel: 'chrome'`) and the already running dev server; it is skipped when no server answers. It checks that the app loads, the camera path runs on a synthetic-face fake camera and shows a status, no request goes to another host, and no page error occurs. It does not check HR (MediaPipe on a drawn face reports LOW_COVERAGE).
- `TraceFrame.subpatches` is a web extension of the 11.1 contract (optional field) pending approval. At IOD ≈ 70 px most sub-patch families are below `npix_min` and pool into their parents, so sub-patch fusion mainly matters for larger faces or locally corrupted patches. The 1 Hz analysis can take ≈140 ms in the worker.
- Frame rejection on large inter-frame motion and patch erosion are implemented but off (no benefit on synthetic data).
- Motion suppression cannot win against a motion peak more than ≈3× the pulse peak in every patch (γ = 0.7 floor).
- Small faces (IOD ≈ 40 px at 640×480) produce mostly LOW_SNR on synthetic data; sub-patches pool, but the patches themselves are at or below `npix_min`.
- All tuning was done on synthetic data; nothing here is validated on real video. Not a medical device.
