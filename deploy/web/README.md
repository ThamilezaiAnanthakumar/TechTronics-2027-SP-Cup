# FaceVitals Live (browser)

Live webcam heart rate in the browser, using the same signal chain as the MATLAB pipeline. Use it to test the algorithm on real faces before building a mobile app.

```
camera ─► MediaPipe Face Landmarker (face box) ─► forehead / cheek ROIs (roiBoxes.m fractions)
       ─► skin-masked mean RGB (roiSumRGB.m) ─► resample to 30 Hz ─► POS / CHROM / GREEN
       ─► detrend + bandpass ─► FFT peak of the newest 10 s window, updated every 1 s
```

## Run on a laptop

```
cd deploy/web
python -m http.server 8000
```

Open http://localhost:8000 in Chrome or Edge, press **Start** and allow the camera. Browsers only allow camera access on `localhost` or `https`. On first load the page fetches MediaPipe from jsDelivr and the face model from Google, so it needs internet.

The defaults are **POS / combined**, the best-performing pair on UBFC. The table on the page shows all 12 method × ROI pairs live.

### Checking accuracy
1. Wear a pulse oximeter or smartwatch.
2. Log its reading in "Validate against a reference" every 20–30 s. The page shows MAE and the share of readings within ±5 bpm.
3. Click **Download session CSVs** and analyse in MATLAB:
   ```matlab
   addpath deploy/web/tools
   analyzeWebSession('facevitals_<stamp>_traces.csv', 'facevitals_<stamp>_refs.csv')
   ```

## Files

| File | Purpose |
|---|---|
| `rppg.js` | Port of `src/signal/*.m`, `roiBoxes.m`, `roiSumRGB.m`, plus `LiveEstimator` for streaming. No DOM, so it runs in Node too. |
| `app.js`, `index.html`, `style.css` | Camera, face tracking, UI, charts, CSV export |
| `tests/parity.mjs` | Checks `rppg.js` against MATLAB on exported UBFC traces |
| `tools/exportParityData.m` | Exports `outputs/<subject>/traces.mat` + MATLAB HR to `outputs/web_parity/*.json` |
| `tools/analyzeWebSession.m` | Runs the MATLAB pipeline on a recorded browser session |

## Parity with MATLAB

```
matlab -batch "cd deploy/web/tools; exportParityData()"
node deploy/web/tests/parity.mjs
```

On 28 UBFC subjects the per-window HR from `rppg.js` matches MATLAB to 0.000 bpm for every method and ROI, and the pulse waveforms correlate at 1.00000. The filter design and `filtfilt` edge handling match MATLAB R2024b to about 1e-14.

The test also replays each trace through `LiveEstimator` frame by frame, the way the browser runs. Using only the newest window costs no accuracy. MAE against the contact-PPG reference:

| | MATLAB offline | JS live replay |
|---|---|---|
| POS / combined | 2.64 bpm | 2.02 bpm |
| CHROM / combined | 3.06 bpm | 2.28 bpm |

## Differences from the MATLAB pipeline
- **Face box:** MATLAB uses Viola-Jones plus KLT tracking. Here the box is the extent of the MediaPipe landmarks, lightly smoothed. The ROI fractions are the same, but the box is a bit taller (it reaches the hairline), so the ROIs sit slightly differently.
- **Timing:** webcam frames arrive at irregular times, so samples are resampled to 30 Hz before processing.
- **Camera auto-adjustment:** auto-exposure and auto white-balance hurt rPPG. **Lock camera** turns them off where the browser allows it (Chrome with many USB webcams; most built-in laptop cameras and Safari don't).

Research demo only. Not a medical device.
