# API notes (probe results)

Each entry: question, probe script, environment, answer. Re-run the probe after a MATLAB update.

## faceDetector (RetinaFace) — availability and outputs
- **Probe:** `tools/probe_facedetector.m` (synthetic/shipped image `visionteam1.jpg` only).
- **Environment:** MATLAB R2026b Update 1 (26.2.0.3433840), Computer Vision Toolbox 26.2, support package *Computer Vision Toolbox Model for RetinaFace Face Detection* installed 26 Nov 2026 with `mpm` into `C:\ProgramData\MATLAB\SupportPackages\R2026b`.
- **Answers:**
  - `faceDetector("small-network")` and `faceDetector("large-network")` both load.
  - Object properties: `ModelName`, `ClassNames`, `InputSize`; only method `detect`.
  - `detect` syntaxes: `bboxes`, `[bboxes, scores]`, `[bboxes, scores, labels]`, datastore input, optional `roi`, name-value `Threshold`, `SelectStrongest`, `MinSize`, `MaxSize`, `MiniBatchSize`, `AutoResize`, `ExecutionEnvironment`, `Acceleration`.
  - Outputs: `bboxes` single Mx4, `scores` single Mx1, third output = **categorical class labels, not landmarks**.
  - **No facial landmarks are returned.** README Sections 4.1 and 4.2(a) assume 5 landmarks from the detector; that does not hold for MATLAB's `faceDetector`. Landmark source must come from elsewhere (open question to the operator).

## MediaPipe FaceLandmarker from MATLAB (dense landmarks, README 4.1 / 4.2b)
- **Probe:** `tools/probe_mediapipe.m` (one UBFC subject1 frame, local only).
- **Environment:** Python 3.11.3 (`C:\Users\mufte\AppData\Local\Programs\Python\Python311`), `pip install --user mediapipe` → mediapipe 1.2.3, numpy 1.26.4 (operator-approved, 26 Nov 2026). MATLAB R2026b `pyenv(Version=..., ExecutionMode="OutOfProcess")`. Model `models/face_landmarker.task` (same file as the web app; `models/` is git-ignored).
- **Answers:** `vision.FaceLandmarker` (IMAGE mode, `num_faces=4`) returns **478 landmarks per face** (468 face mesh + 10 iris), each `(x, y, z)` with x, y normalised to [0, 1] of the image and z relative depth — the indices of README 4.2(b) (Li et al. 2026) apply directly. Timing: first call 6.4 s (model load), then **51 ms/frame** at 640×480 including the MATLAB→Python copy (CPU, XNNPACK). MediaPipe logs warnings to stderr on start-up; harmless.

## imnoise 'poisson' on double images (C1 low-light simulation, README 5)
- **Probe:** `tools/probe_imnoise_poisson.m` (synthetic constant 256×256 image at 0.05 = 12.75/255).
- **Environment:** MATLAB R2026b Update 1 (26.2.0.3433840), Image Processing Toolbox 26.2.
- **Answers:**
  - `imnoise(I,'poisson')` with **double** `I`: values are treated as photon counts ×1e-12 → std 2.23e-7 (= sqrt(0.05/1e12)). **Practically no noise.** Do not use it for low light on doubles.
  - `imnoise(uint8,'poisson')`: the grey level itself is the photon count → std 3.62 levels at 13 (= sqrt(13)). This is a fixed 255-photon full scale, i.e. it ignores the sensor's photon budget.
  - `poissrnd(I*Nphot)/Nphot` on the attenuated double image gives std sqrt(I/Nphot) as expected (3.16e-3 at Nphot = 5000).
- **Decision (`src/degrade/lowLight.m`):** photon model `J = poissrnd(I_att * Nphot) / Nphot` with `I_att = im2double(I) * g`, `g = lux / 300`, and `Nphot` = photo-electrons at full scale (default **5000 e⁻**, typical full well of a small webcam pixel; gives ≈2.5 grey-level shot noise at mid grey at the 300 lux baseline). The lux dependence enters through `I_att`: the expected count per pixel scales with lux, so relative shot noise grows as 1/sqrt(lux). Gaussian read noise: `imnoise(J,'gaussian',0,var)`, var 1e-4…1e-3. Then `im2uint8` (8-bit quantisation is part of the simulation).
