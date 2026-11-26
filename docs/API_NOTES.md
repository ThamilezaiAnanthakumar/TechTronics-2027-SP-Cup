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
