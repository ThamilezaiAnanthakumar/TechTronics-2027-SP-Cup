# Edge deployment

Runs the heart-rate pipeline on a small device. The first target is a **Raspberry Pi 4** (64-bit OS).
Testing starts with dataset videos, then moves to a USB webcam.

MATLAB does not run on the Pi, so this is built on the Python port in `src/python/facevitals`
(from PR #2), not on the MATLAB module in `src/face`.

## Setup (on the Pi)

Use **64-bit** Raspberry Pi OS (`uname -m` prints `aarch64`). Bookworm is recommended because it ships Python 3.11.

```bash
git clone https://github.com/ThamilezaiAnanthakumar/TechTronics-2027-SP-Cup.git
cd TechTronics-2027-SP-Cup && git switch edge-deploy
bash deploy/edge/setup.sh            # venv in deploy/edge/.venv + import check
source deploy/edge/.venv/bin/activate
sudo apt install v4l-utils           # v4l2-ctl, for the webcam settings
```

`mediapipe` is pinned to **0.10.18**. It is the last release that has both the legacy `mp.solutions` API
(used by `facevitals/face.py`) and Linux aarch64 wheels. mediapipe 1.x removed `mp.solutions`, and
0.10.20–0.10.35 have no aarch64 wheels. 0.10.18 needs Python 3.9–3.12, so on Trixie (3.13) install
3.12 with `uv` (`setup.sh` prints the command). The first install can be slow because pip resolves the
`jax` dependency. Tested on an x86 laptop with Python 3.12, not yet on the Pi.

## What is reused

| Stage | Code |
|---|---|
| Face box | `facevitals.face.FaceBox` (MediaPipe full-range detector, largest face) |
| ROI + skin mask | `face.roi_rects`, `face.skin_mask`, regions and thresholds from `config/face_config.json` |
| HR | `facevitals.rppg`: POS, 0.7–3 Hz band-pass, 10 s windows, periodogram peak (`sanity_hr`); `sanity_hr_bg` / `sanity_hr_snr` for the improved methods |

`face.face_traces` reads a whole video and then computes HR offline. For the webcam, the same steps
have to run frame by frame on a rolling 10 s buffer.

## Plan

### Phase 1: dataset videos on the Pi
Copy a few UBFC subjects to the Pi (`data/raw/UBFC/DATASET_2/subjectN/`) and run the existing offline path.
- [ ] Same video, same method → the same MAE as on the laptop (this checks the install)
- [ ] Time each stage (ms/frame: decode, detect, ROI/skin, HR) at VGA and QVGA
- [ ] Find the detection period (`det.every`) that keeps up with 30 fps on the Pi 4, holding the box in between
- [ ] Log CPU temperature and throttling (`vcgencmd measure_temp`, `vcgencmd get_throttled`). The Pi 4 throttles at 80 °C, so use a heatsink or fan.

### Phase 2: USB webcam (real world)
`run_edge.py`: webcam → face box → ROI RGB means → 10 s ring buffer → HR every 1 s.
- [ ] Capture with `cv2.VideoCapture(0, cv2.CAP_V4L2)` at 640×480, MJPG, 30 fps
- [ ] Timestamp each frame at capture (`time.monotonic()` right after `grab()`). Webcam `CAP_PROP_POS_MSEC` is unreliable, and the real fps drifts in low light.
- [ ] **Lock auto-exposure, auto white balance and auto-gain** (`v4l2-ctl -d /dev/video0 --list-ctrls`, then set them to manual). Automatic adjustments change the skin colour over time, and POS picks that up as a false pulse.
- [ ] Use steady light. Mains-powered lamps flicker at 100 Hz (50 Hz mains in Sri Lanka), so keep the exposure a multiple of 10 ms.
- [ ] Reference HR from a fingertip pulse oximeter, recorded at the same time, for the MAE
- [ ] Measure sustained fps, latency to the first HR, CPU % and memory

### Later
- [ ] Move `face.py` to the MediaPipe Tasks `FaceDetector`, so newer mediapipe and Python 3.13 work
- [ ] Test at ≤ 10 fps and ≤ 500 kbps (C5)

## Notes

- The Python port has no C3 multi-face tracking (largest face only) and only the absolute occlusion rule. See `src/python/README.md`.
- `pandas` (dataset loaders) is not installed on the device. Import only `facevitals.face` and `facevitals.rppg` there.
- `scikit-image` is installed only because `face.py` imports `rgb2lab` (for ITA) at module level. Moving that import inside the ITA functions would drop it.
