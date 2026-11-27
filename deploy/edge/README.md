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

## Initial testing (Phase 1)

All commands from the repo root, with the venv active.

**1. Smoke test, no dataset needed.** A synthetic clip with a known pulse (66 → 84 bpm):
```bash
python deploy/edge/make_test_clip.py
python deploy/edge/bench_video.py outputs/edge/synthetic/vid.avi --gt outputs/edge/synthetic/ground_truth.txt
```
Pass: face found 100 %, MAE under 1 bpm for all three methods. This proves the install, not the accuracy.

**2. UBFC on the Pi.** Copy subjects to `data/raw/UBFC/DATASET_2/subjectN/` (`vid.avi` + `ground_truth.txt`), e.g. from the laptop:
```bash
rsync -a data/raw/UBFC/DATASET_2/subject{1,3,4} <user>@<pi>:TechTronics-2027-SP-Cup/data/raw/UBFC/DATASET_2/
```
- Install check, same code as the laptop: `python src/python/run_c2.py --datasets UBFC --subjects subject1 subject3 subject4`.
  The `pos` MAE must match the laptop (1.53 / 3.23 / 2.61 bpm).
- Edge benchmark of one subject: `python deploy/edge/bench_video.py --ubfc 1`

**3. Timing sweep.** Frame width 640 / 320 × detection every 1 / 2 / 3 / 5 frames. It waits for the Pi to cool below 60 °C between runs:
```bash
bash deploy/edge/sweep.sh --ubfc 1
python deploy/edge/summary.py        # table of every run, also outputs/edge/summary.csv
```
Pick the cheapest setting that keeps **fps ≥ 30** (or ≥ the webcam's fps) with no MAE loss. That is the setting for Phase 2.

**What `bench_video.py` reports:** fps vs real time, the face-found rate, ms per stage (decode, resize, detect, background, roi_skin), HR cost per window, MAE/RMSE/SUC@5 per method (`pos`, `pos_bg`, `pos_snr_bg`), the CPU temperature, and `get_throttled`. Results go to `outputs/edge/<name>.json`. If the throttle value is not `0x0`, the Pi slowed down during the run, so the timings are not valid.

**Laptop reference** (x86 laptop, synthetic clip, 640×480): detection every frame gives 51 fps at 19.5 ms/frame; detection every 5 frames gives 71 fps. Face detection is ~14.5 ms per call, which is about ¾ of the work when it runs every frame. Expect the Pi 4 to be several times slower.

## What is reused

| Stage | Code |
|---|---|
| Face box | `facevitals.face.FaceBox` (MediaPipe full-range detector, largest face) |
| ROI + skin mask | `face.roi_rects`, `face.skin_mask`, regions and thresholds from `config/face_config.json` |
| HR | `facevitals.rppg`: POS, 0.7–3 Hz band-pass, 10 s windows, periodogram peak (`sanity_hr`); `sanity_hr_bg` / `sanity_hr_snr` for the improved methods |

`face.face_traces` reads a whole video and then computes HR offline. For the webcam, the same steps
have to run frame by frame on a rolling 10 s buffer.

## Plan

### Phase 1: dataset videos on the Pi (tools ready, see "Initial testing")
- [ ] Synthetic smoke test passes on the Pi
- [ ] UBFC `run_c2.py` MAE equals the laptop's (install check)
- [ ] Sweep: ms/frame and fps at 640 / 320 × detection every 1 / 2 / 3 / 5 frames
- [ ] Choose the setting for the webcam; check no throttling (heatsink or fan: the Pi 4 throttles at 80 °C)

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
