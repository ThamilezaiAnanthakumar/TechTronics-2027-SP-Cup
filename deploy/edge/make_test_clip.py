"""
Synthetic smoke-test clip with a known pulse, in UBFC layout (vid.avi + ground_truth.txt).

A still face (scikit-image's astronaut photo) on a grey wall, with a small head sway, sensor
noise and a pulse added to the skin pixels (green strongest, as in real rPPG). The HR ramps
from --hr0 to --hr1 bpm. Use it to check the install and the timing without a dataset:

  python deploy/edge/make_test_clip.py                 # -> outputs/edge/synthetic/
  python deploy/edge/bench_video.py outputs/edge/synthetic/vid.avi --gt outputs/edge/synthetic/ground_truth.txt

It only checks that the pipeline runs and finds the right frequency; it says nothing about
accuracy on real faces.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from skimage import data

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src" / "python"))
from facevitals.face import FaceBox, load_face_config, skin_mask  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=30)
    ap.add_argument("--fps", type=float, default=30)
    ap.add_argument("--hr0", type=float, default=66, help="HR at the start (bpm)")
    ap.add_argument("--hr1", type=float, default=84, help="HR at the end (bpm)")
    ap.add_argument("--amp", type=float, default=1.5, help="pulse amplitude in grey levels (green)")
    ap.add_argument("--out", default=str(REPO / "outputs" / "edge" / "synthetic"))
    a = ap.parse_args()

    W, H = 640, 480
    face = data.astronaut()[30:330, 120:390]                   # RGB, head and shoulders
    fh, fw = face.shape[:2]
    _, skin, _ = load_face_config()
    # pulse only on skin inside an ellipse over the detected face: skin-coloured pixels elsewhere
    # in the photo (suit, flag) would leak the pulse into the background reference trace
    det = FaceBox()
    x, y, bw, bh = det(np.ascontiguousarray(face))
    det.close()
    yy, xx = np.mgrid[:fh, :fw]
    ell = ((xx - (x + bw / 2)) / (0.5 * bw)) ** 2 + ((yy - (y + bh / 2)) / (0.55 * bh)) ** 2 <= 1
    m = (skin_mask(face, skin) & ell)[..., None].astype(np.float32)
    gain = np.array([0.33, 0.77, 0.53], np.float32)           # pulse colour (R, G, B), G = 0.77

    n = int(round(a.seconds * a.fps))
    t = np.arange(n) / a.fps
    hr = np.linspace(a.hr0, a.hr1, n)                          # bpm
    phase = 2 * np.pi * np.cumsum(hr / 60) / a.fps
    pulse = np.sin(phase) + 0.25 * np.sin(2 * phase)           # with a small first harmonic

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    # lossless codec so the small pulse survives; MJPG as a fallback
    for fourcc in ("FFV1", "MJPG"):
        vw = cv2.VideoWriter(str(out / "vid.avi"), cv2.VideoWriter_fourcc(*fourcc), a.fps, (W, H))
        if vw.isOpened():
            break
    rng = np.random.default_rng(0)
    for k in range(n):
        frame = np.full((H, W, 3), 128, np.float32)
        dx = int(round(6 * np.sin(2 * np.pi * 0.2 * t[k])))   # slow head sway, 6 px
        dy = int(round(3 * np.sin(2 * np.pi * 0.13 * t[k])))
        x0, y0 = (W - fw) // 2 + dx, (H - fh) // 2 + dy
        f = face.astype(np.float32) + (a.amp / 0.77) * pulse[k] * gain * m
        frame[y0:y0 + fh, x0:x0 + fw] = f
        frame += rng.normal(0, 2.0, frame.shape).astype(np.float32)   # sensor noise
        vw.write(cv2.cvtColor(np.clip(frame, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    vw.release()

    # UBFC ground_truth.txt: line 1 PPG, line 2 HR (bpm), line 3 time (s)
    with open(out / "ground_truth.txt", "w") as fh_:
        for row in (pulse, hr, t):
            fh_.write(" ".join(f"{v:.6e}" for v in row) + "\n")
    print(f"{out / 'vid.avi'} ({fourcc}, {n} frames, {W}x{H} @ {a.fps:g} fps, HR {a.hr0:g} -> {a.hr1:g} bpm)")


if __name__ == "__main__":
    main()
