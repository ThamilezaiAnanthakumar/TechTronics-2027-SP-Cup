"""
Live heart-rate demo for the edge device (Phase 2).

camera -> face box -> skin ROI means -> rolling buffer -> HR every second, shown in a window
(face box, ROIs, HR of the three methods, pulse waveform, fps, CPU temperature).

  python deploy/edge/run_edge.py                          # USB webcam /dev/video0 (MJPG 640x480 @ 30)
  python deploy/edge/run_edge.py --camera 1               # /dev/video1
  python deploy/edge/run_edge.py --video outputs/edge/synthetic/vid.avi \
         --gt outputs/edge/synthetic/ground_truth.txt     # replay a file at its frame rate, with reference HR
  python deploy/edge/run_edge.py --headless --log outputs/edge/live.csv     # no window, print + CSV

Keys in the window: q / Esc quit, m next method, r reset the buffer.

Notes
- The HR is that of the last complete 10 s window, so it lags the person by about 5 s.
- Each frame is timestamped right after grab() (time.monotonic()): webcam timestamps from the
  driver are unreliable and the real fps drifts in low light. When processing is slower than
  the camera, frames are dropped; the timestamps keep the HR correct.
- facevitals.rppg needs >= 270 valid samples in the trace it is given (90 % of 10 s at 30 Hz),
  so the buffer holds 20 s: then a camera at >= 14 fps still gives an HR.
- The camera runs with auto exposure and auto white balance (they are switched on at start, in
  case an earlier run left them locked). --lock locks both after a 2 s warm-up: steadier skin
  colour for POS, but only in steady light.
- On the Pi (aarch64) the MediaPipe full-range face model finds no faces, so the short-range
  model (--det-model 0) is the default there.
"""

import argparse
import collections
import csv
import platform
import sys
import threading
import time
from pathlib import Path

import mediapipe  # noqa: F401  (load before scipy, as run_c2.py does)
import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src" / "python"))
sys.path.insert(0, str(HERE))

from facevitals.face import FaceBox, background_rgb, crop, load_face_config, roi_rects, skin_mask  # noqa: E402
from facevitals.rppg import BAND, FS, _bandpass, pos, ppg_hr, sanity_hr, sanity_hr_bg, sanity_hr_snr  # noqa: E402
from bench_video import BOX_ALPHA, LOST_AFTER_S, cpu_temp, load_ubfc_gt, throttled  # noqa: E402

WIN, STEP = 10.0, 1.0        # HR window and update period (s), as in the offline pipeline
BUF_S = 20.0                 # trace kept in memory (s), see the notes above
METHODS = ["pos", "pos_bg", "pos_snr_bg"]
LABELS = {"pos": "POS", "pos_bg": "POS + background", "pos_snr_bg": "POS + SNR fusion + bg"}


# --------------------------------------------------------------- frame sources

class CameraSource:
    """V4L2 webcam read in its own thread; keeps only the newest frames (drops when busy)."""

    def __init__(self, index, width, height, fps, lock=False, exposure=None, warmup_s=2.0):
        self.cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise IOError(f"cannot open camera {index} (/dev/video{index}). Is a USB webcam connected? "
                          "List cameras with: ls /sys/class/video4linux; cat /sys/class/video4linux/*/name")
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.nominal_fps = self.cap.get(cv2.CAP_PROP_FPS) or fps
        self.size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.lock, self.exposure, self.warmup_s = lock, exposure, warmup_s
        # only write controls that are off: some cheap webcams reset on USB when written to
        if self.cap.get(cv2.CAP_PROP_AUTO_EXPOSURE) == 1:  # V4L2: 1 = manual, 3 = auto (aperture priority)
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
        if self.cap.get(cv2.CAP_PROP_AUTO_WB) == 0:        # -1 = the camera has no such control
            self.cap.set(cv2.CAP_PROP_AUTO_WB, 1)
        self.locked_msg = "auto exposure / WB" if not lock else "warming up (auto exposure)"
        self.lost_since = None     # time the camera stopped delivering frames (unplugged, USB error)
        self.q = collections.deque(maxlen=1)
        self.cv = threading.Condition()
        self.grabbed = self.dropped = 0
        self.stop_flag = False
        self.live = True
        self.th = threading.Thread(target=self._run, daemon=True)
        self.th.start()

    def _lock_controls(self):
        c = self.cap
        exp = c.get(cv2.CAP_PROP_EXPOSURE) if self.exposure is None else self.exposure
        c.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)          # V4L2: 1 = manual, 3 = aperture priority (auto)
        c.set(cv2.CAP_PROP_EXPOSURE, exp)
        wb = c.get(cv2.CAP_PROP_WB_TEMPERATURE)
        c.set(cv2.CAP_PROP_AUTO_WB, 0)
        if wb > 0:
            c.set(cv2.CAP_PROP_WB_TEMPERATURE, wb)
        self.locked_msg = f"exposure locked at {exp:g}, WB locked" + (f" at {wb:g} K" if wb > 0 else "")

    def _run(self):
        t_start = time.monotonic()
        while not self.stop_flag:
            try:
                ok = self.cap.grab()
                t = time.monotonic()
                ok, bgr = self.cap.retrieve() if ok else (False, None)
            except cv2.error:              # MJPG decode of an empty buffer: camera unplugged / USB error
                ok = False
            if not ok:
                self.lost_since = self.lost_since or time.monotonic()
                time.sleep(0.05)
                continue
            self.lost_since = None
            self.grabbed += 1
            if self.lock and self.locked_msg.startswith("warming") and t - t_start > self.warmup_s:
                self._lock_controls()
            with self.cv:
                if self.q:
                    self.dropped += 1
                self.q.append((t, bgr))
                self.cv.notify()

    def read(self, timeout=1.0):
        with self.cv:
            if not self.q and not self.cv.wait_for(lambda: bool(self.q), timeout):
                return None
            return self.q.popleft()

    def close(self):
        self.stop_flag = True
        self.th.join(timeout=2)
        if self.lock:
            # V4L2 controls outlive the program: give the camera back its automatic settings
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
            self.cap.set(cv2.CAP_PROP_AUTO_WB, 1)
        self.cap.release()

    def lost(self, after_s=3.0):
        return self.lost_since is not None and time.monotonic() - self.lost_since > after_s


class FileSource:
    """A video file played like a camera: frame times from the file, paced to real time.
    No frames are dropped: if processing is slower than the file, playback is slower too
    (the HR stays correct because it uses the file's frame times)."""

    def __init__(self, path, loop=False, realtime=True):
        self.path, self.loop, self.realtime = str(path), loop, realtime
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise IOError(f"cannot open {path}")
        self.nominal_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.k, self.offset, self.t0 = 0, 0.0, None
        self.loops = 0             # completed passes (--loop); the caller resets its buffer on a new pass
        self.grabbed = self.dropped = 0
        self.live = False
        self.locked_msg = f"file {Path(path).name}"

    def read(self, timeout=None):
        ok, bgr = self.cap.read()
        if not ok:
            if not self.loop:
                return None
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.offset += self.k / self.nominal_fps
            self.k = 0
            self.loops += 1
            ok, bgr = self.cap.read()
            if not ok:
                return None
        t = self.offset + self.k / self.nominal_fps
        self.k += 1
        self.grabbed += 1
        if self.realtime:
            now = time.monotonic()
            if self.t0 is None:
                self.t0 = now - t
            wait = self.t0 + t - now
            if wait > 0:
                time.sleep(wait)
        return t, bgr

    def close(self):
        self.cap.release()


# --------------------------------------------------------------- per-frame features

class FrameProcessor:
    """Same per-frame steps as bench_video.traces: face box (held between detections),
    background trace, per-ROI skin means and the face mean with the occlusion rule."""

    def __init__(self, det_model, det_every, max_width):
        self.roi, self.skin, occl = load_face_config()
        self.names = list(self.roi["names"])
        self.on_below = occl.get("onBelow", 0.40)
        self.min_pix = occl.get("minPix", 10)
        self.det = FaceBox(model=det_model)
        self.det_every, self.max_width = det_every, max_width
        self.box, self.last_seen, self.k = None, -np.inf, 0
        self.lost = False          # last detector call found no face (box held for up to LOST_AFTER_S)

    def reset(self):
        self.box, self.last_seen, self.lost = None, -np.inf, False

    def __call__(self, t, bgr):
        h, w = bgr.shape[:2]
        if self.max_width and w > self.max_width:
            bgr = cv2.resize(bgr, (self.max_width, int(round(h * self.max_width / w))), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        t_det = None
        if self.k % self.det_every == 0:
            t1 = time.perf_counter()
            b = self.det(rgb)
            t_det = 1000 * (time.perf_counter() - t1)
            self.lost = b is None
            if b is not None:
                self.box = b if self.box is None else (1 - BOX_ALPHA) * self.box + BOX_ALPHA * b
                self.last_seen, q = t, 2
            elif self.box is not None and t - self.last_seen <= LOST_AFTER_S:
                q = 1
            else:
                self.box, q = None, 0
        else:
            q = 1 if self.box is not None else 0
        self.k += 1

        bg = background_rgb(rgb, self.box, self.skin)[0]
        face = np.full(3, np.nan)
        regions = np.full((len(self.names), 3), np.nan)
        rects = {}
        if q > 0:
            sums, nall = np.zeros(3), 0
            rects = roi_rects(self.box, self.roi, self.names)
            for i, rect in enumerate(rects.values()):
                patch = crop(rgb, rect)
                if patch.size == 0:
                    continue
                m = skin_mask(patch, self.skin)
                ns = int(m.sum())
                if ns > 0:
                    s = patch[m].sum(axis=0, dtype=np.float64)
                    regions[i] = s / ns
                    if not (ns / m.size < self.on_below or ns < self.min_pix):
                        sums += s
                        nall += ns
            if nall > 0:
                face = sums / nall
        return dict(bgr=bgr, q=q, box=self.box, held=self.lost, rects=rects, face=face, regions=regions, bg=bg,
                    det_ms=t_det, bright=float(rgb[::8, ::8].mean()))

    def close(self):
        self.det.close()


# --------------------------------------------------------------- rolling buffer and HR

class HRBuffer:
    def __init__(self, n_regions):
        self.R = n_regions
        self.reset()

    def reset(self):
        self.t = collections.deque()
        self.face, self.regions, self.bg, self.q = (collections.deque() for _ in range(4))

    def add(self, t, f):
        self.t.append(t); self.face.append(f["face"]); self.regions.append(f["regions"])
        self.bg.append(f["bg"]); self.q.append(f["q"])
        while self.t and self.t[-1] - self.t[0] > BUF_S:
            for d in (self.t, self.face, self.regions, self.bg, self.q):
                d.popleft()

    def span(self):
        return self.t[-1] - self.t[0] if len(self.t) > 1 else 0.0

    def estimate(self):
        """HR (bpm) of each method for the latest complete window, and the pulse waveform."""
        t = np.array(self.t)
        face, bg = np.array(self.face), np.array(self.bg)
        reg = np.array(self.regions).reshape(-1, self.R, 3)
        c = [t[-1] - WIN / 2]
        hr = {
            "pos": sanity_hr(t, face, WIN, STEP, centers=c)[0][0],
            "pos_bg": sanity_hr_bg(t, face, bg, WIN, STEP, centers=c)[0][0],
            "pos_snr_bg": sanity_hr_snr(t, [reg[:, i, :] for i in range(self.R)], WIN, STEP, centers=c, rgb_bg=bg)[0][0],
        }
        # pulse waveform of the last window (POS on the face trace, band-passed), for the plot
        wave = None
        sel = (t >= t[-1] - WIN) & np.all(np.isfinite(face), axis=1)
        if sel.sum() > 2 * FS:
            tv = t[sel]
            tu = np.arange(tv[0], tv[-1] + 1e-9, 1 / FS)
            X = np.column_stack([np.interp(tu, tv, face[sel, k]) for k in range(3)])
            if len(tu) > 3 * 1.6 * FS:
                wave = _bandpass(pos(X, FS), FS, BAND)
        face_ok = 100 * float(np.mean(np.array(self.q)[t >= t[-1] - WIN] > 0))
        return hr, wave, face_ok


# --------------------------------------------------------------- drawing

GREEN, YELLOW, RED, CYAN, WHITE, GREY = (80, 220, 80), (0, 220, 255), (60, 60, 255), (255, 220, 0), (255, 255, 255), (150, 150, 150)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def exposure_warning(bright):
    """Mean frame brightness (0-255) outside the range where a face can be found and its colour measured."""
    if bright < 30:
        return f"image too dark ({bright:.0f}/255): camera covered, or more light needed"
    if bright > 225:
        return f"image too bright ({bright:.0f}/255): overexposed"
    return ""


def _text(img, s, org, scale=0.55, color=WHITE, thick=1):
    cv2.putText(img, s, org, FONT, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, s, org, FONT, scale, color, thick, cv2.LINE_AA)


def draw(f, st):
    img = f["bgr"].copy()
    H, W = img.shape[:2]
    if f["box"] is not None:
        x, y, bw, bh = [int(round(v)) for v in f["box"]]
        cv2.rectangle(img, (x, y), (x + bw, y + bh), YELLOW if f["held"] else GREEN, 2)   # yellow: face lost, box held
        for (x1, y1, x2, y2) in f["rects"].values():
            cv2.rectangle(img, (x1, y1), (x2, y2), CYAN, 1)

    # HR panel
    ov = img.copy()
    cv2.rectangle(ov, (0, 0), (min(W, 330), 150), (0, 0, 0), -1)
    img = cv2.addWeighted(ov, 0.45, img, 0.55, 0)
    m = st["method"]
    hr = st["hr"].get(m, np.nan)
    if st["span"] < WIN:
        _text(img, f"collecting signal {st['span']:4.1f} / {WIN:.0f} s", (10, 30), 0.65, YELLOW, 2)
        cv2.rectangle(img, (10, 42), (10 + int(300 * min(1, st["span"] / WIN)), 52), YELLOW, -1)
    elif np.isfinite(hr):
        _text(img, f"{hr:5.1f}", (10, 58), 1.8, GREEN, 3)
        _text(img, "bpm", (205, 58), 0.8, GREEN, 2)
    else:
        _text(img, "  --  ", (10, 58), 1.8, GREY, 3)
    _text(img, LABELS[m] + "  (m: switch)", (10, 82), 0.5, WHITE)
    line = "  ".join(f"{k}: {st['hr'][k]:.0f}" if np.isfinite(st["hr"].get(k, np.nan)) else f"{k}: --" for k in METHODS)
    _text(img, line, (10, 104), 0.45, GREY)
    if st["ref"] is not None:
        mae = f"  MAE {st['mae']:.2f}" if np.isfinite(st["mae"]) else ""
        _text(img, f"reference {st['ref']:.1f} bpm{mae}", (10, 126), 0.5, CYAN)
    if f["box"] is None:
        _text(img, "no face", (W // 2 - 50, H // 2), 0.9, RED, 2)
    warn = exposure_warning(f["bright"])
    if warn:
        _text(img, warn, (10, H // 2 + 40), 0.6, RED, 2)

    # status line
    temp = f"{st['temp']:.0f} C" if st["temp"] is not None else ""
    thr = "  THROTTLED" if st["throttled"] not in (None, "0x0") else ""
    stat = f"proc {st['fps']:4.1f} fps"
    if st["live"]:
        stat += f" | cam {st['cam_fps']:4.1f} fps | dropped {st['drop_pct']:3.0f}%"
    stat += f" | face {st['face_ok']:3.0f}% | det every {st['det_every']} ({st['det_ms']:.0f} ms) | {temp}{thr}"
    _text(img, stat, (10, H - 30), 0.45, RED if thr else WHITE)
    _text(img, st["src_msg"], (10, H - 10), 0.45, GREY)

    # pulse waveform strip
    strip = np.full((110, W, 3), 25, np.uint8)
    _text(strip, "pulse (POS, last 10 s)", (8, 18), 0.45, GREY)
    w = st["wave"]
    if w is not None and len(w) > 10 and np.ptp(w) > 0:
        ys = 60 - 40 * (w - w.mean()) / (np.abs(w - w.mean()).max() + 1e-12)
        xs = np.linspace(5, W - 5, len(w))
        cv2.polylines(strip, [np.column_stack([xs, ys]).astype(np.int32)], False, GREEN, 2, cv2.LINE_AA)
    return np.vstack([img, strip])


# --------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--camera", type=int, default=0, help="V4L2 camera index (default 0 = /dev/video0)")
    src.add_argument("--video", help="replay a video file instead of the camera")
    ap.add_argument("--gt", help="UBFC-format ground_truth.txt for --video: shows the reference HR and MAE")
    ap.add_argument("--loop", action="store_true", help="loop the video file")
    ap.add_argument("--fast", action="store_true", help="video file: do not pace to real time")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=float, default=30)
    ap.add_argument("--lock", action="store_true", help="lock exposure / white balance after 2 s "
                                                         "(default: auto exposure and auto white balance)")
    ap.add_argument("--exposure", type=float, help="manual exposure (V4L2 units, usually 100 us: 100 = 10 ms, "
                                                   "a multiple of 10 ms avoids 50 Hz mains flicker)")
    ap.add_argument("--max-width", type=int, default=640, help="downscale wider frames")
    ap.add_argument("--det-every", type=int, default=3, help="face detection period in frames (box held between)")
    ap.add_argument("--det-model", type=int, choices=(0, 1), default=None,
                    help="MediaPipe model: 0 short-range, 1 full-range (default: 0 on aarch64, 1 elsewhere)")
    ap.add_argument("--method", choices=METHODS, default="pos", help="method shown large")
    ap.add_argument("--headless", action="store_true", help="no window: print one line per HR update")
    ap.add_argument("--fullscreen", action="store_true")
    ap.add_argument("--log", help="CSV file, one row per HR update")
    ap.add_argument("--seconds", type=float, help="stop after this many seconds")
    a = ap.parse_args()
    if a.det_model is None:
        a.det_model = 0 if platform.machine() == "aarch64" else 1

    if a.video:
        source = FileSource(a.video, loop=a.loop, realtime=not a.fast)
    else:
        source = CameraSource(a.camera, a.width, a.height, a.fps, lock=a.lock, exposure=a.exposure)
    ref_fn = None
    if a.gt:
        tg, ppg = load_ubfc_gt(a.gt)
        ref_fn = lambda tc: ppg_hr(tg, ppg, WIN, STEP, centers=[tc])[0][0]  # noqa: E731

    proc = FrameProcessor(a.det_model, a.det_every, a.max_width)
    buf = HRBuffer(len(proc.names))
    print(f"source: {'file ' + a.video if a.video else f'camera {a.camera}'} {source.size[0]}x{source.size[1]} "
          f"@ {source.nominal_fps:g} fps | det model {a.det_model}, every {a.det_every} | method {a.method}")

    win_name = "FaceVitals edge demo"
    if not a.headless:
        # GUI_NORMAL: no Qt toolbar; AUTOSIZE: the window has the image's size (640x590)
        if a.fullscreen:
            cv2.namedWindow(win_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL)
            cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)

    log_f = log_w = None
    if a.log:
        Path(a.log).parent.mkdir(parents=True, exist_ok=True)
        log_f = open(a.log, "w", newline="")
        log_w = csv.writer(log_f)
        log_w.writerow(["t_s", *[f"hr_{m}" for m in METHODS], "hr_ref", "proc_fps", "cam_fps", "face_pct",
                        "hr_ms", "temp_c", "throttled"])

    st = dict(method=a.method, hr={}, wave=None, span=0.0, ref=None, mae=np.nan, fps=0.0, cam_fps=0.0,
              drop_pct=0.0, face_ok=0.0, det_every=a.det_every, det_ms=0.0, temp=cpu_temp(),
              throttled=throttled(), live=source.live, src_msg=source.locked_msg)
    errs = []
    next_hr, next_temp, loops_seen = None, 0.0, 0
    t_wall0 = time.monotonic()
    proc_times = collections.deque(maxlen=60)
    grab_count = collections.deque(maxlen=60)
    try:
        while True:
            item = source.read()
            if item is None:
                if a.video:
                    break
                if source.lost():
                    print("camera stopped delivering frames (unplugged?). Stopping; run again to reconnect.")
                    break
                continue
            t, bgr = item
            if not source.live and source.loops != loops_seen:
                # new pass of a looped file: the jump back to the start is not a real signal
                loops_seen = source.loops
                buf.reset(); proc.reset(); next_hr = None
            f = proc(t, bgr)
            buf.add(t, f)
            now = time.monotonic()
            proc_times.append(now)
            grab_count.append((now, source.grabbed))
            if f["det_ms"] is not None:
                st["det_ms"] = 0.8 * st["det_ms"] + 0.2 * f["det_ms"] if st["det_ms"] else f["det_ms"]
            if len(proc_times) > 1:
                st["fps"] = (len(proc_times) - 1) / max(1e-6, proc_times[-1] - proc_times[0])
                (t_a, n_a), (t_b, n_b) = grab_count[0], grab_count[-1]
                st["cam_fps"] = (n_b - n_a) / max(1e-6, t_b - t_a)      # frames delivered by the camera
            st["drop_pct"] = 100 * source.dropped / max(1, source.grabbed)
            st["span"] = buf.span()
            st["src_msg"] = source.locked_msg

            if next_hr is None:
                next_hr = t + WIN
            if t >= next_hr:
                next_hr += STEP
                t1 = time.perf_counter()
                st["hr"], st["wave"], st["face_ok"] = buf.estimate()
                hr_ms = 1000 * (time.perf_counter() - t1)
                tc = t - WIN / 2
                if ref_fn is not None:
                    st["ref"] = ref_fn(tc - source.offset)     # reference time within the clip
                    e = st["hr"][a.method] - st["ref"]
                    if np.isfinite(e):
                        errs.append(abs(e))
                    st["mae"] = float(np.mean(errs)) if errs else np.nan
                if now > next_temp:
                    st["temp"], st["throttled"], next_temp = cpu_temp(), throttled(), now + 5
                hrs = "  ".join(f"{m} {st['hr'][m]:5.1f}" for m in METHODS)
                ref = f"  ref {st['ref']:5.1f}" if st["ref"] is not None else ""
                tmp = f"  {st['temp']:.0f}C" if st["temp"] is not None else ""
                if a.headless:
                    warn = exposure_warning(f["bright"])
                    print(f"t {t:7.1f}s  {hrs}{ref}  | {st['fps']:4.1f} fps  face {st['face_ok']:3.0f}%  hr {hr_ms:4.1f} ms{tmp}"
                          + (f"  ! {warn}" if warn else ""), flush=True)
                if log_w:
                    log_w.writerow([f"{t:.3f}", *[f"{st['hr'][m]:.2f}" for m in METHODS],
                                    "" if st["ref"] is None else f"{st['ref']:.2f}", f"{st['fps']:.2f}",
                                    f"{st['cam_fps']:.2f}", f"{st['face_ok']:.1f}", f"{hr_ms:.1f}",
                                    st["temp"], st["throttled"]])
                    log_f.flush()                               # readable while the test runs

            if not a.headless:
                cv2.imshow(win_name, draw(f, st))
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("m"):
                    st["method"] = METHODS[(METHODS.index(st["method"]) + 1) % len(METHODS)]
                if key == ord("r"):
                    buf.reset(); proc.reset(); errs.clear(); next_hr = None
            if a.seconds and now - t_wall0 > a.seconds:
                break
    except KeyboardInterrupt:
        pass
    finally:
        source.close()
        proc.close()
        if log_f:
            log_f.close()
        if not a.headless:
            cv2.destroyAllWindows()
    if errs:
        print(f"MAE ({a.method}) over {len(errs)} updates: {np.mean(errs):.2f} bpm")
    print(f"processed {source.grabbed - source.dropped} frames, dropped {source.dropped}, "
          f"mean {st['fps']:.1f} fps (last 60 frames)")


if __name__ == "__main__":
    main()
