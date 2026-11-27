"""
Edge benchmark: the FaceVitals HR pipeline on one video, with per-stage timing.

Same per-frame steps as facevitals.face.face_traces (MediaPipe face box, ROI skin means,
background trace), plus:
  --det-every N   run the face detector every N frames and hold the box in between
  --max-width W   downscale frames wider than W (640 = VGA, 320 = QVGA)
then HR with the three methods (pos, pos_bg, pos_snr_bg) and, if a reference is given,
MAE / RMSE / SUC@5 against the contact PPG.

With --det-every 1 and --max-width 960 the traces equal face_traces, so the MAE must match
run_c2.py on the same video (a check of the install).

  python deploy/edge/bench_video.py --ubfc 1                    # data/raw/UBFC/DATASET_2/subject1
  python deploy/edge/bench_video.py --ubfc 1 --det-every 3 --max-width 320
  python deploy/edge/bench_video.py path/to/vid.avi --gt path/to/ground_truth.txt

Writes outputs/edge/<name>.json (timing, HR error, CPU temperature, throttling).
"""

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import mediapipe  # noqa: F401  (load before scipy, as run_c2.py does)
import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src" / "python"))

from facevitals.face import (FaceBox, _fix_times, background_rgb, crop, load_face_config,  # noqa: E402
                             roi_rects, skin_mask)
from facevitals.rppg import error_stats, ppg_hr, sanity_hr, sanity_hr_bg, sanity_hr_snr  # noqa: E402

BOX_ALPHA = 0.3      # box smoothing on detector frames (as face_traces)
LOST_AFTER_S = 1.0   # hold the last box this long after the face is lost (as face_traces)
WIN, STEP = 10.0, 1.0
STAGES = ["decode", "resize", "detect", "background", "roi_skin"]


# --------------------------------------------------------------- device state

def _run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=2).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def cpu_temp():
    """CPU temperature in degC (vcgencmd on a Pi, else the first thermal zone), or None."""
    s = _run(["vcgencmd", "measure_temp"])           # temp=48.7'C
    if s.startswith("temp="):
        return float(s[5:].split("'")[0])
    try:
        return int(Path("/sys/class/thermal/thermal_zone0/temp").read_text()) / 1000
    except (OSError, ValueError):
        return None


def throttled():
    """vcgencmd get_throttled bits (0x0 = never throttled since boot), or None off a Pi."""
    s = _run(["vcgencmd", "get_throttled"])          # throttled=0x0
    return s.split("=")[1] if "=" in s else None


def device_info():
    model = ""
    try:
        model = Path("/proc/device-tree/model").read_text().strip("\x00\n")
    except OSError:
        model = platform.processor() or platform.machine()
    return dict(model=model, machine=platform.machine(), python=platform.python_version(),
                mediapipe=mediapipe.__version__, opencv=cv2.__version__)


# --------------------------------------------------------------- reference

def load_ubfc_gt(path):
    """UBFC ground_truth.txt: line 1 PPG, line 2 HR, line 3 time (s). No pandas needed."""
    lines = [l for l in Path(path).read_text().splitlines() if l.strip()]
    return np.array(lines[2].split(), float), np.array(lines[0].split(), float)


# --------------------------------------------------------------- traces with timing

def traces(video, roi, skin, occl, max_width, det_every, max_frames, use_bg, det_model=1):
    names = list(roi["names"])
    R = len(names)
    on_below = occl.get("onBelow", 0.40)
    min_pix = occl.get("minPix", 10)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise IOError(f"cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    det = FaceBox(model=det_model)

    tm = {s: [] for s in STAGES}
    T, Q, face, reg, bg = [], [], [], [], []
    box, last_seen, k = None, -np.inf, 0
    temps, t_temp = [], 0.0
    try:
        while max_frames is None or k < max_frames:
            t0 = time.perf_counter()
            t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            ok, bgr = cap.read()
            if not ok:
                break
            t1 = time.perf_counter()
            h, w = bgr.shape[:2]
            if max_width and w > max_width:
                bgr = cv2.resize(bgr, (max_width, int(round(h * max_width / w))), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            t2 = time.perf_counter()

            if k % det_every == 0:
                b = det(rgb)
                if b is not None:
                    box = b if box is None else (1 - BOX_ALPHA) * box + BOX_ALPHA * b
                    last_seen, q = t, 2
                elif box is not None and t - last_seen <= LOST_AFTER_S:
                    q = 1
                else:
                    box, q = None, 0
            else:
                q = 1 if box is not None else 0
            t3 = time.perf_counter()

            bg.append(background_rgb(rgb, box, skin)[0] if use_bg else np.full(3, np.nan))
            t4 = time.perf_counter()

            fr, rr = np.full(3, np.nan), np.full((R, 3), np.nan)
            if q > 0:
                sums, nall = np.zeros(3), 0
                for i, rect in enumerate(roi_rects(box, roi, names).values()):
                    patch = crop(rgb, rect)
                    if patch.size == 0:
                        continue
                    m = skin_mask(patch, skin)
                    ns = int(m.sum())
                    if ns > 0:
                        s = patch[m].sum(axis=0, dtype=np.float64)
                        rr[i] = s / ns
                    if not (ns / m.size < on_below or ns < min_pix):
                        sums += s
                        nall += ns
                if nall > 0:
                    fr = sums / nall
            t5 = time.perf_counter()

            for s, a, b_ in zip(STAGES, (t0, t1, t2, t3, t4), (t1, t2, t3, t4, t5)):
                tm[s].append(1000 * (b_ - a))
            T.append(t); Q.append(q); face.append(fr); reg.append(rr)
            if t5 - t_temp > 2.0:                       # sample the CPU temperature every 2 s
                temps.append(cpu_temp()); t_temp = t5
            k += 1
    finally:
        det.close()
        cap.release()

    reg = np.array(reg).reshape(-1, R, 3)
    return dict(t=_fix_times(T, fps), fps=fps, quality=np.array(Q), rgb_face=np.array(face).reshape(-1, 3),
                regions=[reg[:, i, :] for i in range(R)], rgb_bg=np.array(bg).reshape(-1, 3),
                frame_size=list(rgb.shape[1::-1]) if k else None,
                timing={s: np.array(v) for s, v in tm.items()},
                temps=[x for x in temps if x is not None])


# --------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", nargs="?", help="video file (or use --ubfc)")
    ap.add_argument("--gt", help="UBFC-format ground_truth.txt (reference PPG)")
    ap.add_argument("--ubfc", type=int, help="UBFC subject number in data/raw/UBFC/DATASET_2")
    ap.add_argument("--max-width", type=int, default=640, help="downscale wider frames (640 VGA, 320 QVGA)")
    ap.add_argument("--det-every", type=int, default=1, help="face detection period in frames")
    ap.add_argument("--det-model", type=int, choices=(0, 1), default=1,
                    help="MediaPipe model: 1 full-range (default), 0 short-range (use 0 on the Pi: 1 finds no faces there)")
    ap.add_argument("--max-frames", type=int, default=None, help="only the first N frames")
    ap.add_argument("--no-bg", action="store_true", help="skip the background trace (no pos_bg / pos_snr_bg)")
    ap.add_argument("--name", help="output name (default from the inputs)")
    a = ap.parse_args()

    if a.ubfc is not None:
        d = REPO / "data/raw/UBFC/DATASET_2" / f"subject{a.ubfc}"
        a.video, a.gt = a.video or d / "vid.avi", a.gt or d / "ground_truth.txt"
    if not a.video:
        ap.error("give a video or --ubfc N")
    video = Path(a.video)
    if not video.exists():
        sys.exit(f"error: {video} not found")
    name = a.name or (f"ubfc{a.ubfc}" if a.ubfc is not None else video.parent.name or video.stem)
    name += f"_w{a.max_width}_det{a.det_every}" + (f"_m{a.det_model}" if a.det_model != 1 else "") + (f"_first{a.max_frames}" if a.max_frames else "")

    info = device_info()
    print(f"{info['model']} | Python {info['python']} | mediapipe {info['mediapipe']} | opencv {info['opencv']}")
    print(f"{video}  (max width {a.max_width}, detect every {a.det_every} frame(s), model {a.det_model})")
    temp0, thr0 = cpu_temp(), throttled()

    roi, skin, occl = load_face_config()
    t_start = time.perf_counter()
    d = traces(video, roi, skin, occl, a.max_width, a.det_every, a.max_frames, not a.no_bg, a.det_model)
    wall = time.perf_counter() - t_start
    N = len(d["t"])
    if N == 0:
        sys.exit("error: no frames read")

    # HR: time each method on the whole trace; per-window cost = live cost per HR update
    methods = {"pos": lambda c: sanity_hr(d["t"], d["rgb_face"], WIN, STEP, centers=c)[0]}
    if not a.no_bg:
        methods["pos_bg"] = lambda c: sanity_hr_bg(d["t"], d["rgb_face"], d["rgb_bg"], WIN, STEP, centers=c)[0]
        methods["pos_snr_bg"] = lambda c: sanity_hr_snr(d["t"], d["regions"], WIN, STEP, centers=c,
                                                        rgb_bg=d["rgb_bg"])[0]
    if a.gt:
        tg, ppg = load_ubfc_gt(a.gt)
        hr_ref, tc = ppg_hr(tg, ppg, WIN, STEP)
    else:
        hr_ref, tc = None, np.arange(d["t"][0] + WIN / 2, d["t"][-1] - WIN / 2 + 1e-9, STEP)
    hr = {}
    for m, fn in methods.items():
        t0 = time.perf_counter()
        est = fn(tc)
        ms = 1000 * (time.perf_counter() - t0)
        row = dict(ms_total=ms, ms_per_window=ms / max(1, len(tc)), hr_median=float(np.nanmedian(est)))
        if hr_ref is not None:
            row.update(error_stats(est, hr_ref))
        hr[m] = row

    # report
    tmg = d["timing"]
    per_frame = sum(tmg.values())
    det_frames = tmg["detect"][::a.det_every]
    q = d["quality"]
    print(f"\n{N} frames {d['frame_size']} in {wall:.1f} s -> {N / wall:.1f} fps "
          f"(video {d['fps']:.1f} fps, {'real-time' if N / wall >= d['fps'] else 'SLOWER than real-time'})")
    print(f"face found {100 * (q > 0).mean():.1f} % of frames, detected {100 * (q == 2).mean():.1f} %")
    print("\nstage        mean ms   p95 ms")
    for s in STAGES:
        print(f"{s:<12} {tmg[s].mean():7.2f}  {np.percentile(tmg[s], 95):7.2f}")
    print(f"{'detect call':<12} {det_frames.mean():7.2f}  {np.percentile(det_frames, 95):7.2f}   (frames where it runs)")
    print(f"{'total':<12} {per_frame.mean():7.2f}  {np.percentile(per_frame, 95):7.2f}   (budget at 30 fps: 33.3)")
    print("note: the decoder works ahead in its own threads, so 'decode' looks small while 'detect'\n"
          "      runs every frame and grows when it does not. Compare runs by 'total' and fps.")
    print("\nmethod       HR ms/window" + ("   MAE   RMSE  SUC@5  windows" if hr_ref is not None else "   median HR"))
    for m, r in hr.items():
        tail = (f"  {r['mae']:5.2f}  {r['rmse']:5.2f}  {r['suc5_pct']:5.1f}  {r['n_windows']:5d}"
                if hr_ref is not None else f"   {r['hr_median']:.1f} bpm")
        print(f"{m:<12} {r['ms_per_window']:8.2f}    {tail}")
    if hr_ref is not None:
        print(f"reference    median {np.nanmedian(hr_ref):.1f} bpm")
    temps = [x for x in [temp0] + d["temps"] + [cpu_temp()] if x is not None]
    thr1 = throttled()
    if temps:
        print(f"\nCPU temperature: start {temps[0]:.1f} C, max {max(temps):.1f} C")
    if thr1 is not None:
        print(f"throttled: {thr0} -> {thr1}" + ("" if thr1 == "0x0" else "   (!) timings not valid, cool the Pi"))

    out = dict(name=name, video=str(video), device=info, args=vars(a) | {"video": str(video), "gt": str(a.gt)},
               frames=N, frame_size=d["frame_size"], video_fps=d["fps"], wall_s=wall, fps=N / wall,
               face_found_pct=100 * float((q > 0).mean()),
               stages_ms={s: dict(mean=float(v.mean()), p95=float(np.percentile(v, 95))) for s, v in tmg.items()},
               detect_call_ms=float(det_frames.mean()), total_ms=float(per_frame.mean()), hr=hr,
               temp_start_c=temps[0] if temps else None, temp_max_c=max(temps) if temps else None,
               throttled_start=thr0, throttled_end=thr1)
    od = REPO / "outputs" / "edge"
    od.mkdir(parents=True, exist_ok=True)
    (od / f"{name}.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"\nsaved outputs/edge/{name}.json")


if __name__ == "__main__":
    main()
