"""
One command for the live HR test on the Pi (or a laptop).

  python3 deploy/edge/live_test.py                 # USB webcam if connected, else the synthetic clip
  python3 deploy/edge/live_test.py --clip          # always the synthetic clip (looped, with reference HR)
  python3 deploy/edge/live_test.py --camera 0      # this camera index
  python3 deploy/edge/live_test.py --headless --seconds 60     # no window, print + CSV log, stop after 60 s

Any other option goes to run_edge.py (e.g. --det-every 5, --fullscreen, --method pos_snr_bg, --lock).
The camera runs with auto exposure and auto white balance unless --lock is given.

What it does:
  1. re-runs itself with deploy/edge/.venv/bin/python if that venv exists (no need to activate it)
  2. sets DISPLAY=:0 when started over SSH, so the window opens on the Pi's screen
  3. picks the source: a USB webcam (the Pi's own codec/ISP video devices are skipped), else the
     synthetic clip (made with make_test_clip.py if it is missing)
  4. prints the device, CPU temperature and throttle state, then runs run_edge.py
  5. writes outputs/edge/live_<time>.csv (one row per HR update)
Window keys: q / Esc quit, m next method, r reset the buffer.
"""

import argparse
import datetime
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
VENV_PY = HERE / ".venv" / "bin" / "python"
CLIP = REPO / "outputs" / "edge" / "synthetic"
NOT_CAMERAS = ("bcm2835", "rpi-", "pispbe", "rp1-cfe")   # Raspberry Pi codec / ISP nodes, not cameras


def use_venv():
    if VENV_PY.exists() and Path(sys.prefix).resolve() != VENV_PY.parent.parent.resolve():
        os.execv(str(VENV_PY), [str(VENV_PY), str(Path(__file__).resolve()), *sys.argv[1:]])


def find_camera():
    """(index, name) of the best capture device, or None. A plugged-in USB webcam comes first, then a
    built-in USB camera, then anything else (virtual cameras such as Iriun / OBS)."""
    found = []
    for d in Path("/sys/class/video4linux").glob("video*"):
        try:
            name = (d / "name").read_text().strip()
            index = (d / "index").read_text().strip()
        except OSError:
            continue
        if name.startswith(NOT_CAMERAS) or index != "0":       # index 0 = the device's capture node
            continue
        try:                                                    # USB port: "removable" (external) / "fixed"
            removable = (d / "device" / ".." / "removable").resolve().read_text().strip()
        except OSError:
            removable = ""
        rank = {"removable": 0, "fixed": 1}.get(removable, 2)
        found.append((rank, int(d.name[5:]), name))
    if not found:
        return None
    for rank, i, name in sorted(found):
        print(f"  camera {i}: {name} ({['USB webcam', 'built-in', 'other / virtual'][rank]})")
    _, i, name = min(found)
    return i, name


def run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=3).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def main():
    use_venv()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", action="store_true", help="use the synthetic test clip even if a webcam is connected")
    ap.add_argument("--camera", type=int, help="camera index (default: first USB webcam found)")
    ap.add_argument("--headless", action="store_true", help="no window: print one line per HR update")
    ap.add_argument("--no-log", action="store_true", help="do not write the CSV log")
    a, rest = ap.parse_known_args()

    if not a.headless and not os.environ.get("DISPLAY") and Path("/tmp/.X11-unix/X0").exists():
        os.environ["DISPLAY"] = ":0"                            # started over SSH: use the Pi's screen
        os.environ.setdefault("XAUTHORITY", str(Path.home() / ".Xauthority"))
    if not a.headless and not os.environ.get("DISPLAY"):
        print("no display found: running headless")
        a.headless = True

    model = run(["cat", "/proc/device-tree/model"]).strip("\x00") or run(["uname", "-m"])
    print(f"device: {model} | python {sys.version.split()[0]} ({sys.executable})")
    temp, thr = run(["vcgencmd", "measure_temp"]), run(["vcgencmd", "get_throttled"])
    if temp:
        print(f"{temp}  {thr}" + ("" if thr.endswith("0x0") else "   (throttled since boot: cool the Pi)"))

    args = []
    cam = (a.camera, f"/dev/video{a.camera}") if a.camera is not None else (None if a.clip else find_camera())
    if cam is not None:
        print(f"source: camera {cam[0]} ({cam[1]})")
        args += ["--camera", str(cam[0])]
    else:
        print("source: synthetic test clip, looped" + ("" if a.clip else " (no USB webcam found)"))
        if not (CLIP / "vid.avi").exists():
            print("making the test clip (about 1 min)...")
            subprocess.run([sys.executable, str(HERE / "make_test_clip.py")], check=True)
        args += ["--video", str(CLIP / "vid.avi"), "--gt", str(CLIP / "ground_truth.txt"), "--loop"]
    if a.headless:
        args.append("--headless")
    if not a.no_log:
        log = REPO / "outputs" / "edge" / f"live_{datetime.datetime.now():%Y%m%d_%H%M%S}.csv"
        args += ["--log", str(log)]
        print(f"log: {log.relative_to(REPO)}")
    args += rest

    sys.path.insert(0, str(HERE))
    import run_edge                                             # noqa: E402  (after the venv switch)
    sys.argv = [str(HERE / "run_edge.py"), *args]
    run_edge.main()


if __name__ == "__main__":
    main()
