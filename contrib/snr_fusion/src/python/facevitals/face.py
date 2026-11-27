"""
Face box, ROI, skin mask, RGB traces and ITA.

tracked_face_traces(): port of src/face/process_video.m on top of tracker.FaceTracker
(RetinaFace + Kalman/Hungarian + KLT, One-Euro, occlusion hysteresis). This is the default
face module of run_c2.py and follows the MATLAB algorithm step by step.

face_traces(): the earlier simplified version, kept as --face-module simple. Python
counterpart of the MATLAB face module (src/face). Shared with it:
  - ROI boxes as face-box fractions from config/face_config.json
  - YCbCr skin mask with MATLAB rgb2ycbcr (studio-range) coefficients, same thresholds
  - rgb_face = pixel-weighted mean of the non-occluded regions
  - frame times read from the video, no frame dropped (NaN rows when no face)

Simplified compared with MATLAB (fine for single-subject datasets, documented in the README):
  - detector: MediaPipe full-range face detection instead of RetinaFace (R2025a+ only);
    the box is extended upwards because MediaPipe boxes stop near the eyebrows
  - largest face per frame, no Kalman/Hungarian multi-face tracking (C3) or KLT
  - box smoothed with an exponential filter instead of One-Euro
  - occlusion: absolute skin-ratio rule only (no hysteresis / relative rule)
"""

import json
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from skimage.color import rgb2lab

REPO = Path(__file__).resolve().parents[3]

# --------------------------------------------------------------- ITA classes

ITA_CLASSES = [  # (lower bound, name, approximate Fitzpatrick type)
    (55, "Very light", "I"),
    (41, "Light", "II"),
    (28, "Intermediate", "III"),
    (10, "Tan", "IV"),
    (-30, "Brown", "V"),
    (-np.inf, "Dark", "VI"),
]
ITA_GROUPS = [(41, "I-II"), (10, "III-IV"), (-np.inf, "V-VI")]  # merged groups for small N

# ITA quality gates
ITA_MIN_SKIN_FRACTION = 0.30
ITA_MIN_SKIN_PIXELS = 30
ITA_MIN_L = 25            # darker regions carry no colour information
ITA_MIN_GOOD = 3          # samples per region
ITA_MAX_SPREAD = 15       # degrees between regions -> "inconsistent"
ITA_HAIR_DL = 15          # region L* this far below the median region: hair / shadow, dropped
ITA_MIN_B = 3             # b* below this is not physical skin -> white-balance warning


def ita_class(ita):
    if not np.isfinite(ita):
        return "Unknown", ""
    for lo, name, fitz in ITA_CLASSES:
        if ita > lo:
            return name, fitz
    return "Dark", "VI"


def ita_group(ita):
    if not np.isfinite(ita):
        return "Unknown"
    for lo, name in ITA_GROUPS:
        if ita > lo:
            return name
    return "V-VI"


def ita_from_lb(L, b):
    return float(np.degrees(np.arctan2(L - 50, b)))


# --------------------------------------------------------------- config / skin

def load_face_config(path=None):
    path = Path(path) if path else REPO / "config" / "face_config.json"
    cfg = json.loads(path.read_text())
    return cfg["roi"], cfg["skin"], cfg.get("occl", {})


def skin_mask(rgb, skin):
    """src/face/skin_mask.m: MATLAB rgb2ycbcr (studio range) chroma box. rgb: HxWx3 uint8."""
    x = rgb.astype(np.float32) * (1.0 / 255.0)
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    y = 16 + 65.481 * r + 128.553 * g + 24.966 * b
    cb = 128 - 37.797 * r - 74.203 * g + 112.0 * b
    cr = 128 + 112.0 * r - 93.786 * g - 18.214 * b
    return ((cb >= skin["cb"][0]) & (cb <= skin["cb"][1])
            & (cr >= skin["cr"][0]) & (cr <= skin["cr"][1])
            & (y >= skin["yMin"]) & (rgb.max(axis=2) < skin["satMax"]))


def lab_median(pixels):
    """Median L*, b* of Kx3 uint8 RGB skin pixels (sRGB, D65 - same as MATLAB rgb2lab)."""
    lab = rgb2lab(pixels.reshape(-1, 1, 3).astype(np.float64) / 255.0).reshape(-1, 3)
    return float(np.median(lab[:, 0])), float(np.median(lab[:, 2]))


def region_lab(patch, mask):
    """ITA sample of one region in one frame, or None if it fails the quality gates."""
    n = int(mask.sum())
    if n < ITA_MIN_SKIN_PIXELS or n < ITA_MIN_SKIN_FRACTION * mask.size:
        return None
    L, b = lab_median(patch[mask])
    return (L, b) if L >= ITA_MIN_L else None


def summarize_ita(samples):
    """samples: {region: [(L, b), ...]} -> per-region rows and one per-video summary dict."""
    regions, use = [], []
    for r, vals in samples.items():
        ok = len(vals) >= ITA_MIN_GOOD
        L = float(np.median([v[0] for v in vals])) if ok else np.nan
        b = float(np.median([v[1] for v in vals])) if ok else np.nan
        regions.append(dict(region=r, good=len(vals), L=L, b=b,
                            ITA=ita_from_lb(L, b) if ok else np.nan))
        if ok:
            use.append(regions[-1])
    dropped = []
    if len(use) >= 2:
        Lmed = float(np.median([u["L"] for u in use]))
        dropped = [u["region"] for u in use if u["L"] < Lmed - ITA_HAIR_DL]
        use = [u for u in use if u["region"] not in dropped]
    for row in regions:
        row["dropped"] = row["region"] in dropped
    if use:
        L = float(np.median([u["L"] for u in use]))
        b = float(np.median([u["b"] for u in use]))
        ita = ita_from_lb(L, b)
        spread = float(max(u["ITA"] for u in use) - min(u["ITA"] for u in use))
    else:
        L = b = ita = spread = np.nan
    cls, fitz = ita_class(ita)
    summary = dict(ITA=ita, ITA_class=cls, fitz_approx=fitz, ITA_group=ita_group(ita), L=L, b=b,
                   ita_regions=",".join(u["region"] for u in use), ita_dropped=",".join(dropped),
                   ita_spread=spread, ita_consistent=bool(use) and spread <= ITA_MAX_SPREAD,
                   wb_warning=bool(np.isfinite(b) and b < ITA_MIN_B))
    return regions, summary


# --------------------------------------------------------------- face box / ROI

class FaceBox:
    """Largest face per frame (MediaPipe full-range model), box extended to cover the forehead."""

    def __init__(self, min_conf=0.5):
        self.det = mp.solutions.face_detection.FaceDetection(model_selection=1,
                                                             min_detection_confidence=min_conf)

    def __call__(self, rgb):
        res = self.det.process(rgb)
        if not res.detections:
            return None
        h, w = rgb.shape[:2]
        bb = max((d.location_data.relative_bounding_box for d in res.detections),
                 key=lambda b: b.width * b.height)
        x, y, bw, bh = bb.xmin * w, bb.ymin * h, bb.width * w, bb.height * h
        return np.array([x, y - 0.15 * bh, bw, 1.15 * bh])

    def close(self):
        self.det.close()


def roi_rects(box, roi, names):
    x, y, w, h = box
    return {n: (int(round(x + roi[n][0] * w)), int(round(y + roi[n][1] * h)),
                int(round(x + roi[n][2] * w)), int(round(y + roi[n][3] * h))) for n in names}


def crop(img, r):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = max(0, r[0]), max(0, r[1]), min(w, r[2]), min(h, r[3])
    return img[y1:y2, x1:x2] if x2 > x1 and y2 > y1 else img[0:0, 0:0]


# --------------------------------------------------------------- traces

def open_video(path, max_width):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)

    def frames():
        k = 0
        while True:
            t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0   # read before the frame (like MATLAB CurrentTime)
            ok, bgr = cap.read()
            if not ok:
                break
            h, w = bgr.shape[:2]
            if max_width and w > max_width:
                bgr = cv2.resize(bgr, (max_width, int(round(h * max_width / w))), interpolation=cv2.INTER_AREA)
            yield k, t, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            k += 1
        cap.release()

    return fps, frames()


def _fix_times(t, fps):
    """Container timestamps are sometimes missing or stuck; fall back to k/fps where they are."""
    t = np.asarray(t, dtype=np.float64)
    if len(t) > 1 and (np.any(np.diff(t) <= 0) or not np.all(np.isfinite(t))):
        if fps and fps > 0:
            t = np.arange(len(t)) / fps
    return t


BG_SCALE = 4         # background is averaged on a 1/4-size copy of the frame
BG_FACE_EXPAND = 1.6  # face box is enlarged by this factor (hair, ears) before it is excluded
BG_MIN_PIX = 200      # background pixels (on the small frame) needed for a valid sample


def background_rgb(rgb, box, skin):
    """
    Mean RGB of the scene outside the enlarged face box, without skin pixels (hands, neck
    and arms carry a pulse) and without clipped pixels. This is the illumination reference:
    it sees the same light flicker / exposure changes as the face, but no blood volume pulse.
    Returns (mean RGB (3,), pixel count).
    """
    h, w = rgb.shape[:2]
    small = cv2.resize(rgb, (max(1, w // BG_SCALE), max(1, h // BG_SCALE)), interpolation=cv2.INTER_AREA)
    sh, sw = small.shape[:2]
    keep = ~skin_mask(small, skin) & (small.max(axis=2) < skin["satMax"])
    if box is not None:
        x, y, bw, bh = np.asarray(box) / BG_SCALE
        cx, cy = x + bw / 2, y + bh / 2
        hw, hh = BG_FACE_EXPAND * bw / 2, BG_FACE_EXPAND * bh / 2
        x1, x2 = int(max(0, cx - hw)), int(min(sw, cx + hw))
        y1, y2 = int(max(0, cy - hh)), int(min(sh, cy + hh))
        keep[y1:y2, x1:x2] = False
    n = int(keep.sum())
    if n < BG_MIN_PIX:
        return np.full(3, np.nan), n
    return small[keep].mean(axis=0, dtype=np.float64), n


def face_traces(path, roi, skin, occl, max_width=960, ita_every=30, box_alpha=0.3,
                lost_after_s=1.0, max_frames=None):
    """
    Per-frame RGB traces of the largest face.

    Returns dict: t (N,), rgb_face (N,3), rgb_<region> (N,3), npix (N,R), skin_ratio (N,R),
    occluded (N,R), quality (N,) [0 no face, 1 held box, 2 detected], roi_names, fps,
    rgb_bg (N,3) background illumination reference, bg_npix (N,), face_box (N,4) [x y w h],
    ita_samples {region: [(L,b), ...]} (every ita_every frames).
    """
    names = list(roi["names"])
    R = len(names)
    on_below = occl.get("onBelow", 0.40)
    min_pix = occl.get("minPix", 10)
    fps, frames = open_video(path, max_width)
    det = FaceBox()

    T, Q, face, reg, npix, ratio, occ, bg, bgn, boxes = [], [], [], [], [], [], [], [], [], []
    ita = {n: [] for n in names}
    box, last_seen = None, -np.inf
    try:
        for k, t, rgb in frames:
            if max_frames and k >= max_frames:
                break
            T.append(t)
            b = det(rgb)
            if b is not None:
                box = b if box is None else (1 - box_alpha) * box + box_alpha * b
                last_seen, q = t, 2
            elif box is not None and t - last_seen <= lost_after_s:
                q = 1
            else:
                box, q = None, 0
            Q.append(q)
            b_rgb, b_n = background_rgb(rgb, box, skin)
            bg.append(b_rgb); bgn.append(b_n)
            boxes.append(np.full(4, np.nan) if box is None else np.asarray(box, float))
            fr, rr, nn, ra, oc = np.full(3, np.nan), np.full((R, 3), np.nan), np.zeros(R), np.zeros(R), np.ones(R, bool)
            if q > 0:
                sums, nall = np.zeros(3), 0
                for i, (n, rect) in enumerate(roi_rects(box, roi, names).items()):
                    patch = crop(rgb, rect)
                    if patch.size == 0:
                        continue
                    m = skin_mask(patch, skin)
                    ns = int(m.sum())
                    nn[i], ra[i] = ns, ns / m.size
                    if ns > 0:
                        s = patch[m].sum(axis=0, dtype=np.float64)
                        rr[i] = s / ns
                    oc[i] = ra[i] < on_below or ns < min_pix
                    if not oc[i]:
                        sums += s
                        nall += ns
                    if k % ita_every == 0 and q == 2:
                        v = region_lab(patch, m)
                        if v is not None:
                            ita[n].append(v)
                if nall > 0:
                    fr = sums / nall
            face.append(fr); reg.append(rr); npix.append(nn); ratio.append(ra); occ.append(oc)
    finally:
        det.close()

    reg = np.array(reg).reshape(-1, R, 3)
    out = dict(t=_fix_times(T, fps), rgb_face=np.array(face).reshape(-1, 3), npix=np.array(npix),
               skin_ratio=np.array(ratio), occluded=np.array(occ), quality=np.array(Q),
               roi_names=np.array(names), fps=fps, ita_samples=ita,
               rgb_bg=np.array(bg).reshape(-1, 3), bg_npix=np.array(bgn), face_box=np.array(boxes).reshape(-1, 4))
    for i, n in enumerate(names):
        out["rgb_" + n] = reg[:, i, :]
    return out


def roi_skin_stats(I, poly, skin):
    """roi_skin_stats.m: skin pixels inside one ROI polygon (Kx2 [x y], 0-based).
    Returns (rgb_sum (3,), n_skin, n_poly, patch, mask) - patch/mask of the bounding rectangle."""
    from skimage.draw import polygon2mask
    H, W = I.shape[:2]
    if poly is None or not np.all(np.isfinite(poly)):
        return np.zeros(3), 0, 0, None, None
    x1 = max(0, int(np.floor(poly[:, 0].min()))); x2 = min(W - 1, int(np.ceil(poly[:, 0].max())))
    y1 = max(0, int(np.floor(poly[:, 1].min()))); y2 = min(H - 1, int(np.ceil(poly[:, 1].max())))
    if x2 <= x1 or y2 <= y1:
        return np.zeros(3), 0, 0, None, None
    in_poly = polygon2mask((y2 - y1 + 1, x2 - x1 + 1), np.column_stack([poly[:, 1] - y1, poly[:, 0] - x1]))
    patch = I[y1:y2 + 1, x1:x2 + 1]
    M = in_poly & skin_mask(patch, skin)
    n_skin = int(M.sum())
    s = patch[M].sum(axis=0, dtype=np.float64) if n_skin else np.zeros(3)
    return s, n_skin, int(in_poly.sum()), patch, M


def tracked_face_traces(path, p=None, detector=None, ita_every=30, max_frames=None, max_width=None,
                        verbose=False):
    """
    Port of src/face/process_video.m on top of tracker.FaceTracker (RetinaFace, Kalman +
    Hungarian, KLT + similarity transform), with the same per-frame ROI steps:
    roi_polygons -> One-Euro smoothing of all polygon corners and the face box ->
    skin pixels per polygon -> occlusion (absolute AND relative rule, both with hysteresis on
    the EMA-smoothed skin ratio, or n_skin < minPix) -> rgb_face over non-occluded regions.
    Frames with quality 0 are NaN and reset the smoother and occlusion states.

    Extra outputs used by the C2 evaluation (not in MATLAB): rgb_bg / bg_npix (background
    illumination reference) and ita_samples (L*, b* of each region's skin every ita_every frames).
    max_width=None processes the native resolution, as MATLAB does.
    """
    from .retinaface import RetinaFaceDetector
    from .tracker import FaceTracker, OneEuroFilter, face_params, roi_polygons

    p = p or face_params()
    skin = p["skin"]
    names = list(p["roi"]["names"])
    nR = len(names)
    occl = p["occl"]
    ref_idx = names.index(occl["reference"]) if occl["reference"] in names else 0
    use_rel = occl["reference"] in names
    det = detector or RetinaFaceDetector(p["det"]["threshold"], p["det"]["minSize"])

    fps, frames = open_video(path, max_width)
    tracker = smoother = None
    ratio_ema = np.full(nR, np.nan)
    abs_state = np.zeros(nR, bool)
    rel_state = np.zeros(nR, bool)

    T, Q, NF, TID, face, reg, npix, ratio, occ, bg, bgn, boxes = ([] for _ in range(12))
    ita = {n: [] for n in names}
    for k, t, rgb in frames:
        if max_frames and k >= max_frames:
            break
        if tracker is None:
            tracker = FaceTracker(det, p, rgb.shape[:2])
            smoother = OneEuroFilter(p["smooth"]["minCutoff"], p["smooth"]["beta"], p["smooth"]["dCutoff"])
        T.append(t)
        st = tracker.step(rgb, t)
        Q.append(st["quality"]); NF.append(st["n_faces"]); TID.append(st["target_id"])
        fr, rr = np.full(3, np.nan), np.full((nR, 3), np.nan)
        nn, ra, oc = np.zeros(nR), np.full(nR, np.nan), np.zeros(nR, bool)
        fbox = None
        if st["quality"] == 0:
            smoother.reset()
            ratio_ema[:] = np.nan; abs_state[:] = False; rel_state[:] = False
        else:
            R = roi_polygons(st["anchor_box"], st["A"], p["roi"])
            vec = np.concatenate([R[n].T.ravel() for n in names] + [R["box"]])   # MATLAB column-major P(:)
            if p["smooth"]["enable"]:
                vec = smoother.update(vec, t)
            fbox = vec[-4:]
            sums, nS = np.zeros((nR, 3)), np.zeros(nR)
            for r, n in enumerate(names):
                P = vec[8 * r:8 * r + 8].reshape(2, 4).T
                s, ns, npoly, patch, M = roi_skin_stats(rgb, P, skin)
                sums[r], nS[r] = s, ns
                ra[r] = ns / max(npoly, 1)
                ratio_ema[r] = ra[r] if np.isnan(ratio_ema[r]) else \
                    (1 - occl["emaAlpha"]) * ratio_ema[r] + occl["emaAlpha"] * ra[r]
                if k % ita_every == 0 and M is not None:
                    v = region_lab(patch, M)
                    if v is not None:
                        ita[n].append(v)
            s_all, n_all = np.zeros(3), 0
            for r in range(nR):
                abs_state[r] = ratio_ema[r] < (occl["offAbove"] if abs_state[r] else occl["onBelow"])
                if not use_rel or r == ref_idx:
                    rel_state[r] = True
                else:
                    rel = ratio_ema[r] / max(ratio_ema[ref_idx], np.finfo(float).eps)
                    rel_state[r] = rel < (occl["relOff"] if rel_state[r] else occl["relOn"])
                oc[r] = (abs_state[r] and rel_state[r]) or nS[r] < occl["minPix"]
                nn[r] = nS[r]
                if nS[r] > 0:
                    rr[r] = sums[r] / nS[r]
                if not oc[r]:
                    s_all += sums[r]; n_all += nS[r]
            if n_all > 0:
                fr = s_all / n_all
        b_rgb, b_n = background_rgb(rgb, fbox, skin)
        face.append(fr); reg.append(rr); npix.append(nn); ratio.append(ra); occ.append(oc)
        bg.append(b_rgb); bgn.append(b_n); boxes.append(np.full(4, np.nan) if fbox is None else fbox)
        if verbose and k % 300 == 0:
            print(f"   frame {k}  q={st['quality']}  faces={st['n_faces']}", flush=True)

    reg = np.array(reg).reshape(-1, nR, 3)
    out = dict(t=_fix_times(T, fps), rgb_face=np.array(face).reshape(-1, 3), npix=np.array(npix),
               skin_ratio=np.array(ratio), occluded=np.array(occ), quality=np.array(Q),
               n_faces=np.array(NF), target_id=np.array(TID, float), roi_names=np.array(names), fps=fps,
               ita_samples=ita, rgb_bg=np.array(bg).reshape(-1, 3), bg_npix=np.array(bgn),
               face_box=np.array(boxes).reshape(-1, 4),
               n_detector_calls=tracker.num_detections if tracker else 0)
    for i, n in enumerate(names):
        out["rgb_" + n] = reg[:, i, :]
    return out


def crop_traces(paths, skin, ita_every=30):
    """
    AiPEX-style pre-cropped ROI clips: skin mask on the whole crop, no face detection.
    paths: {region: video}. rgb_face = pixel-weighted mean over the regions.
    """
    per = {}
    ita = {r: [] for r in paths}
    fps = None
    for r, p in paths.items():
        fps, frames = open_video(p, None)
        T, S, N = [], [], []
        for k, t, rgb in frames:
            m = skin_mask(rgb, skin)
            ns = int(m.sum())
            T.append(t)
            S.append(rgb[m].sum(axis=0, dtype=np.float64) if ns else np.zeros(3))
            N.append(ns)
            if k % ita_every == 0:
                v = region_lab(rgb, m)
                if v is not None:
                    ita[r].append(v)
        per[r] = (_fix_times(T, fps), np.array(S), np.array(N))
    n = min(len(v[0]) for v in per.values())
    t = next(iter(per.values()))[0][:n]
    sums = sum(v[1][:n] for v in per.values())
    cnt = sum(v[2][:n] for v in per.values())
    with np.errstate(invalid="ignore", divide="ignore"):
        face = np.where(cnt[:, None] > 0, sums / cnt[:, None], np.nan)
    out = dict(t=t, rgb_face=face, quality=(cnt > 0).astype(int) * 2, roi_names=np.array(list(paths)),
               fps=fps, ita_samples=ita)
    for r, (_, s, c) in per.items():
        with np.errstate(invalid="ignore", divide="ignore"):
            out["rgb_" + r] = np.where(c[:n, None] > 0, s[:n] / c[:n, None], np.nan)
    return out
