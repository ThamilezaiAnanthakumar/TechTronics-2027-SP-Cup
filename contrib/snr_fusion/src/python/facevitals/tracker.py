"""
Multi-face tracker with one designated target: line-by-line port of src/face/FaceTracker.m
and its helpers (face_params.m, OneEuroFilter.m, roi_polygons.m, apply_tform.m, box_iou.m,
clip_box.m). Same algorithm and parameters as MATLAB:

  - RetinaFace every det.every frames (det.everyLost while the target is lost)
  - one constant-velocity Kalman filter per face (MATLAB configureKalmanFilter settings)
  - Hungarian assignment on 1 - IoU with MATLAB's assignDetectionsToTracks padding
  - target chosen once (largest-central or trk.targetBox), followed every frame by KLT points
    and a similarity transform (RANSAC), re-anchored on the detection when IoU < klt.reanchorIoU
  - on KLT failure: detection in a window around the last position; target lost otherwise,
    re-acquired only within trk.reacqDist face widths (any face after trk.reacqAnyAfter s)

Library equivalents: detectMinEigenFeatures -> cv2.goodFeaturesToTrack (min-eigenvalue),
vision.PointTracker -> cv2.calcOpticalFlowPyrLK with forward-backward check,
estgeotform2d("similarity") -> cv2.estimateAffinePartial2D (RANSAC),
assignDetectionsToTracks -> scipy linear_sum_assignment on the padded cost matrix.
Coordinates are 0-based (MATLAB: 1-based); boxes are [x y w h].
"""

import copy
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

REPO = Path(__file__).resolve().parents[3]


# --------------------------------------------------------------- parameters (face_params.m)

DEFAULTS = {
    "version": "1.2",
    "det": {"model": "small-network", "device": "cpu", "threshold": 0.5, "minSize": [10, 10],
            "every": 10, "everyLost": 3, "searchScale": 2.0, "upscale": 1},
    "trk": {"targetPolicy": "largest-central", "centralWeight": 0.5, "targetBox": [],
            "minIoU": 0.10, "costNonAssign": 0.50, "maxMisses": 3, "reacqDist": 1.0,
            "reacqAnyAfter": 5, "kfInitErr": [200, 50], "kfMotionNoise": [100, 25], "kfMeasNoise": 100},
    "klt": {"maxPts": 300, "minPts": 8, "minQuality": 0.01, "featInset": [0.15, 0.05],
            "maxBidirErr": 2, "maxDist": 3, "refreshFrac": 0.5, "reanchorIoU": 0.75},
    "roi": {"names": ["forehead", "lcheek", "rcheek"],
            "forehead": [0.32, 0.12, 0.68, 0.27], "lcheek": [0.15, 0.50, 0.38, 0.72],
            "rcheek": [0.62, 0.50, 0.85, 0.72], "glabella": [0.42, 0.28, 0.58, 0.40],
            "nose": [0.44, 0.40, 0.56, 0.58], "lmalar": [0.15, 0.46, 0.38, 0.60],
            "rmalar": [0.62, 0.46, 0.85, 0.60]},
    "skin": {"cb": [77, 135], "cr": [128, 173], "yMin": 20, "satMax": 250},
    "occl": {"reference": "forehead", "onBelow": 0.40, "offAbove": 0.50, "relOn": 0.60,
             "relOff": 0.70, "emaAlpha": 0.15, "minPix": 10},
    "smooth": {"enable": True, "minCutoff": 1.0, "beta": 0.02, "dCutoff": 1.0},
}


def _merge(a, b):
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            _merge(a[k], v)
        else:
            a[k] = v
    return a


def face_params(config_path=None):
    """Defaults of face_params.m overridden by config/face_config.json (same merge rule)."""
    p = copy.deepcopy(DEFAULTS)
    path = Path(config_path) if config_path else REPO / "config" / "face_config.json"
    if path.is_file():
        _merge(p, json.loads(path.read_text()))
    for n in p["roi"]["names"]:
        if n not in p["roi"] or len(p["roi"][n]) != 4:
            raise ValueError(f'Region "{n}" needs a [u1 v1 u2 v2] entry in roi.')
    return p


# --------------------------------------------------------------- geometry helpers

def box_center(b):
    b = np.atleast_2d(b)
    return np.column_stack([b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2])


def box_iou(a, B):
    """IoU of box a with boxes B (Mx4); NaN boxes give 0 (box_iou.m)."""
    B = np.atleast_2d(np.asarray(B, float))
    if B.size == 0:
        return np.zeros(0)
    x1 = np.maximum(a[0], B[:, 0]); y1 = np.maximum(a[1], B[:, 1])
    x2 = np.minimum(a[0] + a[2], B[:, 0] + B[:, 2]); y2 = np.minimum(a[1] + a[3], B[:, 1] + B[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    with np.errstate(invalid="ignore", divide="ignore"):
        iou = inter / (a[2] * a[3] + B[:, 2] * B[:, 3] - inter)
    iou[~np.isfinite(iou)] = 0
    return iou


def clip_box(b, frame_size, as_integer=False):
    """clip_box.m in 0-based coordinates: x in [0, W], y in [0, H]."""
    b = np.atleast_2d(np.asarray(b, float))
    H, W = frame_size
    x1, y1, x2, y2 = b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    if as_integer:
        x1, y1, x2, y2 = np.round(x1), np.round(y1), np.round(x2), np.round(y2)
    x1 = np.maximum(0, x1); y1 = np.maximum(0, y1); x2 = np.minimum(W, x2); y2 = np.minimum(H, y2)
    out = np.column_stack([x1, y1, np.maximum(0, x2 - x1), np.maximum(0, y2 - y1)])
    return out[0] if out.shape[0] == 1 else out


def apply_tform(A, P):
    """[x';y';1] = A*[x;y;1] for points P (Kx2) (apply_tform.m)."""
    P = np.asarray(P, float)
    return (A[:2, :2] @ P.T + A[:2, 2:3]).T


def roi_polygons(anchor_box, A, roi):
    """roi_polygons.m: one 4x2 polygon per region moved by A, plus the moved face box."""
    x, y, w, h = anchor_box
    R = {}
    for name in roi["names"]:
        r = roi[name]
        P = np.array([[x + r[0] * w, y + r[1] * h], [x + r[2] * w, y + r[1] * h],
                      [x + r[2] * w, y + r[3] * h], [x + r[0] * w, y + r[3] * h]])
        R[name] = apply_tform(A, P)
    C = apply_tform(A, np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]]))
    R["box"] = np.array([C[:, 0].min(), C[:, 1].min(), np.ptp(C[:, 0]), np.ptp(C[:, 1])])
    return R


def expand_box(b, s, frame_size):
    c = box_center(b)[0]
    return clip_box([c[0] - s * b[2] / 2, c[1] - s * b[3] / 2, s * b[2], s * b[3]], frame_size, True)


def pick_near(B, ref, max_dist):
    """Index of the box in B closest to ref (< max_dist face widths, similar size); -1 if none."""
    B = np.atleast_2d(np.asarray(B, float))
    if B.size == 0 or ref is None or np.any(np.isnan(ref)):
        return -1
    d = np.linalg.norm(box_center(B) - box_center(ref), axis=1) / ref[2]
    ratio = B[:, 2] * B[:, 3] / (ref[2] * ref[3])
    d[(ratio < 0.5) | (ratio > 2)] = np.inf
    k = int(np.argmin(d))
    return k if d[k] < max_dist else -1


# --------------------------------------------------------------- Kalman filter

class KalmanCV:
    """MATLAB configureKalmanFilter("ConstantVelocity", loc, InitialEstimateError,
    MotionNoise, MeasurementNoise) in 2-D: state [x vx y vy], unit time step."""

    def __init__(self, loc, init_err, motion_noise, meas_noise):
        self.A = np.array([[1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]], float)
        self.H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]], float)
        self.x = np.array([loc[0], 0, loc[1], 0], float)
        self.P = np.diag([init_err[0], init_err[1], init_err[0], init_err[1]]).astype(float)
        self.Q = np.diag([motion_noise[0], motion_noise[1], motion_noise[0], motion_noise[1]]).astype(float)
        self.R = float(meas_noise) * np.eye(2)

    def predict(self):
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q
        return self.H @ self.x

    def correct(self, z):
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (np.asarray(z, float) - self.H @ self.x)
        self.P = (np.eye(4) - K @ self.H) @ self.P
        return self.H @ self.x


def assign_detections_to_tracks(cost, cost_non_assign):
    """MATLAB assignDetectionsToTracks: Munkres on the cost matrix padded with dummy rows /
    columns whose diagonal costs cost_non_assign. Returns (pairs Kx2, unassigned tracks,
    unassigned detections)."""
    nT, nD = cost.shape
    BIG = 1e12
    M = np.full((nT + nD, nD + nT), BIG)
    M[:nT, :nD] = cost
    M[:nT, nD:] = np.where(np.eye(nT, dtype=bool), cost_non_assign, BIG)
    M[nT:, :nD] = np.where(np.eye(nD, dtype=bool), cost_non_assign, BIG)
    M[nT:, nD:] = 0
    r, c = linear_sum_assignment(M)
    pairs = np.array([(i, j) for i, j in zip(r, c) if i < nT and j < nD], dtype=int).reshape(-1, 2)
    uT = np.array([i for i in range(nT) if i not in set(pairs[:, 0])], dtype=int)
    uD = np.array([j for j in range(nD) if j not in set(pairs[:, 1])], dtype=int)
    return pairs, uT, uD


# --------------------------------------------------------------- One-Euro filter

class OneEuroFilter:
    """OneEuroFilter.m (Casiez et al., CHI 2012)."""

    def __init__(self, min_cutoff, beta, d_cutoff):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.reset()

    @staticmethod
    def _alpha(fc, dt):
        tau = 1.0 / (2 * np.pi * fc)
        return 1.0 / (1.0 + tau / dt)

    def update(self, x, t):
        x = np.asarray(x, float)
        if self.x_prev is None or self.x_prev.shape != x.shape:
            self.x_prev, self.dx_prev, self.t_prev = x.copy(), np.zeros_like(x), t
            return x
        dt = t - self.t_prev
        if not dt > 0:
            dt = 1 / 30
        dx = (x - self.x_prev) / dt
        dx_hat = self._alpha(self.d_cutoff, dt) * dx + (1 - self._alpha(self.d_cutoff, dt)) * self.dx_prev
        a = self._alpha(self.min_cutoff + self.beta * np.abs(dx_hat), dt)
        y = a * x + (1 - a) * self.x_prev
        self.x_prev, self.dx_prev, self.t_prev = y, dx_hat, t
        return y

    def reset(self):
        self.x_prev = self.dx_prev = self.t_prev = None


# --------------------------------------------------------------- tracker

class _Track:
    def __init__(self, tid, box, p):
        self.id = tid
        self.box = np.asarray(box, float)
        self.det = np.asarray(box, float)
        self.kf = KalmanCV(box_center(box)[0], p["trk"]["kfInitErr"], p["trk"]["kfMotionNoise"],
                           p["trk"]["kfMeasNoise"])
        self.misses = 0
        self.hits = 1


class FaceTracker:
    """FaceTracker.m. step(rgb, t) once per frame, in order -> state dict."""

    def __init__(self, detector, p, frame_size):
        self.p, self.det, self.frame_size = p, detector, tuple(frame_size[:2])
        self.tracks = []
        self.next_id = 1
        self.target_id = None
        self.last_target_box = None
        self.lost = True
        self.lost_since = -np.inf
        self.since_det = np.inf
        self.prev_gray = None
        self.pts = np.zeros((0, 2), np.float32)
        self.num_pts_init = 0
        self.anchor_box = np.full(4, np.nan)
        self.A = np.eye(3)
        self.num_detections = 0

    # ---- public
    def step(self, rgb, t):
        p = self.p
        G = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        confirmed = detected = False

        if not self.lost and not self._klt_update(G):
            win = expand_box(self.last_target_box, p["det"]["searchScale"], self.frame_size)
            B = self._run_detector(rgb, win)
            detected = True
            j = pick_near(B, self.last_target_box, p["trk"]["reacqDist"])
            if j >= 0:
                self._anchor(G, B[j])
                self._set_target_track_box(B[j], correct_kalman=True)
                confirmed = True
            else:
                self._lose_target(t)
        if not self.lost:
            self._set_target_track_box(self._current_target_box(), correct_kalman=False)

        period = p["det"]["everyLost"] if self.lost else p["det"]["every"]
        self.since_det += 1
        if self.since_det >= period:
            self.since_det = 0
            B = self._run_detector(rgb, None)
            detected = True
            self._update_tracks(B)
            ti = self._target_index()
            if ti is not None and self.tracks[ti].misses == 0:
                db = self.tracks[ti].det
                if self.lost or box_iou(self._current_target_box(), db[None])[0] < p["klt"]["reanchorIoU"]:
                    self._anchor(G, db)
                confirmed = True
            elif self.lost:
                j = self._choose_target(t)
                if j is not None:
                    self.target_id = self.tracks[j].id
                    self._anchor(G, self.tracks[j].det)
                    confirmed = True

        self.prev_gray = G
        st = {"detected": detected, "num_pts": len(self.pts)}
        if self.lost:
            st.update(quality=0, target_id=np.nan, target_box=np.full(4, np.nan),
                      anchor_box=np.full(4, np.nan), A=np.eye(3))
        else:
            st.update(quality=1 + int(confirmed), target_id=self.target_id,
                      target_box=self._current_target_box(), anchor_box=self.anchor_box.copy(), A=self.A.copy())
        st["boxes"], st["ids"] = self._visible_faces()
        st["n_faces"] = len(st["ids"])
        return st

    # ---- detection and tracks
    def _run_detector(self, rgb, roi):
        self.num_detections += 1
        if roi is not None and (roi[2] < 16 or roi[3] < 16):
            return np.zeros((0, 4))
        B, _ = self.det.detect(rgb, roi)
        if len(B) == 0:
            return np.zeros((0, 4))
        B = clip_box(B, self.frame_size)
        B = np.atleast_2d(B)
        return B[(B[:, 2] >= 2) & (B[:, 3] >= 2)]

    def _update_tracks(self, B):
        p = self.p
        nT, nD = len(self.tracks), len(B)
        ti = self._target_index()
        for i, tr in enumerate(self.tracks):
            c = tr.kf.predict()
            if not (i == ti and not self.lost):          # tracked target keeps its KLT box
                tr.box = np.array([c[0] - tr.box[2] / 2, c[1] - tr.box[3] / 2, tr.box[2], tr.box[3]])
        if nT > 0 and nD > 0:
            gate = 1e3
            cost = np.zeros((nT, nD))
            for i, tr in enumerate(self.tracks):
                iou = box_iou(tr.box, B)
                c = 1 - iou
                c[iou < p["trk"]["minIoU"]] = gate
                cost[i] = c
            asg, uT, uD = assign_detections_to_tracks(cost, p["trk"]["costNonAssign"])
            if len(asg):
                bad = cost[asg[:, 0], asg[:, 1]] >= gate
                uT = np.concatenate([uT, asg[bad, 0]])
                uD = np.concatenate([uD, asg[bad, 1]])
                asg = asg[~bad]
        else:
            asg, uT, uD = np.zeros((0, 2), int), np.arange(nT), np.arange(nD)

        for i, j in asg:
            tr = self.tracks[i]
            tr.kf.correct(box_center(B[j])[0])
            tr.box = B[j].copy(); tr.det = B[j].copy(); tr.misses = 0; tr.hits += 1
        for i in uT:
            self.tracks[i].misses += 1
            self.tracks[i].det = np.full(4, np.nan)

        keep = []
        for i, tr in enumerate(self.tracks):
            is_live_target = (i == ti and not self.lost)
            if tr.misses <= p["trk"]["maxMisses"] or is_live_target:
                keep.append(tr)
        self.tracks = keep
        for j in uD:
            self.tracks.append(_Track(self.next_id, B[j], p))
            self.next_id += 1

    def _target_index(self):
        if not self.tracks or self.target_id is None:
            return None
        for i, tr in enumerate(self.tracks):
            if tr.id == self.target_id:
                return i
        return None

    def _set_target_track_box(self, box, correct_kalman):
        ti = self._target_index()
        if ti is None:
            return
        self.tracks[ti].box = np.asarray(box, float)
        if correct_kalman:
            self.tracks[ti].kf.correct(box_center(box)[0])
            self.tracks[ti].misses = 0

    def _choose_target(self, t):
        p = self.p
        vis = [i for i, tr in enumerate(self.tracks) if tr.misses == 0]
        if not vis:
            return None
        boxes = np.array([self.tracks[i].det for i in vis])
        if (t - self.lost_since) <= p["trk"]["reacqAnyAfter"] and self.last_target_box is not None:
            k = pick_near(boxes, self.last_target_box, p["trk"]["reacqDist"])
            return vis[k] if k >= 0 else None
        if self.target_id is None and len(p["trk"].get("targetBox") or []) == 4:
            iou = box_iou(np.asarray(p["trk"]["targetBox"], float), boxes)
            k = int(np.argmax(iou))
            return vis[k] if iou[k] > 0 else None
        area = boxes[:, 2] * boxes[:, 3]
        imc = np.array([self.frame_size[1], self.frame_size[0]]) / 2
        dist = np.linalg.norm(box_center(boxes) - imc, axis=1) / np.linalg.norm(imc)
        return vis[int(np.argmax(area / area.max() - p["trk"]["centralWeight"] * dist))]

    def _visible_faces(self):
        boxes, ids = [], []
        ti = self._target_index()
        for i, tr in enumerate(self.tracks):
            is_tgt = (i == ti and not self.lost)
            if tr.misses == 0 or is_tgt:
                boxes.append(self._current_target_box() if is_tgt else tr.box)
                ids.append(tr.id)
        return np.array(boxes).reshape(-1, 4), np.array(ids)

    # ---- target anchoring and KLT
    def _anchor(self, G, box):
        box = clip_box(box, self.frame_size)
        self.anchor_box = np.asarray(box, float)
        self.A = np.eye(3)
        self.lost = False
        self.last_target_box = self.anchor_box.copy()
        self._init_points(G, self.anchor_box)

    def _init_points(self, G, box):
        p = self.p
        ins = p["klt"]["featInset"]
        r = clip_box([box[0] + ins[0] * box[2], box[1] + ins[1] * box[3],
                      box[2] * (1 - 2 * ins[0]), box[3] * (1 - 2 * ins[1])], self.frame_size, True)
        loc = np.zeros((0, 2), np.float32)
        if r[2] >= 4 and r[3] >= 4:
            mask = np.zeros(G.shape, np.uint8)
            x, y, w, h = (int(v) for v in r)
            mask[y:y + h, x:x + w] = 255
            c = cv2.goodFeaturesToTrack(G, maxCorners=int(p["klt"]["maxPts"]), qualityLevel=p["klt"]["minQuality"],
                                        minDistance=1, mask=mask, blockSize=5, useHarrisDetector=False)
            if c is not None:
                loc = c.reshape(-1, 2).astype(np.float32)
        self.pts = loc
        self.num_pts_init = len(loc)
        self.prev_gray = G

    def _klt_update(self, G):
        p = self.p
        if len(self.pts) < p["klt"]["minPts"] or self.prev_gray is None:
            return False
        lk = dict(winSize=(31, 31), maxLevel=2,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        p0 = self.pts.reshape(-1, 1, 2)
        p1, s1, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, G, p0, None, **lk)
        p0b, s2, _ = cv2.calcOpticalFlowPyrLK(G, self.prev_gray, p1, None, **lk)
        bidir = np.linalg.norm(p0b.reshape(-1, 2) - self.pts, axis=1)
        valid = (s1.ravel() == 1) & (s2.ravel() == 1) & (bidir <= p["klt"]["maxBidirErr"])
        old, new = self.pts[valid], p1.reshape(-1, 2)[valid]
        if len(new) < p["klt"]["minPts"]:
            return False
        M, inl = cv2.estimateAffinePartial2D(old, new, method=cv2.RANSAC,
                                             ransacReprojThreshold=p["klt"]["maxDist"],
                                             maxIters=1000, confidence=0.99)
        if M is None or inl is None or int(inl.sum()) < p["klt"]["minPts"]:
            return False
        T = np.vstack([M, [0, 0, 1]])
        Anew = T @ self.A
        s = np.sqrt(abs(np.linalg.det(Anew[:2, :2])))
        if s < 0.5 or s > 2:
            return False
        self.A = Anew
        self.pts = new[inl.ravel() == 1].astype(np.float32)
        self.last_target_box = self._current_target_box()
        if len(self.pts) < p["klt"]["refreshFrac"] * self.num_pts_init:
            self._init_points(G, self.last_target_box)
        return True

    def _current_target_box(self):
        b = self.anchor_box
        P = apply_tform(self.A, np.array([[b[0], b[1]], [b[0] + b[2], b[1]],
                                          [b[0] + b[2], b[1] + b[3]], [b[0], b[1] + b[3]]]))
        return np.array([P[:, 0].min(), P[:, 1].min(), np.ptp(P[:, 0]), np.ptp(P[:, 1])])

    def _lose_target(self, t):
        self.lost = True
        self.lost_since = t
        self.pts = np.zeros((0, 2), np.float32)
        self.since_det = np.inf
