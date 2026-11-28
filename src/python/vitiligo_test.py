"""
C2 vitiligo test on AiPEX crops: synthetic depigmented patches on real pigmented skin.

Synthesis (per crop, in memory, nothing saved)
  * patch field: smooth random field (periodic), thresholded at the requested coverage, so the
    patches grow with coverage (10 % patches lie inside the 25 % ones, ...). Edges softened.
  * patches stay on the skin: the skin's frame-to-frame shift inside the crop is estimated by
    phase correlation and the patch field is moved with it (leaky, so it cannot drift away).
    Without this the patch would move with the crop and the edge would never cross the skin.
  * depigmentation: in linear light, per-channel gains exp(k * a_c + g),
    a = (0.43, 0.58, 1.0) relative melanin absorption (R, G, B), g a neutral (all-wavelength)
    reflectance gain. (k, g) are solved per crop so the patch's median colour reaches
    L* = TARGET_L, b* = TARGET_B (pale, slightly warm: ITA ~ 58 deg). A melanin-only gain
    turns very dark skin blue (b* < 0) before it gets light. Patches only cover skin pixels.
    The pulse is scaled with the pixel (AC/DC unchanged) and so is the recorded camera
    noise -> conservative.

Handling ("aware" method)
  1. detect: per-pixel ITA on the first seconds; pigmented reference = 20th percentile of ITA,
     a pixel is depigmented when ITA > reference + DEPIG_JUMP (50 deg); flagged when more than
     DETECT_MIN_W (3 %) of the skin is depigmented. Clipped pixels (any channel >= CLIP_LEVEL)
     count for neither class; the depigmented sample is dropped if > CLIP_MAX_FRAC is clipped.
     Built for clean skin crops: on full-face SRMC videos it flags every clip (shadows and hair
     set the reference), see src/python/README.md.
  2. skin mask: standard YCbCr mask for pigmented skin + depigmented pixels by the ITA rule
     (the fixed chroma box can reject pale skin).
  3. split: each crop -> pigmented and depigmented traces, both masks eroded so the moving
     edge between them is not averaged.
  4. fuse: Kalman denoising + SNR fusion over the split traces (up to 6).

Usage
  python src/python/vitiligo_test.py --subjects India_video9
"""

import mediapipe  # noqa: F401  # must load before scipy/pandas on Windows (DLL init conflict)
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage
from skimage.color import rgb2lab

from facevitals.datasets import subjects
from facevitals.face import REPO, load_face_config, skin_mask, _fix_times
from facevitals.rppg import kalman_denoise, sanity_hr, sanity_hr_snr

MEL_ABS = np.array([0.43, 0.58, 1.0])   # relative melanin absorption R, G, B
TARGET_L, TARGET_B = 66.0, 10.0         # depigmented patch colour (ITA ~ 58 deg)
PATCH_SIGMA = 18                        # px, size of the random blobs on the 200 px crop
EDGE_SIGMA = 3                          # px, soft patch edge
MOTION_LEAK = 0.98                      # patch offset decays to 0 (crop is centred on the skin)
CALIB_FRAMES = 45                       # first frames: k calibration and detection (3 s at 15 fps)
DEPIG_JUMP = 50.0                       # degrees ITA above the pigmented reference
DETECT_MIN_W = 0.03                     # depigmented share of skin needed to flag
CLIP_LEVEL = 240                        # any channel >= this: clipped, no pulse
CLIP_MAX_FRAC = 0.30                    # depigmented sample dropped if more of it is clipped
L_MIN = 15                              # L* below: hair / eyebrow / shadow, never skin
ERODE_PX = 5                            # erosion kernel for the split masks
MIN_PIX = 100                           # pixels needed for a class sample in a frame
COVERAGES = (0.0, 0.10, 0.25, 0.50)
REGIONS = ("forehead", "leftcheek", "rightcheek")


# --------------------------------------------------------------- colour helpers

def srgb_to_lin(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def lin_to_srgb(x):
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(np.maximum(x, 0), 1 / 2.4) - 0.055)


_LUT = None


def _lut():
    """ITA and L* for every 7-bit RGB triple (2M entries, built once)."""
    global _LUT
    if _LUT is None:
        q = (np.arange(128) * 2 + 1).astype(np.float64) / 255.0
        r, g, b = np.meshgrid(q, q, q, indexing="ij")
        lab = rgb2lab(np.stack([r, g, b], -1).reshape(-1, 1, 3)).reshape(-1, 3)
        _LUT = (np.degrees(np.arctan2(lab[:, 0] - 50, lab[:, 2])).astype(np.float32),
                lab[:, 0].astype(np.float32))
    return _LUT


def ita_map(rgb):
    ita, L = _lut()
    x = rgb >> 1
    idx = (x[..., 0].astype(np.int32) << 14) | (x[..., 1].astype(np.int32) << 7) | x[..., 2]
    return ita[idx], L[idx]


def depigment(rgb, alpha, k):
    """rgb uint8 HxWx3, alpha HxW in [0,1], k = (melanin, neutral) -> uint8 depigmented."""
    lin = srgb_to_lin(rgb.astype(np.float64) / 255.0)
    gain = np.exp(k[0] * MEL_ABS + k[1])
    out = lin * (1 + alpha[..., None] * (gain - 1))
    return np.clip(np.round(lin_to_srgb(np.minimum(out, 1.0)) * 255), 0, 255).astype(np.uint8)


# --------------------------------------------------------------- patches

def patch_field(shape, seed):
    rng = np.random.default_rng(seed)
    return ndimage.gaussian_filter(rng.standard_normal(shape), PATCH_SIGMA, mode="wrap")


def patch_alpha(field, coverage):
    if coverage <= 0:
        return np.zeros_like(field)
    binary = (field >= np.quantile(field, 1 - coverage)).astype(np.float64)
    return np.clip(ndimage.gaussian_filter(binary, EDGE_SIGMA, mode="wrap"), 0, 1)


def shift_wrap(a, dx, dy):
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(a.astype(np.float32), M, (a.shape[1], a.shape[0]),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


def read_frames(path, n=None):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    k = 0
    while n is None or k < n:
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        ok, bgr = cap.read()
        if not ok:
            break
        yield fps, t, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        k += 1
    cap.release()


def calibrate_k(frames, skin):
    """(k, g) such that fully depigmented skin of the first frames has median L*, b* = target."""
    from scipy.optimize import minimize
    pix = np.concatenate([f[skin_mask(f, skin)] for f in frames[::5]])
    if len(pix) < 50:
        return (0.0, 0.0), np.nan
    pix = pix[np.random.default_rng(0).choice(len(pix), min(len(pix), 5000), replace=False)][:, None, :]
    one = np.ones((len(pix), 1))

    def lab_at(k):
        lab = rgb2lab(depigment(pix, one, k).astype(np.float64) / 255).reshape(-1, 3)
        return np.median(lab[:, 0]), np.median(lab[:, 2])

    def cost(x):
        L, b = lab_at((max(x[0], 0), x[1]))
        return (L - TARGET_L) ** 2 + (b - TARGET_B) ** 2

    best = min((minimize(cost, x0, method="Nelder-Mead", options=dict(xatol=1e-3, fatol=1e-3))
                for x0 in ([0.5, 1.0], [2.0, 0.5], [0.0, 2.0])), key=lambda r: r.fun)
    k = (max(best.x[0], 0.0), float(best.x[1]))
    L, b = lab_at(k)
    return k, float(np.degrees(np.arctan2(L - 50, b)))


def detect(frames, skin, satmax):
    """
    Depigmented skin = ITA far above the person's own pigmented skin.
    Pigmented reference = 20th percentile of per-pixel ITA (patches cover < ~60 % of skin, so
    this lies in normal skin); a pixel is depigmented when ITA > reference + DEPIG_JUMP.
    Shading / highlights on normal skin move ITA by ~10-30 deg; vitiligo by ~100 deg on dark skin.
    """
    vals, clipped = [], []
    for f in frames[::5]:
        ita, L = ita_map(f)
        ok = L > L_MIN
        vals.append(ita[ok])
    v = np.concatenate(vals)
    ref = float(np.percentile(v, 20))
    thr = ref + DEPIG_JUMP
    w = float((v > thr).mean())
    flagged = bool(w > DETECT_MIN_W)
    return flagged, thr, dict(ita_pig_ref=ref, threshold=thr, w_depig=w,
                              ita_depig_median=float(np.median(v[v > thr])) if w > 0 else np.nan)


# --------------------------------------------------------------- one subject

def run_subject(s, skin, out_fig):
    paths = {r: s.video[r] for r in REGIONS}
    seed = sum(map(ord, s.subject))  # stable across runs (hash() is salted)
    rows_info, traces = {}, {}
    erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ERODE_PX, ERODE_PX))
    montage = {c: [] for c in COVERAGES}
    mask_montage = {c: [] for c in COVERAGES}

    for ri, r in enumerate(paths):
        first = [f for _, _, f in read_frames(paths[r], CALIB_FRAMES)]
        H, W = first[0].shape[:2]
        k, ita_reached = calibrate_k(first, skin)
        field = patch_field((H, W), seed * 10 + ri)
        alphas = {c: patch_alpha(field, c) for c in COVERAGES}
        info = dict(k=k, ita_patch=ita_reached)

        # detection on the composited first frames (same motion model as below)
        det = {}
        for c in COVERAGES:
            comp = [depigment(f, alphas[c] * cv2.GaussianBlur(skin_mask(f, skin).astype(np.float32), (0, 0), 2), k)
                    for f in first]
            det[c] = detect(comp, skin, skin["satMax"])
            info[f"det_{c}"] = det[c][2] | dict(flagged=det[c][0])

        T = []
        acc = {(c, m): [] for c in COVERAGES for m in ("std", "pig", "depig")}
        cnt = {(c, m): [] for c in COVERAGES for m in ("std", "pig", "depig")}
        cov_skin = {c: [] for c in COVERAGES}
        prev, ox, oy, fps = None, 0.0, 0.0, None
        for k_f, (fps, t, f) in enumerate(read_frames(paths[r])):
            T.append(t)
            g = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY).astype(np.float32)
            if prev is not None:
                (dx, dy), resp = cv2.phaseCorrelate(prev, g)
                if resp > 0.1 and abs(dx) < 20 and abs(dy) < 20:
                    ox, oy = MOTION_LEAK * ox + dx, MOTION_LEAK * oy + dy
                else:
                    ox, oy = MOTION_LEAK * ox, MOTION_LEAK * oy
            prev = g
            for c in COVERAGES:
                a = shift_wrap(alphas[c], ox, oy) if c > 0 else alphas[c]
                if c > 0:   # patches only on skin (not eyebrows / hair / background)
                    a = a * cv2.GaussianBlur(skin_mask(f, skin).astype(np.float32), (0, 0), 2)
                fc = depigment(f, a, k) if c > 0 else f
                m_std = skin_mask(fc, skin)
                acc[(c, "std")].append(fc[m_std].mean(axis=0) if m_std.sum() >= MIN_PIX else np.full(3, np.nan))
                cnt[(c, "std")].append(int(m_std.sum()))
                flagged, thr, _ = det[c]
                if flagged:
                    ita, L = ita_map(fc)
                    clip = fc.max(axis=2) >= CLIP_LEVEL
                    dep_all = (ita > thr) & (L > L_MIN)
                    # pigmented: away from any patch pixel (clipped ones included)
                    near_dep = cv2.dilate(dep_all.astype(np.uint8), erode).astype(bool)
                    pig = cv2.erode((m_std & ~near_dep & ~clip).astype(np.uint8), erode).astype(bool)
                    dep = cv2.erode((dep_all & ~clip).astype(np.uint8), erode).astype(bool)
                    if dep_all.sum() and (dep_all & clip).sum() > CLIP_MAX_FRAC * dep_all.sum():
                        dep = np.zeros_like(dep)
                else:
                    pig, dep = m_std, np.zeros_like(m_std)
                for name, m in (("pig", pig), ("depig", dep)):
                    n = int(m.sum())
                    acc[(c, name)].append(fc[m].mean(axis=0) if n >= MIN_PIX else np.full(3, np.nan))
                    cnt[(c, name)].append(n)
                cov_skin[c].append(float((a > 0.5)[m_std | (a > 0.5)].mean()) if c > 0 else 0.0)
                if k_f == 300:
                    montage[c].append(fc)
                    ov = fc.copy()
                    ov[pig] = (0.5 * ov[pig] + [0, 0, 127]).astype(np.uint8)
                    ov[dep] = (0.5 * ov[dep] + [0, 127, 0]).astype(np.uint8)
                    mask_montage[c].append(ov)
        t = _fix_times(T, fps)
        traces[r] = dict(t=t, fps=fps, acc={key: np.array(v) for key, v in acc.items()},
                         cnt={key: np.array(v) for key, v in cnt.items()})
        info["patch_share_of_skin"] = {c: float(np.mean(cov_skin[c])) for c in COVERAGES}
        rows_info[r] = info

    # figures: composited crops and the aware split (blue = pigmented, green = depigmented)
    out_fig.mkdir(parents=True, exist_ok=True)
    img = np.vstack([np.hstack(montage[c]) for c in COVERAGES])
    img2 = np.vstack([np.hstack(mask_montage[c]) for c in COVERAGES])
    cv2.imwrite(str(out_fig / f"{s.subject}_patches.png"), cv2.cvtColor(np.hstack([img, img2]), cv2.COLOR_RGB2BGR))

    # HR
    n = min(len(v["t"]) for v in traces.values())
    t = traces[REGIONS[0]]["t"][:n]
    fps = float(traces[REGIONS[0]]["fps"])
    ref = s.meta["hr_ref"]
    res, win_hr = [], {}
    for c in COVERAGES:
        std = [traces[r]["acc"][(c, "std")][:n] for r in REGIONS]
        std_n = [traces[r]["cnt"][(c, "std")][:n] for r in REGIONS]
        tot = sum(np.nan_to_num(x * nn[:, None]) for x, nn in zip(std, std_n))
        ntot = sum(std_n)
        with np.errstate(invalid="ignore", divide="ignore"):
            face = np.where(ntot[:, None] > 0, tot / ntot[:, None], np.nan)
        split, split_n = [], []
        for r in REGIONS:
            for m in ("pig", "depig"):
                split.append(traces[r]["acc"][(c, m)][:n])
                split_n.append(traces[r]["cnt"][(c, m)][:n])
        methods = {
            "pos": lambda: sanity_hr(t, face)[0],
            "kfsnr": lambda: sanity_hr_snr(t, [kalman_denoise(x, fps) for x in std])[0],
            "aware": lambda: sanity_hr_snr(t, [kalman_denoise(x, fps, npix=nn) for x, nn in zip(split, split_n)],
                                           return_weights=True),
        }
        row = dict(subject=s.subject, coverage=c, hr_ref=ref)
        for name, fn in methods.items():
            out = fn()
            if name == "aware":
                hr, _, w = out
                row["aware_weights"] = json.dumps({f"{r}_{m}": round(float(x), 3) for (r, m), x in
                                                   zip([(r, m) for r in REGIONS for m in ("pig", "depig")], w)})
            else:
                hr = out
            win_hr[(c, name)] = hr
            est = float(np.nanmedian(hr))
            row[f"{name}_hr"] = est
            row[f"{name}_err"] = est - ref
            row[f"{name}_mae"] = abs(est - ref)
        row["flagged"] = sum(rows_info[r][f"det_{c}"]["flagged"] for r in REGIONS)
        row["patch_share_of_skin"] = np.mean([rows_info[r]["patch_share_of_skin"][c] for r in REGIONS])
        row["depig_valid_pct"] = 100 * np.mean([np.isfinite(traces[r]["acc"][(c, "depig")][:n, 0]).mean()
                                                for r in REGIONS])
        res.append(row)
    # paired window-level change against the clean clip (more sensitive than one ref value)
    for row in res:
        c = row["coverage"]
        for name in ("pos", "kfsnr", "aware"):
            d = np.abs(win_hr[(c, name)] - win_hr[(0.0, "kfsnr")])
            row[f"{name}_win_dev_vs_clean"] = float(np.nanmean(d))
            row[f"{name}_win_within5_pct"] = float(100 * np.nanmean(np.abs(win_hr[(c, name)] - ref) < 5))
    return res, rows_info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="+", required=True)
    a = ap.parse_args()
    _, skin, _ = load_face_config()
    out_tab = REPO / "outputs" / "tables"
    out_fig = REPO / "outputs" / "figures" / "vitiligo"
    rows, infos = [], {}
    for s in subjects(["AiPEX"]):
        if s.subject not in a.subjects:
            continue
        t0 = time.time()
        r, info = run_subject(s, skin, out_fig)
        rows += r
        infos[s.subject] = info
        print(f"{s.subject}: {time.time() - t0:.0f} s", flush=True)
    df = pd.DataFrame(rows)
    tag = "_".join(a.subjects) if len(a.subjects) <= 2 else "all"
    df.to_csv(out_tab / f"vitiligo_{tag}.csv", index=False)
    (out_tab / f"vitiligo_{tag}_info.json").write_text(json.dumps(infos, indent=1, default=float))
    cols = ["coverage", "patch_share_of_skin", "flagged", "depig_valid_pct", "pos_hr", "kfsnr_hr", "aware_hr",
            "pos_mae", "kfsnr_mae", "aware_mae", "kfsnr_win_dev_vs_clean", "aware_win_dev_vs_clean",
            "kfsnr_win_within5_pct", "aware_win_within5_pct"]
    pd.set_option("display.width", 250)
    print(df[["subject"] + cols].round(2).to_string(index=False))
    for subj, info in infos.items():
        for r, i in info.items():
            print(subj, r, "k=%.2f g=%.2f ITA_patch=%.1f" % (i["k"][0], i["k"][1], i["ita_patch"]),
                  {c: (round(i[f"det_{c}"]["ita_pig_ref"], 1), round(i[f"det_{c}"]["threshold"], 1),
                       round(i[f"det_{c}"]["w_depig"], 2), i[f"det_{c}"]["flagged"]) for c in COVERAGES})
    print("aware weights:", df[["coverage", "aware_weights"]].to_string(index=False))


if __name__ == "__main__":
    main()
