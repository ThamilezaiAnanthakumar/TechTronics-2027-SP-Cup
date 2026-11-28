"""
Motion test on cached tracker traces (no video is reprocessed).

For every recording:
  * motion signals from the tracker: face-centre x / y in face widths, log face width,
    total skin-pixel count (facevitals.rppg.motion_regressors)
  * motion score per 10 s window: RMS of the head movement inside the HR band (0.7-3 Hz)
  * methods
      pos            baseline (face trace)
      pos_mot        + motion signals regressed out of the pulse signal per window
      kfsnr          current (Kalman + SNR fusion of the 3 regions)
      kfsnr_mot      + motion regression
      *_amot         adaptive: motion regression only in windows whose motion score is above
                     the threshold (still windows keep the plain estimate). On still people the
                     head moves slightly with each heartbeat (ballistocardiography), so
                     regressing motion everywhere removes part of the pulse.
      *_rej          the same, recording estimate = median over the STILL windows only
                     (motion score <= threshold; at least MIN_KEEP windows, the stillest ones)
  * threshold = 95th percentile of the UBFC window scores (people sitting still), fixed
    before looking at SRMC errors.

UBFC (PPG reference per window) is the clean control: motion handling must not hurt it.
SRMC-N Resting (one PR value per recording) is the target.

Usage
  python src/python/motion_test.py
"""

import mediapipe  # noqa: F401  # must load before scipy/pandas on Windows (DLL init conflict)
import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

from facevitals.datasets import subjects
from facevitals.face import REPO
from facevitals.rppg import (kalman_denoise, motion_regressors, motion_score, ppg_hr, sanity_hr_snr)

CACHE = REPO / "outputs" / "traces_py" / "tracker"
OUT = REPO / "outputs" / "tables"
WIN, STEP = 10.0, 1.0
MIN_KEEP = 3
METHODS = ["pos", "pos_mot", "pos_amot", "kfsnr", "kfsnr_mot", "kfsnr_amot"]


def load(s):
    f = CACHE / f"{s.dataset}_{s.subject}.npz"
    if not f.exists():
        return None
    d = dict(np.load(f, allow_pickle=False))
    d["fps"] = float(d["fps"])
    return d


def window_hrs(d, centers, thr=None):
    t = d["t"]
    names = [str(n) for n in d["roi_names"]]
    regions = [d["rgb_" + n] for n in names]
    npix = np.asarray(d["npix"], float)
    fps = d["fps"] if np.isfinite(d["fps"]) and d["fps"] > 0 else 30.0
    M = motion_regressors(d["face_box"], npix)
    kf = [kalman_denoise(x, fps, npix=npix[:, r]) for r, x in enumerate(regions)]
    hr = {
        "pos": sanity_hr_snr(t, [d["rgb_face"]], WIN, STEP, centers=centers)[0],
        "pos_mot": sanity_hr_snr(t, [d["rgb_face"]], WIN, STEP, centers=centers, refs=M)[0],
        "kfsnr": sanity_hr_snr(t, kf, WIN, STEP, centers=centers)[0],
        "kfsnr_mot": sanity_hr_snr(t, kf, WIN, STEP, centers=centers, refs=M)[0],
    }
    sc = motion_score(t, M, centers, WIN)
    moving = np.isfinite(sc) & (sc > thr) if thr is not None else np.zeros(len(sc), bool)
    for m in ("pos", "kfsnr"):
        hr[m + "_amot"] = np.where(moving, hr[m + "_mot"], hr[m])
    return hr, M


def centers_for(t):
    t = np.asarray(t, float)
    t = t[np.isfinite(t)]
    return np.arange(t[0] + WIN / 2, t[-1] - WIN / 2 + 1e-9, STEP)


def keep_still(score, thr):
    ok = np.isfinite(score)
    keep = ok & (score <= thr)
    if keep.sum() < MIN_KEEP and ok.sum():
        idx = np.argsort(np.where(ok, score, np.inf))[:MIN_KEEP]
        keep = np.zeros_like(ok)
        keep[idx] = True
        keep &= ok
    return keep


def ubfc_threshold():
    scores = []
    for s in subjects(["UBFC"]):
        d = load(s)
        if d is None:
            continue
        M = motion_regressors(d["face_box"], d["npix"])
        scores.append(motion_score(d["t"], M, centers_for(d["t"]), WIN))
    return float(np.nanpercentile(np.concatenate(scores), 95))


def run_ubfc(thr):
    rows = []
    for s in subjects(["UBFC"]):
        d = load(s)
        if d is None:
            continue
        _, tg, ppg = s.reference(d["t"], d["fps"])
        ref, tc = ppg_hr(tg, ppg, WIN, STEP)
        hr, M = window_hrs(d, tc, thr)
        sc = motion_score(d["t"], M, tc, WIN)
        row = dict(subject=s.subject)
        for m in METHODS:
            e = np.abs(hr[m] - ref)
            row[m + "_mae"] = float(np.nanmean(e))
        row["motion_med"] = float(np.nanmedian(sc))
        rows.append((row, hr, ref, sc))
    out = []
    for row, hr, ref, sc in rows:
        keep = keep_still(sc, thr)
        for m in METHODS:
            row[m + "_rej_mae"] = float(np.nanmean(np.abs(hr[m] - ref)[keep]))
        row["kept_pct"] = 100 * keep.mean()
        out.append(row)
    return pd.DataFrame(out)


def run_srmc(thr):
    rows = []
    for s in subjects(["SRMC"]):
        if s.meta.get("protocol") != "Resting" or s.meta.get("group") != "Normal":
            continue
        d = load(s)
        if d is None:
            continue
        tc = centers_for(d["t"])
        if len(tc) == 0:
            continue
        hr, M = window_hrs(d, tc, thr)
        sc = motion_score(d["t"], M, tc, WIN)
        keep = keep_still(sc, thr)
        ref = s.meta["hr_ref"]
        row = dict(subject=s.subject, skin_tone=s.meta["skin_tone"], age=s.meta["age"], hr_ref=ref,
                   motion_med=float(np.nanmedian(sc)), motion_max=float(np.nanmax(sc)),
                   kept_pct=100 * keep.mean(), n_faces_max=int(np.max(d["n_faces"])),
                   width=float(np.nanmedian(np.asarray(d["face_box"])[:, 2])))
        for m in METHODS:
            for tag, sel in (("", np.isfinite(hr[m])), ("_rej", keep & np.isfinite(hr[m]))):
                est = float(np.nanmedian(hr[m][sel])) if sel.any() else np.nan
                row[m + tag + "_hr"] = est
                row[m + tag + "_mae"] = abs(est - ref)
        rows.append(row)
    return pd.DataFrame(rows)


def summary(df, label):
    cols = [m + t for m in METHODS for t in ("", "_rej")]
    print(f"\n== {label}: n={len(df)}")
    base = df["pos_mae"]
    print(f"{'method':16s} {'MAE':>6s} {'median':>7s} {'<5bpm%':>7s} {'vs pos':>7s} {'better':>6s} {'worse':>6s} {'p':>6s}")
    for c in cols:
        x = df[c + "_mae"]
        d = x - base
        p = wilcoxon(base, x).pvalue if c != "pos" and np.any(d != 0) else np.nan
        print(f"{c:16s} {x.mean():6.2f} {x.median():7.2f} {100 * (x < 5).mean():7.1f} {d.mean():+7.2f} "
              f"{int((d < -0.5).sum()):6d} {int((d > 0.5).sum()):6d} {p:6.3f}")


def main():
    pd.set_option("display.width", 250)
    thr = ubfc_threshold()
    ubfc = run_ubfc(thr)
    print(f"motion threshold (UBFC 95th pct of window scores) = {thr:.4f} face widths RMS")
    summary(ubfc.rename(columns={c: c for c in ubfc.columns}), "UBFC (window MAE per subject, PPG ref)")
    print(f"UBFC windows kept: {ubfc.kept_pct.mean():.1f} %")
    srmc = run_srmc(thr)
    srmc = srmc[np.isfinite(srmc.pos_mae)]
    summary(srmc, "SRMC Resting (median of windows vs PR)")
    print(f"SRMC windows kept: {srmc.kept_pct.mean():.1f} %  (median clip motion {srmc.motion_med.median():.4f})")
    r = spearmanr(srmc.motion_med, srmc.pos_mae)
    print(f"Spearman clip motion vs pos error: rho={r.statistic:.2f} p={r.pvalue:.3f}")
    srmc["motion_tercile"] = pd.qcut(srmc.motion_med, 3, labels=["still", "medium", "moving"])
    print(srmc.groupby("motion_tercile", observed=True)[[m + "_mae" for m in ["pos", "pos_mot", "pos_amot", "pos_rej",
                                                                            "kfsnr", "kfsnr_amot", "kfsnr_rej"]]]
          .mean().round(2).to_string())
    print(srmc.groupby("skin_tone")[["motion_med", "pos_mae", "pos_amot_mae", "pos_rej_mae", "kfsnr_mae", "kfsnr_amot_mae"]]
          .mean().round(3).to_string())
    ubfc.to_csv(OUT / "motion_ubfc.csv", index=False)
    srmc.to_csv(OUT / "motion_srmc_resting.csv", index=False)
    (OUT / "motion_threshold.json").write_text(json.dumps(dict(threshold=thr, source="UBFC 95th percentile")))
    for s in ("S_N84_Resting", "S_N36_Resting", "S_N8_Resting"):
        r = srmc[srmc.subject == s]
        if len(r):
            print(s, r[["hr_ref", "motion_med", "kept_pct", "pos_hr", "pos_mot_hr", "pos_amot_hr", "pos_rej_hr"]]
                  .round(2).to_dict("records")[0])


if __name__ == "__main__":
    sys.exit(main())
