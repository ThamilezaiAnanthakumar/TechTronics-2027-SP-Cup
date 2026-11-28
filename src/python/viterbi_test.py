"""
Viterbi HR tracking (port of the team's viterbiHR.m) on cached tracker traces.

Methods (window spectra from sanity_hr_snr, then either per-window argmax or the Viterbi path):
  pos, pos_vit            face trace
  kfsnr, kfsnr_vit        Kalman + SNR fusion of the 3 regions (CP3 core)
  posmg, posmg_vit        pos with SNR-gated motion regression (window SNR < 0 dB: box motion
                          regressed out; see motion_test.py)
UBFC: window MAE vs PPG. SRMC-N Normal Resting: median of windows vs PR.

Usage
  python src/python/viterbi_test.py
"""

import mediapipe  # noqa: F401  # must load before scipy/pandas on Windows (DLL init conflict)
import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import wilcoxon

import motion_test as mt
from facevitals.datasets import subjects
from facevitals.face import REPO
from facevitals.rppg import (BAND, FS, _bandpass, _resample, kalman_denoise, motion_regressors, pos,
                             ppg_hr, region_snr, sanity_hr_snr, viterbi_hr)

GATE_DB = 0.0
METHODS = ["pos", "pos_vit", "kfsnr", "kfsnr_vit", "posmg", "posmg_vit"]


def window_snr_db(t, rgb, tc, win=10.0):
    ok = np.all(np.isfinite(rgb), 1) & np.isfinite(t)
    tu = np.arange(t[ok][0], t[ok][-1] + 1e-9, 1 / FS)
    h = _bandpass(pos(_resample(t, rgb, tu), FS), FS, BAND)
    out = np.full(len(tc), np.nan)
    for i, c in enumerate(tc):
        sel = (tu >= c - win / 2) & (tu < c + win / 2)
        if sel.sum() < 0.9 * win * FS:
            continue
        f, P = signal.periodogram(h[sel] - h[sel].mean(), FS, window=signal.windows.hann(int(sel.sum()), sym=True),
                                  nfft=8192, detrend=False)
        out[i] = 10 * np.log10(region_snr(f, P, BAND)[0])
    return out


def estimates(d, tc):
    t = d["t"]
    names = [str(n) for n in d["roi_names"]]
    npix = np.asarray(d["npix"], float)
    fps = d["fps"] if np.isfinite(d["fps"]) and d["fps"] > 0 else 30.0
    kf = [kalman_denoise(d["rgb_" + n], fps, npix=npix[:, r]) for r, n in enumerate(names)]
    M = motion_regressors(d["face_box"])
    out = {}
    h0, _, (P0, fb) = sanity_hr_snr(t, [d["rgb_face"]], 10, 1, centers=tc, return_spectra=True)
    hk, _, (Pk, _) = sanity_hr_snr(t, kf, 10, 1, centers=tc, return_spectra=True)
    hm, _, (Pm, _) = sanity_hr_snr(t, [d["rgb_face"]], 10, 1, centers=tc, refs=M, return_spectra=True)
    gate = window_snr_db(t, d["rgb_face"], tc) < GATE_DB
    Pg = np.where(gate[:, None], Pm, P0)
    out["pos"], out["kfsnr"], out["posmg"] = h0, hk, np.where(gate, hm, h0)
    for name, P in (("pos", P0), ("kfsnr", Pk), ("posmg", Pg)):
        out[name + "_vit"] = viterbi_hr(P, fb)[0] if P.shape[1] else np.full(len(tc), np.nan)
        out[name + "_vit"][~np.isfinite(out[name])] = np.nan     # no estimate where no data
    return out


def table(rows, label, base="pos"):
    df = pd.DataFrame(rows)
    print(f"\n== {label}: n={len(df)}")
    b = df[base]
    for m in METHODS:
        x = df[m]
        dlt = x - b
        p = wilcoxon(b, x).pvalue if m != base and np.any(np.abs(dlt) > 1e-9) else np.nan
        print(f"  {m:10s} MAE {x.mean():6.2f}  median {x.median():6.2f}  <5bpm {100 * (x < 5).mean():5.1f}%  "
              f"vs pos {dlt.mean():+6.2f}  better {int((dlt < -0.5).sum()):3d} worse {int((dlt > 0.5).sum()):3d}  p {p:.3f}")
    return df


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="Normal", choices=["Normal", "Abnormal"], help="SRMC group")
    group = ap.parse_args().group
    ub = []
    for s in subjects(["UBFC"]):
        d = mt.load(s)
        if d is None:
            continue
        _, tg, ppg = s.reference(d["t"], d["fps"])
        ref, tc = ppg_hr(tg, ppg, 10, 1)
        e = estimates(d, tc)
        ub.append(dict(subject=s.subject, **{m: float(np.nanmean(np.abs(e[m] - ref))) for m in METHODS}))
    table(ub, "UBFC (window MAE vs PPG)")
    sr = []
    for s in subjects(["SRMC"]):
        if s.meta.get("protocol") != "Resting" or s.meta.get("group") != group:
            continue
        d = mt.load(s)
        if d is None:
            continue
        tc = mt.centers_for(d["t"])
        if len(tc) == 0:
            continue
        e = estimates(d, tc)
        if not np.isfinite(e["pos"]).any():
            continue
        ref = s.meta["hr_ref"]
        sr.append(dict(subject=s.subject, skin_tone=s.meta["skin_tone"], hr_ref=ref, age=s.meta.get("age"),
                       **{m + "_hr": float(np.nanmedian(e[m])) for m in METHODS},
                       **{m: abs(float(np.nanmedian(e[m])) - ref) for m in METHODS}))
    df = table(sr, f"SRMC {group} Resting (median of windows vs PR)")
    print(df.groupby("skin_tone")[METHODS].mean().round(2).to_string())
    df.to_csv(REPO / "outputs" / "tables" / ("viterbi_srmc_resting.csv" if group == "Normal" else "viterbi_srmc_resting_abnormal.csv"), index=False)
    pd.DataFrame(ub).to_csv(REPO / "outputs" / "tables" / "viterbi_ubfc.csv", index=False)


if __name__ == "__main__":
    main()
