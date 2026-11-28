"""Harmonic check: score(f) = P(f) + a * max P(2f +- 1.5 bpm), searched in 42-180 bpm, on a 0.7-6 Hz spectrum.
Method = pos + background + SNR-gated motion. a chosen on SRMC Normal, tested on Abnormal; UBFC as control.

Usage
  python src/python/harmonic_test.py
"""
import mediapipe  # noqa
import sys, numpy as np, pandas as pd
import motion_test as mt, viterbi_test as vt
from scipy.stats import wilcoxon, kruskal
from facevitals.datasets import subjects
from facevitals.face import REPO
from facevitals.rppg import sanity_hr_snr, motion_regressors, ppg_hr
WIDE = (0.7, 6.0)
ALPHAS = [0.0, 0.25, 0.5, 1.0]

def spectra(d, tc):
    t = d["t"]; f = [d["rgb_face"]]; bg = d.get("rgb_bg"); M = motion_regressors(d["face_box"])
    hn, _ = sanity_hr_snr(t, f, 10, 1, centers=tc, rgb_bg=bg)            # standard narrow band (no harmonic)
    hnm, _ = sanity_hr_snr(t, f, 10, 1, centers=tc, rgb_bg=bg, refs=M)
    _, _, (Pb, fw) = sanity_hr_snr(t, f, 10, 1, band=WIDE, centers=tc, rgb_bg=bg, return_spectra=True)
    _, _, (Pbm, _) = sanity_hr_snr(t, f, 10, 1, band=WIDE, centers=tc, rgb_bg=bg, refs=M, return_spectra=True)
    g = vt.window_snr_db(t, d["rgb_face"], tc) < vt.GATE_DB
    return np.where(g, hnm, hn), np.where(g[:, None], Pbm, Pb), fw * 60

def pick(P, bpm, a):
    out = np.full(len(P), np.nan)
    srch = np.where((bpm >= 42) & (bpm <= 180))[0]
    db = bpm[1] - bpm[0]; k = int(np.ceil(1.5 / db))
    for i, p in enumerate(P):
        if not np.all(np.isfinite(p)) or p.sum() <= 0: continue
        p = p / p.sum()
        h2 = np.interp(2 * bpm[srch], bpm, p)
        if a:
            # local max around the harmonic (bin tolerance)
            idx = np.clip(np.searchsorted(bpm, 2 * bpm[srch]), 0, len(bpm) - 1)
            h2 = np.array([p[max(j - k, 0):j + k + 1].max() for j in idx])
        out[i] = bpm[srch][np.argmax(p[srch] + a * h2)]
    return out

def run_srmc():
    rows = []
    for s in subjects(["SRMC"]):
        if s.meta.get("protocol") != "Resting": continue
        d = mt.load(s)
        if d is None: continue
        tc = mt.centers_for(d["t"])
        if len(tc) == 0: continue
        h, P, bpm = spectra(d, tc)
        if not np.isfinite(h).any(): continue
        ref = s.meta["hr_ref"]
        row = dict(clip=s.subject, group=s.meta["group"], skin=s.meta["skin_tone"], ref=ref, std=abs(np.nanmedian(h) - ref))
        for a in ALPHAS:
            row[f"a{a}"] = abs(np.nanmedian(pick(P, bpm, a)) - ref)
        rows.append(row)
    return pd.DataFrame(rows)

def run_ubfc():
    rows = []
    for s in subjects(["UBFC"]):
        d = mt.load(s)
        if d is None: continue
        _, tg, ppg = s.reference(d["t"], d["fps"]); ref, tc = ppg_hr(tg, ppg, 10, 1)
        h, P, bpm = spectra(d, tc)
        row = dict(clip=s.subject, std=np.nanmean(np.abs(h - ref)))
        for a in ALPHAS:
            row[f"a{a}"] = np.nanmean(np.abs(pick(P, bpm, a) - ref))
        rows.append(row)
    return pd.DataFrame(rows)

C = ["std"] + [f"a{a}" for a in ALPHAS]
def show(x, label, skin=True):
    print(f"== {label} n={len(x)}")
    for c in C:
        p = wilcoxon(x["std"], x[c]).pvalue if c != "std" and np.any(np.abs(x[c] - x["std"]) > 1e-9) else np.nan
        dd = x[c] - x["std"]
        s = ""
        if skin:
            k = kruskal(*[y[c] for _, y in x.groupby("skin")]).pvalue
            s = f" skin p {k:.2f} " + " ".join(f"{sk[:4]} {y[c].mean():.1f}" for sk, y in x.groupby("skin"))
        print(f"  {c:6s} MAE {x[c].mean():6.2f} median {x[c].median():6.2f} <5 {int((x[c]<5).sum()):3d} "
              f"better {int((dd<-.5).sum()):3d} worse {int((dd>.5).sum()):3d} p {p:.3f}{s}")

sr = run_srmc(); ub = run_ubfc()
show(sr[sr.group == "Normal"], "SRMC Normal (choose a here)")
show(sr[sr.group == "Abnormal"], "SRMC Abnormal (test)")
show(sr, "SRMC all")
show(ub, "UBFC (window MAE vs PPG)", skin=False)
OUT = REPO / "outputs" / "tables"
sr.to_csv(OUT / "harmonic_srmc_resting.csv", index=False); ub.to_csv(OUT / "harmonic_ubfc.csv", index=False)
