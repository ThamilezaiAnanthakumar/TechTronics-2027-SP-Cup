"""
SNR-weighted median: one HR per recording from the window HRs, each window weighted by the
SNR of its (cleaned) pulse spectrum, so clear windows count more than noisy ones.

Method per window: pos + background reference + SNR-gated motion regression (0 dB).
Aggregations
  median     plain median of window HRs (current)
  wmed_p1    weighted median, weight = SNR (linear)
  wmed_p2    weighted median, weight = SNR ** 2
  top50      median of the 50 % windows with the highest SNR
Chosen on SRMC Normal Resting, tested on SRMC Abnormal Resting; UBFC (recording-level,
reference = mean PPG HR) as a clean control.

Usage
  python src/python/snrmedian_test.py
"""
import mediapipe  # noqa: F401  # must load before scipy/pandas on Windows
import numpy as np
import pandas as pd
from scipy.stats import kruskal, wilcoxon

import motion_test as mt
import viterbi_test as vt
from facevitals.datasets import subjects
from facevitals.face import REPO
from facevitals.rppg import motion_regressors, ppg_hr, region_snr, sanity_hr_snr

AGGS = ["median", "wmed_p1", "wmed_p2", "top50"]


def windows(d, tc):
    t = d["t"]; f = [d["rgb_face"]]; bg = d.get("rgb_bg"); M = motion_regressors(d["face_box"])
    hb, _, (Pb, fb) = sanity_hr_snr(t, f, 10, 1, centers=tc, rgb_bg=bg, return_spectra=True)
    hbm, _, (Pbm, _) = sanity_hr_snr(t, f, 10, 1, centers=tc, rgb_bg=bg, refs=M, return_spectra=True)
    g = vt.window_snr_db(t, d["rgb_face"], tc) < vt.GATE_DB
    h = np.where(g, hbm, hb)
    P = np.where(g[:, None], Pbm, Pb) if Pb.shape[1] else Pb
    snr = np.array([region_snr(fb, p, (fb[0], fb[-1]))[0] if np.all(np.isfinite(p)) and p.sum() > 0 else np.nan
                    for p in P]) if P.shape[1] else np.full(len(h), np.nan)
    return h, snr


def wmedian(x, w):
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if not ok.any():
        return np.nan
    x, w = x[ok], w[ok]
    o = np.argsort(x); x, w = x[o], w[o]
    c = np.cumsum(w)
    return float(x[np.searchsorted(c, 0.5 * c[-1])])


def aggregate(h, snr):
    ok = np.isfinite(h) & np.isfinite(snr)
    top = ok & (snr >= np.nanmedian(snr[ok])) if ok.any() else ok
    return dict(median=float(np.nanmedian(h)), wmed_p1=wmedian(h, snr), wmed_p2=wmedian(h, snr ** 2),
                top50=float(np.median(h[top])) if top.any() else np.nan)


def show(x, label, skin=True):
    print(f"== {label} n={len(x)}")
    for c in AGGS:
        dd = x[c] - x["median"]
        p = wilcoxon(x["median"], x[c]).pvalue if c != "median" and np.any(np.abs(dd) > 1e-9) else np.nan
        s = ""
        if skin:
            k = kruskal(*[y[c] for _, y in x.groupby("skin")]).pvalue
            s = f"  skin p {k:.2f} " + " ".join(f"{sk[:4]} {y[c].mean():.1f}" for sk, y in x.groupby("skin"))
        print(f"  {c:8s} MAE {x[c].mean():6.2f} median {x[c].median():6.2f} <5 {int((x[c] < 5).sum()):3d} "
              f"better {int((dd < -.5).sum()):3d} worse {int((dd > .5).sum()):3d} p {p:.3f}{s}")


def main():
    rows = []
    for s in subjects(["SRMC"]):
        if s.meta.get("protocol") != "Resting":
            continue
        d = mt.load(s)
        if d is None:
            continue
        tc = mt.centers_for(d["t"])
        if len(tc) == 0:
            continue
        h, snr = windows(d, tc)
        if not np.isfinite(h).any():
            continue
        ref = s.meta["hr_ref"]
        a = aggregate(h, snr)
        rows.append(dict(clip=s.subject, group=s.meta["group"], skin=s.meta["skin_tone"], ref=ref,
                         **{k: abs(v - ref) for k, v in a.items()}, **{k + "_hr": v for k, v in a.items()}))
    sr = pd.DataFrame(rows)
    ub = []
    for s in subjects(["UBFC"]):
        d = mt.load(s)
        if d is None:
            continue
        _, tg, ppg = s.reference(d["t"], d["fps"])
        ref, tc = ppg_hr(tg, ppg, 10, 1)
        h, snr = windows(d, tc)
        r = float(np.nanmean(ref))
        ub.append(dict(clip=s.subject, **{k: abs(v - r) for k, v in aggregate(h, snr).items()}))
    ub = pd.DataFrame(ub)
    show(sr[sr.group == "Normal"], "SRMC Normal Resting (choose here)")
    show(sr[sr.group == "Abnormal"], "SRMC Abnormal Resting (test)")
    show(sr, "SRMC all")
    show(ub, "UBFC (recording HR vs mean PPG HR)", skin=False)
    out = REPO / "outputs" / "tables"
    sr.to_csv(out / "snrmedian_srmc_resting.csv", index=False)
    ub.to_csv(out / "snrmedian_ubfc.csv", index=False)


if __name__ == "__main__":
    main()
