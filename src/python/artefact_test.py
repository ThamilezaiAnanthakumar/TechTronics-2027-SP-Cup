"""
What causes the low-frequency artefact on SRMC (wrong peaks at ~55-65 bpm)?

Per recording (cached tracker traces, SRMC Resting):
  artefact   strongest peak of the averaged window spectrum that is > 8 bpm away from the
             reference HR (POS face signal, no cleaning, and after background + motion)
  ref_rank   power at the reference HR relative to the artefact peak
Candidate sources, each tested for a peak at the artefact frequency:
  background   POS + luminance of the background trace (light flicker, auto exposure, compression)
  motion       face-box x, y, scale (head sway / breathing movement)
  breathing    breathing rate from slow vertical head motion and slow face luminance
               (0.1-0.5 Hz); artefact compared with 2x, 3x, 4x breathing rate
  keyframes    periodic jumps in frame-to-frame background brightness (video GOP); artefact
               compared with the jump rate
  regions      artefact present in forehead / both cheeks alike (global) or in one region
  filter edge  where the artefacts sit relative to the 42 bpm (0.7 Hz) band edge
"""
import mediapipe  # noqa: F401
import sys
import numpy as np
import pandas as pd
from scipy import signal

import motion_test as mt
import viterbi_test as vt
from facevitals.datasets import subjects
from facevitals.face import REPO
from facevitals.rppg import BAND, FS, _bandpass, _resample, motion_regressors, pos, sanity_hr_snr

NFFT = 8192


def avg_spec(x, fs, win=10.0, step=1.0, band=BAND):
    """Mean normalised Hann periodogram of 10 s windows of x (already filtered)."""
    L = int(win * fs); acc = None; n = 0
    for a in range(0, len(x) - L + 1, int(step * fs)):
        y = x[a:a + L] - x[a:a + L].mean()
        if not np.all(np.isfinite(y)):
            continue
        f, P = signal.periodogram(y, fs, window=signal.windows.hann(L, sym=True), nfft=NFFT, detrend=False)
        m = (f >= band[0]) & (f <= band[1])
        if P[m].sum() <= 0:
            continue
        p = P[m] / P[m].sum()
        acc = p if acc is None else acc + p; n += 1
    return (f[m] * 60, acc / n) if n else (None, None)


def peak_rel(bpm, P, at, tol=2.0):
    """Power near `at` relative to the median in-band power (>> 1 = a peak there)."""
    m = np.abs(bpm - at) <= tol
    return float(P[m].max() / np.median(P)) if m.any() else np.nan


def main():
    rows = []
    for s in subjects(["SRMC"]):
        if s.meta.get("protocol") != "Resting":
            continue
        d = mt.load(s)
        if d is None:
            continue
        t = d["t"]
        ok = np.all(np.isfinite(d["rgb_face"]), 1) & np.isfinite(t)
        if ok.sum() < 12 * FS:
            continue
        tu = np.arange(t[ok][0], t[ok][-1] + 1e-9, 1 / FS)
        ref = s.meta["hr_ref"]
        face = _bandpass(pos(_resample(t, d["rgb_face"], tu), FS), FS, BAND)
        bpm, Pf = avg_spec(face, FS)
        if bpm is None:
            continue
        far = np.abs(bpm - ref) > 8
        art = float(bpm[far][np.argmax(Pf[far])])
        art_pow = float(Pf[far].max())
        ref_pow = float(Pf[np.abs(bpm - ref) <= 3].max())
        # same after background + gated motion (our final method)
        tc = mt.centers_for(t)
        M = motion_regressors(d["face_box"])
        _, _, (Pb, fb) = sanity_hr_snr(t, [d["rgb_face"]], 10, 1, centers=tc, rgb_bg=d["rgb_bg"], return_spectra=True)
        _, _, (Pbm, _) = sanity_hr_snr(t, [d["rgb_face"]], 10, 1, centers=tc, rgb_bg=d["rgb_bg"], refs=M, return_spectra=True)
        g = vt.window_snr_db(t, d["rgb_face"], tc) < vt.GATE_DB
        P = np.where(g[:, None], Pbm, Pb)
        okw = np.isfinite(P).all(1) & (np.nansum(P, 1) > 0)
        Pc = (P[okw] / P[okw].sum(1, keepdims=True)).mean(0); bc = fb * 60
        art_after = float(bc[np.abs(bc - ref) > 8][np.argmax(Pc[np.abs(bc - ref) > 8])])
        # background
        bg = _resample(t, d["rgb_bg"], tu)
        bg_pos = _bandpass(pos(bg, FS), FS, BAND)
        bg_lum = _bandpass(bg.mean(1) / np.nanmean(bg.mean(1)) - 1, FS, BAND)
        bb, Pbg = avg_spec(bg_pos, FS); _, Pbl = avg_spec(bg_lum, FS)
        # motion
        Mu = _resample(t, M, tu)
        mot = {}
        for k, name in enumerate(["x", "y", "scale"]):
            mb, Pm = avg_spec(_bandpass(Mu[:, k] - np.nanmean(Mu[:, k]), FS, BAND), FS)
            mot[name] = peak_rel(mb, Pm, art) if mb is not None else np.nan
        # breathing rate: slow vertical head motion and slow face luminance (6-30 breaths/min)
        def slow_peak(x):
            x = np.asarray(x, float)
            if not np.all(np.isfinite(x)) or len(x) < 20 * FS:
                return np.nan
            sb, sa = signal.butter(2, [0.1, 0.5], btype="band", fs=FS)
            y = signal.filtfilt(sb, sa, x - x.mean())
            f, Pp = signal.periodogram(y, FS, window="hann", nfft=1 << 15)
            m = (f >= 0.1) & (f <= 0.5)
            return float(f[m][np.argmax(Pp[m])] * 60)
        rr_y = slow_peak(Mu[:, 1]); rr_l = slow_peak(_resample(t, d["rgb_face"], tu).mean(1))
        rr = rr_y
        harm = {k: (abs(art - k * rr) <= 3) if np.isfinite(rr) else False for k in (2, 3, 4)}
        # keyframes: periodic jumps in frame-to-frame brightness of the background (original frame times)
        lum = np.asarray(d["rgb_bg"], float).mean(1)
        jump = np.abs(np.diff(lum)); jump = np.where(np.isfinite(jump), jump, 0)
        fps = float(d["fps"]) if d["fps"] > 0 else 30.0
        ac = np.correlate(jump - jump.mean(), jump - jump.mean(), "full")[len(jump) - 1:]
        ac = ac / max(ac[0], 1e-12)
        lo, hi = int(0.3 * fps), int(3.0 * fps)
        lag = lo + int(np.argmax(ac[lo:hi])) if hi < len(ac) else np.nan
        gop_bpm = 60 * fps / lag if np.isfinite(lag) else np.nan
        gop_strength = float(ac[lag]) if np.isfinite(lag) else np.nan
        # regions
        reg = {}
        for n in [str(x) for x in d["roi_names"]]:
            rb, Pr = avg_spec(_bandpass(pos(_resample(t, d["rgb_" + n], tu), FS), FS, BAND), FS)
            reg[n] = peak_rel(rb, Pr, art) if rb is not None else np.nan
        W, H = None, None
        rows.append(dict(
            clip=s.subject.replace("_Resting", ""), group=s.meta["group"], skin=s.meta["skin_tone"], ref=ref,
            fps=fps, face_w=float(np.nanmedian(np.asarray(d["face_box"])[:, 2])),
            artefact=art, artefact_after=art_after, art_over_ref=art_pow / ref_pow,
            art_peak=peak_rel(bpm, Pf, art), ref_peak=peak_rel(bpm, Pf, ref),
            bg_pos_at_art=peak_rel(bb, Pbg, art), bg_lum_at_art=peak_rel(bb, Pbl, art),
            bg_pos_at_ref=peak_rel(bb, Pbg, ref),
            mot_x=mot["x"], mot_y=mot["y"], mot_scale=mot["scale"],
            rr_y=rr_y, rr_lum=rr_l, art_2rr=harm[2], art_3rr=harm[3], art_4rr=harm[4],
            gop_bpm=gop_bpm, gop_strength=gop_strength, art_eq_gop=bool(abs(art - gop_bpm) <= 3) if np.isfinite(gop_bpm) else False,
            **{"reg_" + k: v for k, v in reg.items()}))
    df = pd.DataFrame(rows)
    out = REPO / "outputs" / "tables" / "artefact_srmc_resting.csv"
    df.to_csv(out, index=False)
    print(len(df), "clips ->", out)


if __name__ == "__main__":
    sys.exit(main())
