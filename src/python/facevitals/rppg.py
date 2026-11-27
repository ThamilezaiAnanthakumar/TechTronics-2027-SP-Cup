"""
POS rPPG and windowed spectral HR.

Line-by-line port of the MATLAB baseline so Python and MATLAB numbers are comparable:
  sanity_hr.m   -> sanity_hr   (resample to 30 Hz on real timestamps, POS, Butterworth, spectral_hr)
  ppg_hr.m      -> ppg_hr      (same steps on the contact PPG = reference HR)
  spectral_hr.m -> spectral_hr (10 s windows every 1 s, Hann periodogram peak in 0.7-3 Hz)
"""

import numpy as np
from scipy import signal

FS = 30.0
BAND = (0.7, 3.0)


def pos(X, fs=FS):
    """
    Plane-Orthogonal-to-Skin (Wang et al., IEEE TBME 2017), 1.6 s overlap-add.

    Vectorised over windows; identical output to pos_loop (the line-by-line port of
    sanity_hr.m), which is kept as the reference implementation.
    """
    X = np.asarray(X, dtype=np.float64)
    N = len(X)
    l = int(round(1.6 * fs))
    h = np.zeros(N)
    if N < l:
        return h
    W = np.lib.stride_tricks.sliding_window_view(X, l, axis=0)       # (N-l+1, 3, l)
    mu = W.mean(axis=2, keepdims=True)
    dark = np.any(mu[:, :, 0] <= 0, axis=1)                           # window with a zero channel
    with np.errstate(invalid="ignore", divide="ignore"):
        C = W / np.where(mu > 0, mu, 1.0)
    S0 = C[:, 1, :] - C[:, 2, :]                                      # [0 1 -1]
    S1 = -2 * C[:, 0, :] + C[:, 1, :] + C[:, 2, :]                    # [-2 1 1]
    sd1 = S1.std(axis=1, ddof=1)
    ok = (sd1 != 0) & ~dark
    alpha = np.zeros_like(sd1)
    alpha[ok] = S0[ok].std(axis=1, ddof=1) / sd1[ok]
    pp = S0 + alpha[:, None] * S1
    pp = pp - pp.mean(axis=1, keepdims=True)
    pp[~ok] = 0
    for k in range(l):                                                # overlap-add
        h[k:k + N - l + 1] += pp[:, k]
    return h


def pos_loop(X, fs=FS):
    """Reference loop implementation of POS (port of sanity_hr.m)."""
    N = len(X)
    l = int(round(1.6 * fs))
    h = np.zeros(N)
    P = np.array([[0, 1, -1], [-2, 1, 1]], dtype=np.float64)
    for n in range(N - l + 1):
        C = X[n:n + l].T
        S = P @ (C / C.mean(axis=1, keepdims=True))
        sd = S[1].std(ddof=1)
        if sd == 0:
            continue
        pp = S[0] + (S[0].std(ddof=1) / sd) * S[1]
        h[n:n + l] += pp - pp.mean()
    return h


def _bandpass(x, fs, band):
    b, a = signal.butter(3, np.array(band) / (fs / 2), btype="bandpass")
    return signal.filtfilt(b, a, x)


def default_centers(tu, win=10.0, step=1.0):
    return np.arange(tu[0] + win / 2, tu[-1] - win / 2 + 1e-9, step)


def spectral_hr(x, tu, fs=FS, win=10.0, step=1.0, band=BAND, centers=None):
    tc = default_centers(tu, win, step) if centers is None else np.asarray(centers, float)
    hr = np.full(len(tc), np.nan)
    L = int(round(win * fs))
    nfft = int(2 ** np.ceil(np.log2(max(8192, L))))
    for i, c in enumerate(tc):
        seg = x[(tu >= c - win / 2) & (tu < c + win / 2)]
        if len(seg) < 0.9 * L or not np.all(np.isfinite(seg)):
            continue
        f, P = signal.periodogram(seg - seg.mean(), fs=fs, window=signal.windows.hann(len(seg), sym=True),
                                  nfft=nfft, detrend=False)
        m = (f >= band[0]) & (f <= band[1])
        hr[i] = 60 * f[m][np.argmax(P[m])]
    return hr, tc


def sanity_hr(t, rgb, win=10.0, step=1.0, band=BAND, centers=None):
    """HR (bpm) per window from an Nx3 RGB trace (NaN rows allowed) with frame times t (s)."""
    t = np.asarray(t, float)
    rgb = np.asarray(rgb, float)
    valid = np.all(np.isfinite(rgb), axis=1) & np.isfinite(t)
    if valid.sum() < win * FS * 0.9:
        tc = np.asarray(centers if centers is not None else [], float)
        return np.full(len(tc), np.nan), tc
    tv, rv = t[valid], rgb[valid]
    tu = np.arange(tv[0], tv[-1] + 1e-9, 1 / FS)
    X = np.column_stack([np.interp(tu, tv, rv[:, c]) for c in range(3)])
    h = _bandpass(pos(X, FS), FS, band)
    return spectral_hr(h, tu, FS, win, step, band, centers)


# --------------------------------------------------------------- background-reference rejection

KAPPA = 1.0          # strength of the background suppression
BG_SMOOTH_BPM = 1.5  # background spectrum is smoothed over +-this many bpm


BG_MIN_LEVEL = 10.0  # background mean intensity (0-255) below this carries no light information


def background_usable(rgb_bg):
    """False if the background trace is missing or too dark to see the illumination."""
    if rgb_bg is None:
        return False
    B = np.asarray(rgb_bg, float)
    ok = np.all(np.isfinite(B), axis=1)
    return ok.mean() > 0.5 and float(np.median(B[ok].min(axis=1))) >= BG_MIN_LEVEL


def _resample(t, X, tu):
    ok = np.all(np.isfinite(X), axis=1) & np.isfinite(t)
    if ok.sum() < 2:
        return None
    return np.column_stack([np.interp(tu, t[ok], X[ok, c]) for c in range(X.shape[1])])


def sanity_hr_bg(t, rgb, rgb_bg, win=10.0, step=1.0, band=BAND, centers=None, kappa=KAPPA,
                 mode="regress", return_details=False):
    """
    POS HR with background-reference rejection of illumination artefacts.

    The face and a non-skin background trace go through the same steps (30 Hz resampling,
    POS, Butterworth band-pass). The background sees the same light flicker / exposure
    changes as the face, but no blood volume pulse.

    mode="regress" (default): in every window the face pulse signal is least-squares fitted
        by the background signals (background POS + its 3 normalised, band-passed colour
        channels) and the fit is subtracted before the periodogram. Only what the face shares
        with the background (illumination) is removed.
    mode="spectral": the face periodogram is divided by 1 + kappa * Pbg(f) / mean(Pbg).
        Kept for comparison: with 10 s windows the spectral peaks are ~+-10 bpm wide, so the
        flanks of a strong flicker peak survive the division.

    POS itself is unchanged. Falls back to sanity_hr when the background trace is missing.
    """
    t = np.asarray(t, float)
    rgb = np.asarray(rgb, float)
    if not background_usable(rgb_bg):
        return sanity_hr(t, rgb, win, step, band, centers)
    rgb_bg = np.asarray(rgb_bg, float)
    valid = np.all(np.isfinite(rgb), axis=1) & np.isfinite(t)
    if valid.sum() < win * FS * 0.9:
        tc = np.asarray(centers if centers is not None else [], float)
        return np.full(len(tc), np.nan), tc
    tv = t[valid]
    tu = np.arange(tv[0], tv[-1] + 1e-9, 1 / FS)
    Xf, Xb = _resample(t, rgb, tu), _resample(t, rgb_bg, tu)
    if Xb is None:
        return sanity_hr(t, rgb, win, step, band, centers)
    hf = _bandpass(pos(Xf, FS), FS, band)
    hb = _bandpass(pos(Xb, FS), FS, band)
    # regressors: background POS + normalised background colour channels, all band-passed
    Bn = Xb / np.maximum(Xb.mean(axis=0), 1e-9) - 1
    R = np.column_stack([hb] + [_bandpass(Bn[:, c], FS, band) for c in range(3)])

    tc = default_centers(tu, win, step) if centers is None else np.asarray(centers, float)
    hr = np.full(len(tc), np.nan)
    L = int(round(win * FS))
    nfft = int(2 ** np.ceil(np.log2(max(8192, L))))
    k = max(1, int(round(BG_SMOOTH_BPM / 60 / (FS / nfft))))
    kern = np.ones(2 * k + 1) / (2 * k + 1)
    details = []
    for i, c in enumerate(tc):
        sel = (tu >= c - win / 2) & (tu < c + win / 2)
        sf, sb = hf[sel], hb[sel]
        if len(sf) < 0.9 * L or not (np.all(np.isfinite(sf)) and np.all(np.isfinite(sb))):
            continue
        wnd = signal.windows.hann(len(sf), sym=True)
        f, Pf = signal.periodogram(sf - sf.mean(), fs=FS, window=wnd, nfft=nfft, detrend=False)
        _, Pb = signal.periodogram(sb - sb.mean(), fs=FS, window=wnd, nfft=nfft, detrend=False)
        m = (f >= band[0]) & (f <= band[1])
        Pb = np.convolve(Pb, kern, mode="same")[m]
        if mode == "regress":
            Rw = R[sel] - R[sel].mean(axis=0)
            y = sf - sf.mean()
            beta, *_ = np.linalg.lstsq(Rw, y, rcond=None)
            res = y - Rw @ beta
            _, Pc = signal.periodogram(res, fs=FS, window=wnd, nfft=nfft, detrend=False)
            clean = Pc[m]
        else:
            clean = Pf[m] / (1 + kappa * Pb / max(Pb.mean(), 1e-30))
        hr[i] = 60 * f[m][np.argmax(clean)]
        if return_details:
            details.append(dict(tc=c, hr_raw=60 * f[m][np.argmax(Pf[m])], hr_bg_peak=60 * f[m][np.argmax(Pb)],
                                bg_peakiness=float(Pb.max() / Pb.mean())))
    return (hr, tc, details) if return_details else (hr, tc)


# --------------------------------------------------------------- SNR-weighted region fusion

SNR_F0_HZ = 0.1      # signal band around the peak (de Haan & Jeanne 2013)
SNR_F1_HZ = 0.2      # signal band around the first harmonic
SNR_POWER = 1.0      # weight = SNR (linear) ** SNR_POWER


def region_snr(f, P, band=BAND):
    """
    de Haan & Jeanne (2013) SNR of a pulse spectrum, linear: power within +-0.1 Hz of the
    in-band peak and +-0.2 Hz of its first harmonic, divided by the remaining in-band power.
    Returns (snr, peak frequency in Hz).
    """
    m = (f >= band[0]) & (f <= band[1])
    fb, Pb = f[m], P[m]
    f0 = fb[np.argmax(Pb)]
    sig = (np.abs(fb - f0) <= SNR_F0_HZ) | (np.abs(fb - 2 * f0) <= SNR_F1_HZ)
    return float(Pb[sig].sum() / max(Pb[~sig].sum(), 1e-30)), float(f0)


def sanity_hr_snr(t, region_rgbs, win=10.0, step=1.0, band=BAND, centers=None, rgb_bg=None,
                  power=SNR_POWER, return_weights=False):
    """
    HR from several ROIs fused by their per-window pulse SNR.

    Each region's RGB trace goes through the baseline steps (30 Hz resampling, POS,
    Butterworth band-pass). In every window each region's periodogram is normalised to unit
    in-band power and weighted by its SNR ** power; the weighted spectra are summed and the
    peak of the sum is the HR. A region with a weak or noisy pulse (shadow, partial cover,
    low reflectance, mixed skin) gets a small weight in that window only.

    rgb_bg: optional background trace; if given, the background signals are regressed out
    of every region's pulse signal first (as in sanity_hr_bg).
    Returns hr, tc (and the mean normalised weight per region if return_weights).
    """
    t = np.asarray(t, float)
    R = len(region_rgbs)
    regs = [np.asarray(x, float) for x in region_rgbs]
    any_valid = np.zeros(len(t), bool)
    for X in regs:
        any_valid |= np.all(np.isfinite(X), axis=1)
    any_valid &= np.isfinite(t)
    tc_in = np.asarray(centers, float) if centers is not None else None
    if any_valid.sum() < win * FS * 0.9:
        tc = tc_in if tc_in is not None else np.array([])
        out = (np.full(len(tc), np.nan), tc)
        return out + (np.full(R, np.nan),) if return_weights else out
    tv = t[any_valid]
    tu = np.arange(tv[0], tv[-1] + 1e-9, 1 / FS)

    # per-region pulse signals; a region valid in < 50 % of frames is not used
    hs, okmask = [], []
    for X in regs:
        v = np.all(np.isfinite(X), axis=1)
        if v.sum() < 0.5 * any_valid.sum():
            hs.append(None); okmask.append(None)
            continue
        Xu = _resample(t, X, tu)
        hs.append(_bandpass(pos(Xu, FS), FS, band))
        # frames of tu that fall in long gaps of this region are marked invalid
        okmask.append(np.interp(tu, t[np.isfinite(t)], v[np.isfinite(t)].astype(float)) > 0.5)

    Rg = None
    if background_usable(rgb_bg):
        Xb = _resample(t, np.asarray(rgb_bg, float), tu)
        if Xb is not None:
            Bn = Xb / np.maximum(Xb.mean(axis=0), 1e-9) - 1
            Rg = np.column_stack([_bandpass(pos(Xb, FS), FS, band)] +
                                 [_bandpass(Bn[:, c], FS, band) for c in range(3)])

    tc = default_centers(tu, win, step) if tc_in is None else tc_in
    hr = np.full(len(tc), np.nan)
    L = int(round(win * FS))
    nfft = int(2 ** np.ceil(np.log2(max(8192, L))))
    wsum, wcount = np.zeros(R), 0
    for i, c in enumerate(tc):
        sel = (tu >= c - win / 2) & (tu < c + win / 2)
        if sel.sum() < 0.9 * L:
            continue
        wnd = signal.windows.hann(int(sel.sum()), sym=True)
        fused, w_all = None, np.zeros(R)
        for r in range(R):
            if hs[r] is None or okmask[r][sel].mean() < 0.9:
                continue
            y = hs[r][sel] - hs[r][sel].mean()
            if not np.all(np.isfinite(y)):
                continue
            if Rg is not None and np.all(np.isfinite(Rg[sel])):
                Rw = Rg[sel] - Rg[sel].mean(axis=0)
                beta, *_ = np.linalg.lstsq(Rw, y, rcond=None)
                y = y - Rw @ beta
            f, P = signal.periodogram(y, fs=FS, window=wnd, nfft=nfft, detrend=False)
            m = (f >= band[0]) & (f <= band[1])
            tot = P[m].sum()
            if tot <= 0:
                continue
            snr, _ = region_snr(f, P, band)
            w_all[r] = snr ** power
            fused = w_all[r] * P[m] / tot if fused is None else fused + w_all[r] * P[m] / tot
        if fused is None:
            continue
        hr[i] = 60 * f[m][np.argmax(fused)]
        if w_all.sum() > 0:
            wsum += w_all / w_all.sum()
            wcount += 1
    if return_weights:
        return hr, tc, (wsum / wcount if wcount else np.full(R, np.nan))
    return hr, tc


def ppg_hr(tg, ppg, win=10.0, step=1.0, band=BAND, centers=None):
    """Reference HR (bpm) per window from a contact PPG, same method as sanity_hr."""
    tg, iu = np.unique(np.asarray(tg, float), return_index=True)
    ppg = np.asarray(ppg, float)[iu]
    tu = np.arange(tg[0], tg[-1] + 1e-9, 1 / FS)
    x = _bandpass(np.interp(tu, tg, ppg), FS, band)
    return spectral_hr(x, tu, FS, win, step, band, centers)


def error_stats(est, ref):
    """MAE, RMSE, Pearson r, bias, SUC@3/5 over windows where both are finite."""
    est, ref = np.asarray(est, float), np.asarray(ref, float)
    ok = np.isfinite(est) & np.isfinite(ref)
    e = est[ok] - ref[ok]
    n = int(ok.sum())
    if n == 0:
        return dict(n_windows=0, mae=np.nan, rmse=np.nan, r=np.nan, bias=np.nan, suc3_pct=np.nan, suc5_pct=np.nan)
    r = float(np.corrcoef(est[ok], ref[ok])[0, 1]) if n > 2 and np.std(est[ok]) > 0 and np.std(ref[ok]) > 0 else np.nan
    return dict(n_windows=n, mae=float(np.abs(e).mean()), rmse=float(np.sqrt((e ** 2).mean())), r=r,
                bias=float(e.mean()), suc3_pct=100 * float((np.abs(e) < 3).mean()),
                suc5_pct=100 * float((np.abs(e) < 5).mean()))
