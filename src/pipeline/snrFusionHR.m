function [hr, tc, wMean, Pf, fbOut] = snrFusionHR(t, X, npix, opts)
%SNRFUSIONHR CP3 signal chain: per-region Kalman -> POS -> SNR-weighted spectral fusion per window.
%   [hr, tc, wMean] = snrFusionHR(t, X, npix)
%   t     Nx1 frame times (s, real timestamps)
%   X     1xR cell, each Nx3 mean RGB of one region/patch (NaN rows = invalid frames)
%   npix  1xR cell, each Nx1 pixels behind each frame mean ([] = unweighted Kalman)
%   hr    window HR (bpm) at the centres tc (s); NaN where no region qualifies
%   wMean 1xR mean normalised SNR weight per region (diagnostics)
%   Pf    nWin x nBand fused spectra (NaN rows where no region qualifies), fbOut band frequencies (Hz)
%         — used by the optional TROIKA-style tracker (src/track/troikaTrack.m)
%
%   MATLAB port of the teammate's pos_kf_snr (Nadha Irshad, branch c2-skin-tone 425fdc1,
%   contrib/snr_fusion/src/python/facevitals/rppg.py: kalman_denoise + sanity_hr_snr), adopted as
%   the Round-1 configuration CP3 on 27 Nov 2026 (docs/CHANGELOG_PARAMS.md):
%     1. per region: kalman_denoise (Thamil) on native frames, R_k ~ 1/npix, -3 dB at 4 Hz
%     2. linear resampling to a uniform fs (30 Hz) on the real timestamps
%     3. POS (1.6 s overlap-add), zero-phase Butterworth band-pass order 3, 0.7-3 Hz (no detrend)
%     4. windows of Win s every Step s (centres tu(1)+Win/2 ... tu(end)-Win/2); a window needs
%        >= 90 % of its samples; a region needs >= 90 % valid frames in the window and >= 50 %
%        valid frames overall
%     5. Hann periodogram (nfft >= 8192), normalised to unit in-band power; SNR (linear) = power
%        within +-0.1 Hz of the in-band peak and +-0.2 Hz of its harmonic / remaining in-band power
%     6. fused = sum_r SNR_r * P_r / sum(P_r in band); HR = 60 * argmax(fused) (no interpolation)
arguments
    t (:,1) double
    X cell
    npix cell = {}
    opts.Fs (1,1) double = 30
    opts.Band (1,2) double = [0.7 3.0]
    opts.Win (1,1) double = 10
    opts.Step (1,1) double = 1
    opts.Kalman (1,1) logical = true
    opts.Power (1,1) double = 1
    opts.Centers double = []
end
fs = opts.Fs; R = numel(X);
fsNative = 1 / median(diff(t(isfinite(t))));
anyValid = false(numel(t), 1);
for r = 1:R, anyValid = anyValid | all(isfinite(X{r}), 2); end
anyValid = anyValid & isfinite(t);
hr = []; tc = opts.Centers(:); wMean = NaN(1, R); Pf = []; fbOut = [];
if nnz(anyValid) < opts.Win * fs * 0.9 * (fsNative / fs), hr = NaN(numel(tc), 1); return; end
tv = t(anyValid);
tu = (tv(1) : 1/fs : tv(end) + 1e-9)';
[b, a] = butter(3, opts.Band / (fs / 2), 'bandpass');
H = cell(1, R); okU = cell(1, R);
for r = 1:R
    x = X{r}; v = all(isfinite(x), 2) & isfinite(t);
    if nnz(v) < 0.5 * nnz(anyValid), continue; end
    if opts.Kalman
        w = []; if numel(npix) >= r && ~isempty(npix{r}), w = max(npix{r}(:), 1); end
        if isempty(w), x = kalman_denoise(x, fsNative); else, x = kalman_denoise(x, fsNative, Npix=w); end
    end
    xu = interp1(t(v), x(v, :), tu, 'linear');
    xu = fillmissing(xu, 'nearest');
    H{r} = filtfilt(b, a, posPulse(xu, fs));
    okU{r} = interp1(t(isfinite(t)), double(v(isfinite(t))), tu, 'linear', 0) > 0.5;
end
if isempty(tc), tc = (tu(1) + opts.Win/2 : opts.Step : tu(end) - opts.Win/2 + 1e-9)'; end
L = round(opts.Win * fs);
nfft = 2^nextpow2(max(8192, L));
f = (0:nfft/2)' * fs / nfft; m = f >= opts.Band(1) & f <= opts.Band(2); fb = f(m);
hr = NaN(numel(tc), 1); wSum = zeros(1, R); wCnt = 0;
Pf = NaN(numel(tc), numel(fb)); fbOut = fb;
for i = 1:numel(tc)
    sel = tu >= tc(i) - opts.Win/2 & tu < tc(i) + opts.Win/2;
    nS = nnz(sel);
    if nS < 0.9 * L, continue; end
    hw = hann(nS);                                   % symmetric Hann (scipy windows.hann sym=True)
    fused = []; wAll = zeros(1, R);
    for r = 1:R
        if isempty(H{r}) || mean(okU{r}(sel)) < 0.9, continue; end
        y = H{r}(sel); y = y - mean(y);
        if any(~isfinite(y)), continue; end
        P = abs(fft(y .* hw, nfft)).^2; P = P(1:nfft/2+1); Pb = P(m);
        tot = sum(Pb); if tot <= 0, continue; end
        [~, k] = max(Pb); f0 = fb(k);
        sig = abs(fb - f0) <= 0.1 | abs(fb - 2 * f0) <= 0.2;
        snr = sum(Pb(sig)) / max(sum(Pb(~sig)), 1e-30);
        wAll(r) = snr ^ opts.Power;
        if isempty(fused), fused = wAll(r) * Pb / tot; else, fused = fused + wAll(r) * Pb / tot; end
    end
    if isempty(fused), continue; end
    [~, k] = max(fused); hr(i) = 60 * fb(k); Pf(i, :) = fused';
    if sum(wAll) > 0, wSum = wSum + wAll / sum(wAll); wCnt = wCnt + 1; end
end
if wCnt, wMean = wSum / wCnt; end
end

function h = posPulse(Xc, fs)
% POS (Wang et al. 2017), 1.6 s overlap-add; std with N-1 (as numpy ddof=1 in the port).
N = size(Xc, 1); l = round(1.6 * fs); h = zeros(N, 1);
P = [0 1 -1; -2 1 1];
for n = 1:N - l + 1
    C = Xc(n:n+l-1, :)';
    mu = mean(C, 2);
    if any(mu <= 0), continue; end
    S = P * (C ./ mu);
    s2 = std(S(2, :));
    if s2 == 0, continue; end
    pp = S(1, :) + (std(S(1, :)) / s2) * S(2, :);
    h(n:n+l-1) = h(n:n+l-1) + (pp - mean(pp))';
end
end
