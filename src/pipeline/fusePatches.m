function F = fusePatches(X, V, fs, cfg, opts)
%FUSEPATCHES Per-window SNR-weighted top-k spectral fusion of patch traces (README 4.6-4.8).
%   F = fusePatches(X, V, fs, cfg)
%   F = fusePatches(X, V, fs, cfg, TopK=5, Groups=g, MaskedFrac=0.4, LowSnrDb=-3, NoFaceFrac=0.5)
%   X  1xP cell of Nx3 RGB traces on a uniform grid (fs Hz); V 1xP cell of Nx1 validity.
%   Chain per patch (identical to fusedHr in tools/ubfc_patch_ablation.m, UBFC-dev MAE 1.37):
%   invalid samples -> linear fill, POS (rgbToPulse), preprocessPulse, estimateHR spectra
%   normalised to unit in-band energy, window SNR (+-6 bpm around the peak and its harmonic),
%   SNR = -Inf where window coverage < cfg.minValidFrac. Weight = max(SNR_lin - 1, 0); if all
%   weights are 0 the finite-SNR patches get weight 1. Top-k by weight, weighted spectral sum,
%   parabolic peak -> HR.
%   Additions (do not change the HR of unmasked windows):
%     MASKED   mean over group-C patches of (1 - window coverage) >= MaskedFrac: group-C
%              patches are excluded from the fusion (README 4.3.5: fuse A/B only).
%     status   per window: NO_FACE (share of frames with any valid patch < NoFaceFrac),
%              LOW_COVERAGE (no patch reaches cfg.minValidFrac), LOW_SNR (SNR of the fused
%              spectrum < LowSnrDb), MASKED, OK. F.hr is NaN unless OK / MASKED; F.hrRaw keeps
%              the ungated fused estimate.
%     conf     share of fused in-band power within +-6 bpm of the peak and its harmonic,
%              = SNR_lin / (1 + SNR_lin) in [0, 1].
%   F also holds the per-patch window weights / coverage / SNR, the top-k indices, the fused
%   spectra (for viterbiHR) and the preprocessed patch pulses (for the waveform fusion).
arguments
    X cell
    V cell
    fs (1,1) double
    cfg struct
    opts.TopK (1,1) double = 5
    opts.Groups string = strings(1, 0)
    opts.MaskedFrac (1,1) double = 0.4
    opts.LowSnrDb (1,1) double = -3
    opts.NoFaceFrac (1,1) double = 0.5
end
nR = numel(X); N = size(X{1}, 1);
Pn = cell(1, nR); pulses = NaN(N, nR); ok = false(1, nR);
snr = []; cov = []; f = []; s = [];
for r = 1:nR
    x = X{r}; v = V{r}(:) & all(isfinite(x), 2);
    if nnz(v) < 0.5 * numel(v), continue; end
    x(~v, :) = NaN;
    x = fillmissing(x, 'linear', 'EndValues', 'nearest');
    pf = preprocessPulse(rgbToPulse(x, fs, 'POS', cfg), fs, cfg);
    [~, ~, s] = estimateHR(pf, fs, cfg);
    Pn{r} = s.P ./ sum(s.P, 2); f = s.f; ok(r) = true; pulses(:, r) = pf;
    L = s.winLen;
    cov(:, r) = arrayfun(@(st) mean(v(st:st+L-1)), s.starts); %#ok<AGROW>
    snr(:, r) = arrayfun(@(w) snrDb(Pn{r}(w, :), f), (1:size(s.P, 1))'); %#ok<AGROW>
    snr(cov(:, r) < cfg.minValidFrac, r) = -Inf; %#ok<AGROW>
end
L = round(cfg.hrWindowSec * fs); S = max(1, round(cfg.hrStepSec * fs));
starts = (1:S:(N - L + 1))'; nW = numel(starts);
F.t = (starts - 1 + (L - 1) / 2) / fs;                 % window centres, s from sample 1
F.starts = starts; F.winLen = L;
% per-patch coverage for every patch (also the skipped ones) -> MASKED / NO_FACE tests
covAll = zeros(nW, nR); anyValid = false(N, 1);
for r = 1:nR
    v = V{r}(:) & all(isfinite(X{r}), 2); anyValid = anyValid | v;
    covAll(:, r) = arrayfun(@(st) mean(v(st:st+L-1)), starts);
end
faceFrac = arrayfun(@(st) mean(anyValid(st:st+L-1)), starts);
isC = false(1, nR);
if ~isempty(opts.Groups), isC = opts.Groups == "C"; end
masked = false(nW, 1);
if any(isC), masked = mean(1 - covAll(:, isC), 2) >= opts.MaskedFrac; end

F.hr = NaN(nW, 1); F.hrRaw = NaN(nW, 1); F.snrDb = -Inf(nW, 1); F.conf = zeros(nW, 1);
F.weights = zeros(nW, nR); F.top = NaN(nW, opts.TopK); F.nUsed = zeros(nW, 1);
F.status = repmat("LOW_COVERAGE", nW, 1); F.masked = masked; F.faceFrac = faceFrac;
F.coverage = covAll; F.snrPatch = -Inf(nW, nR); F.f = f; F.Pfused = zeros(nW, numel(f));
idx = find(ok);
if ~isempty(idx), F.snrPatch(:, idx) = snr(:, idx); end
for w = 1:nW
    if faceFrac(w) < opts.NoFaceFrac, F.status(w) = "NO_FACE"; continue; end
    if isempty(idx), continue; end
    sw = snr(w, idx);
    if masked(w), sw(isC(idx)) = -Inf; end
    wt = max(10.^(sw / 10) - 1, 0);
    if all(wt == 0), wt = double(isfinite(sw)); end
    if all(wt == 0), continue; end                       % LOW_COVERAGE
    [~, o] = sort(wt, 'descend'); o = o(1:min(opts.TopK, numel(o)));
    o = o(wt(o) > 0);
    m = zeros(1, numel(f));
    for j = o, m = m + wt(j) * Pn{idx(j)}(w, :); end
    [~, k] = max(m); d = 0;
    if k > 1 && k < numel(m)
        a = m(k-1); b = m(k); c = m(k+1); den = a - 2*b + c;
        if den ~= 0, d = 0.5 * (a - c) / den; end
    end
    F.hrRaw(w) = 60 * (f(k) + d * (f(2) - f(1)));
    F.weights(w, idx(o)) = wt(o) / sum(wt(o));
    F.top(w, 1:numel(o)) = idx(o); F.nUsed(w) = numel(o);
    F.Pfused(w, :) = m / sum(m);
    F.snrDb(w) = snrDb(F.Pfused(w, :), f);
    sl = 10^(F.snrDb(w) / 10); F.conf(w) = sl / (1 + sl);
    if F.snrDb(w) < opts.LowSnrDb
        F.status(w) = "LOW_SNR";
    elseif masked(w)
        F.status(w) = "MASKED"; F.hr(w) = F.hrRaw(w);
    else
        F.status(w) = "OK"; F.hr(w) = F.hrRaw(w);
    end
end
F.pulses = pulses; F.patchOk = ok;
end

function s = snrDb(m, f)
[~, k] = max(m); f0 = f(k); bw = 6 / 60;
sig = abs(f - f0) <= bw | abs(f - 2 * f0) <= bw;
s = 10 * log10(sum(m(sig)) / max(sum(m(~sig)), eps));
end
