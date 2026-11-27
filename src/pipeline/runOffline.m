function R = runOffline(videoPath, opts)
%RUNOFFLINE Offline FV-TT pipeline on one video (README 4, 9): HR per second + ROI data + metrics.
%   R = runOffline(videoPath)
%   R = runOffline(videoPath, Reference=gt, Config=cfg, MaxFrames=900, Tracker="viterbi")
%   R = runOffline("", Traces=A, ...)              precomputed patch traces (tests, caches)
%   Chain (current best, frozen; UBFC-dev pooled MAE 1.37 bpm, tools/ubfc_patch_ablation.m):
%     anatomicalPatchTraces (MediaPipe 478 landmarks -> 10 patches of shared/patches.json,
%     1-Euro, skin + specular mask, trimmed mean) -> uniform grid fs = min(30, measured fps)
%     from the real timestamps -> per patch POS -> preprocessPulse (cfg.filterOrder = 2) ->
%     estimateHR spectra (10 s windows, 1 s step) -> fusePatches (per-window SNR-weighted
%     top-5 spectral fusion, status codes, confidence). Tracker="viterbi" replaces the
%     per-window argmax by viterbiHR on the fused spectra (optional, not the default).
%   Reference (optional): struct with t (s) and ppg (contact PPG) and/or hrSensor / hr_bpm.
%     Window reference HR = referenceHR (plain detrend -> BPF -> FFT peak on the same window
%     grid, no fusion, no tracker; README 4.10). Without ppg: mean reference HR per window.
%   R.processed_signal  t (window centres, s), hr_bpm, confidence, status (cellstr),
%                       masked, n_patches_used, top_patches, snr_db, hr_raw (ungated),
%                       bvp / t_bvp / fs (waveform fusion), window_sec, step_sec
%   R.roi_data          per frame: t, polygons (N x P x Kmax x 2 single, NaN padded), valid,
%                       npix, weights (weight of the nearest window); names, groups,
%                       window_t, window_weights
%   R.metrics           (with a reference) computeMetrics on the reported HR (gated) +
%                       .ungated, .vs_sensor, window protocol
%   R.reference         window reference HR and the raw reference
%   R.info              measured fps, resolution, frame count, ITA estimate
arguments
    videoPath (1,1) string
    opts.Reference = []
    opts.Config struct = struct()
    opts.MaxFrames (1,1) double = Inf
    opts.Traces struct = struct()
    opts.Tracker (1,1) string {mustBeMember(opts.Tracker, ["none" "viterbi"])} = "none"
    opts.TopK (1,1) double = 5
    opts.LowSnrDb (1,1) double = -3
    opts.MaskedFrac (1,1) double = 0.4
    opts.CacheFile (1,1) string = ""
end
cfg = opts.Config;
if isempty(fieldnames(cfg)), cfg = getConfig('filterOrder', 2); end

% ---- M1-M3: anatomical patch traces --------------------------------------------------
if isempty(fieldnames(opts.Traces))
    A = anatomicalPatchTraces(videoPath, MaxFrames=opts.MaxFrames, KeepPolygons=true, CacheFile=opts.CacheFile);
else
    A = opts.Traces;
end
nF = numel(A.t); nP = size(A.rgb, 3);
groups = string(A.groups); names = string(A.names);

% ---- M4: uniform grid from the real timestamps ----------------------------------------
dtm = median(diff(A.t));
fsMeas = 1 / dtm;
fs = min(30, fsMeas);
M = floor((A.t(end) - A.t(1)) * fs + 1e-6) + 1;
tu = A.t(1) + (0:M-1)' / fs;
X = cell(1, nP); V = cell(1, nP);
for q = 1:nP
    v = A.valid(:, q) & all(isfinite(A.rgb(:, :, q)), 2);
    vu = interp1(A.t, double(v), tu, 'linear', 0) >= 1 - 1e-9;  % both neighbours valid
    x = NaN(M, 3);
    if nnz(v) >= 2, x = interp1(A.t(v), A.rgb(v, :, q), tu, 'linear'); end
    x(~vu, :) = NaN;
    X{q} = x; V{q} = vu;
end

% ---- M5-M8: per-patch POS, quality, fusion, status -------------------------------------
F = fusePatches(X, V, fs, cfg, TopK=opts.TopK, Groups=groups, MaskedFrac=opts.MaskedFrac, LowSnrDb=opts.LowSnrDb);
hr = F.hr; conf = F.conf;
if opts.Tracker == "viterbi" && ~isempty(F.f)
    [hv, cv] = viterbiHR(F.Pfused + (sum(F.Pfused, 2) == 0), F.f);   % empty windows -> flat
    use = F.status == "OK" | F.status == "MASKED";
    hr(use) = hv(use); conf(use) = cv(use);
end
nW = numel(F.t);

% ---- waveform fusion (bvp): global weights, sign/lag alignment to the best patch ------
wG = mean(F.weights, 1); wG(~F.patchOk) = 0;
bvp = NaN(M, 1);
if any(wG > 0)
    [~, ord] = sort(wG, 'descend'); ord = ord(wG(ord) > 0); ord = ord(1:min(opts.TopK, numel(ord)));
    ref = F.pulses(:, ord(1)); acc = zeros(M, 1); maxLag = round(0.25 * fs);
    for q = ord
        [c, lags] = xcorr(ref, F.pulses(:, q), maxLag, 'coeff');
        [~, j] = max(abs(c)); y = circshift(F.pulses(:, q), lags(j)) * sign(c(j));
        acc = acc + wG(q) * y;
    end
    bvp = (acc - mean(acc)) / max(std(acc), eps);
end

% ---- outputs ---------------------------------------------------------------------------
P.t = F.t + tu(1);
P.hr_bpm = hr; P.confidence = conf; P.status = cellstr(F.status); P.masked = F.masked;
P.n_patches_used = F.nUsed; P.top_patches = F.top; P.snr_db = F.snrDb; P.hr_raw = F.hrRaw;
P.bvp = bvp; P.t_bvp = tu; P.fs = fs;
P.window_sec = cfg.hrWindowSec; P.step_sec = cfg.hrStepSec;
P.tracker = char(opts.Tracker); P.patch_names = cellstr(names);
R.processed_signal = P;

kMax = 0;
if isfield(A, 'poly'), kMax = max(cellfun(@(p) size(p, 1), A.poly), [], 'all'); end
poly = NaN(nF, nP, max(kMax, 1), 2, 'single');
if isfield(A, 'poly')
    for k = 1:nF
        for q = 1:nP
            p = A.poly{k, q};
            if ~isempty(p), poly(k, q, 1:size(p, 1), :) = reshape(single(p), 1, 1, [], 2); end
        end
    end
end
[~, wi] = min(abs(A.t - P.t'), [], 2);                       % nearest window per frame
D.t = A.t; D.names = cellstr(names); D.groups = cellstr(groups);
D.polygons = poly; D.polygon_format = 'frame x patch x vertex x [x y], image pixels (1-based), NaN padded';
D.valid = A.valid; D.npix = A.npix; D.weights = F.weights(wi, :);
D.window_t = P.t; D.window_weights = F.weights; D.window_coverage = F.coverage;
D.source = 'MediaPipe FaceLandmarker 478 pts -> shared/patches.json polygons (Li et al. 2026)';
R.roi_data = D;

% ITA (README 5, C2) from the median glabella + malar colour
sel = ismember(names, ["glabella" "malar_L" "malar_R"]);
rgbMed = median(reshape(permute(A.rgb(:, :, sel), [1 3 2]), [], 3), 1, 'omitnan');
ita = NaN;
if all(isfinite(rgbMed)), lab = rgb2lab(rgbMed / 255); ita = atan2(lab(1) - 50, lab(3)) * 180 / pi; end
R.info = struct('fps_measured', fsMeas, 'fs_proc', fs, 'num_frames', nF, 'num_windows', nW, ...
    'ita_deg', ita, 'video', char(videoPath));

% ---- reference + metrics ---------------------------------------------------------------
R.metrics = struct(); R.reference = struct();
if ~isempty(opts.Reference)
    g = opts.Reference;
    tr = struct('t', tu, 'fps', fs);
    if isfield(g, 'ppg') && ~isempty(g.ppg) && any(isfinite(g.ppg))
        if ~isfield(g, 'hrSensor')
            if isfield(g, 'hr_bpm'), g.hrSensor = g.hr_bpm; else, g.hrSensor = NaN(size(g.ppg)); end
        end
        [~, hrRef, ref] = referenceHR(g, tr, cfg);
        hrSensor = ref.hrSensor(:); src = 'contact PPG, plain chain (referenceHR: detrend -> BPF -> FFT peak), same window grid';
    else
        if isfield(g, 'hrSensor'), h = g.hrSensor; else, h = g.hr_bpm; end
        hu = interp1(g.t, h, tu, 'linear', 'extrap');
        hrRef = arrayfun(@(st) mean(hu(st:st+F.winLen-1)), F.starts); hrSensor = hrRef;
        src = 'reference HR series, mean per window (no PPG given)';
    end
    hrRef = hrRef(1:nW);
    m = computeMetrics(hr, hrRef);
    m.ungated = computeMetrics(F.hrRaw, hrRef);
    m.vs_sensor = computeMetrics(hr, hrSensor(1:nW));
    m.window_protocol = sprintf('%g s windows, %g s step (README 7 primary)', cfg.hrWindowSec, cfg.hrStepSec);
    m.reference = src;
    m.pct_windows_masked = 100 * mean(F.masked);
    m.status_counts = statusCounts(F.status);
    R.metrics = m;
    R.reference = struct('t_win', P.t, 'hr_win', hrRef, 'hr_sensor_win', hrSensor(1:nW), 'source', src);
end
end

function s = statusCounts(st)
s = struct();
for c = ["OK" "MASKED" "NO_FACE" "LOW_SNR" "LOW_COVERAGE"], s.(c) = nnz(st == c); end
end
