function T = ubfc_patch_ablation(opts)
%UBFC_PATCH_ABLATION README 8.3 ablation 1 on the UBFC2 dev split: box-fraction ROIs vs anatomical patches.
%   Same signal chain for both (current best from the AiPEX loop: POS, smoothness-priors detrend,
%   Butterworth order 2, 10 s windows, per-window SNR-weighted spectral fusion), same PPG reference
%   windows (hr_results.mat of the seeded run_ubfc_all). Rows:
%     box3      Viola-Jones box-fraction ROIs (forehead, cheeks), top-3 fusion
%     anat10    10 MediaPipe anatomical patches (shared/patches.json), top-5 fusion (README 4.7)
%     anat10k   same, with Kalman denoising weighted by patch skin-pixel count
%   Anatomical traces are cached in <AnatRoot>/<subject>.mat. Writes outputs/ablation_roi/summary.csv.
arguments
    opts.Subjects (1,:) double = [1 3 5 8 9 10 11 12 13 14 15 16]
    opts.ResultsRoot (1,1) string = "D:\SPCup Data\work\ubfc2_seeded"
    opts.AnatRoot (1,1) string = "D:\SPCup Data\work\anat_traces"
end
root = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root, 'config')); addpath(genpath(fullfile(root, 'src')));
cfg = getConfig('filterOrder', 2);
variants = ["box3" "anat10" "anat10k"];
est = struct(); for v = variants, est.(v) = []; end
refAll = []; perSub = NaN(numel(opts.Subjects), numel(variants));
for s = 1:numel(opts.Subjects)
    subj = "subject" + opts.Subjects(s);
    L = load(fullfile(opts.ResultsRoot, subj, 'traces.mat'), 'traces'); tr = L.traces;
    H = load(fullfile(opts.ResultsRoot, subj, 'hr_results.mat'), 'hrRef');
    fs = tr.fps; N = numel(tr.t);
    c = cfg; c.subject = char(subj); [vp, ~] = loadUBFC(c);
    A = anatomicalPatchTraces(vp, CacheFile=fullfile(opts.AnatRoot, subj + ".mat"));
    nA = min(N, numel(A.t));
    % box-fraction ROIs
    X = {tr.rgb.forehead, tr.rgb.leftCheek, tr.rgb.rightCheek};
    hrBox = fusedHr(X, cellfun(@(x) ones(size(x, 1), 1), X, 'UniformOutput', false), ...
        cellfun(@(x) true(size(x, 1), 1), X, 'UniformOutput', false), fs, 3, false, cfg);
    % anatomical patches (first nA frames, same 1/fps grid as the box traces)
    nP = size(A.rgb, 3);
    Xa = arrayfun(@(q) A.rgb(1:nA, :, q), 1:nP, 'UniformOutput', false);
    Wa = arrayfun(@(q) A.npix(1:nA, q), 1:nP, 'UniformOutput', false);
    Va = arrayfun(@(q) A.valid(1:nA, q), 1:nP, 'UniformOutput', false);
    hrA = fusedHr(Xa, Wa, Va, fs, 5, false, cfg);
    hrAk = fusedHr(Xa, Wa, Va, fs, 5, true, cfg);
    m = min([numel(hrBox), numel(hrA), numel(H.hrRef)]);
    ref = H.hrRef(1:m); refAll = [refAll; ref]; %#ok<AGROW>
    hs = {hrBox(1:m), hrA(1:m), hrAk(1:m)};
    for v = 1:numel(variants)
        est.(variants(v)) = [est.(variants(v)); hs{v}];
        perSub(s, v) = mean(abs(hs{v} - ref), 'omitnan');
    end
    fprintf('%s: box3 %.2f | anat10 %.2f | anat10k %.2f (%d windows)\n', subj, perSub(s, :), m);
end
rows = cell(numel(variants), 6);
for v = 1:numel(variants)
    mm = computeMetrics(est.(variants(v)), refAll);
    rows(v, :) = {variants(v), mm.MAE, mm.RMSE, mm.r, mm.SUC5, mm.coverage};
end
T = cell2table(rows, 'VariableNames', {'Variant', 'PooledMAE', 'RMSE', 'r', 'SUC5', 'Coverage'});
P = array2table(perSub, 'VariableNames', cellstr(variants)); P.Subject = "subject" + opts.Subjects(:);
outDir = fullfile(root, 'outputs', 'ablation_roi'); if ~isfolder(outDir), mkdir(outDir); end
writetable(T, fullfile(outDir, 'summary.csv')); writetable(P, fullfile(outDir, 'per_subject.csv'));
disp(T); disp(P);
end

function hr = fusedHr(X, W, V, fs, topK, useKalman, cfg)
nR = numel(X); Pn = cell(1, nR); snr = []; ok = false(1, nR);
for r = 1:nR
    x = X{r}; v = V{r} & all(isfinite(x), 2);
    if nnz(v) < 0.5 * numel(v), continue; end
    x(~v, :) = NaN;
    x = fillmissing(x, 'linear', 'EndValues', 'nearest');
    if useKalman, x = kalman_denoise(x, fs, Npix=max(W{r}, 1)); end
    pf = preprocessPulse(rgbToPulse(x, fs, 'POS', cfg), fs, cfg);
    [~, ~, s] = estimateHR(pf, fs, cfg);
    Pn{r} = s.P ./ sum(s.P, 2); f = s.f; ok(r) = true;
    L = s.winLen; cov = arrayfun(@(st) mean(v(st:st+L-1)), s.starts);
    snr(:, r) = arrayfun(@(w) snrDb(Pn{r}(w, :), f), (1:size(s.P, 1))'); %#ok<AGROW>
    snr(cov < cfg.minValidFrac, r) = -Inf; %#ok<AGROW>
end
idx = find(ok); nW = size(Pn{idx(1)}, 1); hr = NaN(nW, 1);
for w = 1:nW
    s = snr(w, idx); wt = max(10.^(s / 10) - 1, 0);
    if all(wt == 0), wt = double(isfinite(s)); end
    if all(wt == 0), continue; end
    [~, o] = sort(wt, 'descend'); o = o(1:min(topK, numel(o)));
    m = zeros(1, numel(f));
    for j = o, m = m + wt(j) * Pn{idx(j)}(w, :); end
    [~, k] = max(m); d = 0;
    if k > 1 && k < numel(m), a = m(k-1); b = m(k); c = m(k+1); den = a - 2*b + c; if den ~= 0, d = 0.5*(a-c)/den; end, end
    hr(w) = 60 * (f(k) + d * (f(2) - f(1)));
end
end

function s = snrDb(m, f)
[~, k] = max(m); f0 = f(k); bw = 6 / 60;
sig = abs(f - f0) <= bw | abs(f - 2 * f0) <= bw;
s = 10 * log10(sum(m(sig)) / max(sum(m(~sig)), eps));
end
