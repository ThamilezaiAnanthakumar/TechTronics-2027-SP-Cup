function T = eval_ubfc_cp3(split)
%EVAL_UBFC_CP3 CP3 (port of the teammate's pos_kf_snr, src/pipeline/snrFusionHR.m) on a UBFC2 split.
%   T = eval_ubfc_cp3("dev")   subjects 1, 3, 5, 8-16
%   T = eval_ubfc_cp3("test")  the other 30 — run ONCE after the CP3 re-freeze
%   Front end: cached anatomical-patch traces (MediaPipe, 10 patches, trimmed-mean skin pixels).
%   Reference: contact PPG interpolated to the same 30 Hz grid, plain chain (preprocessPulse:
%   detrend + band-pass, no fusion/Kalman), periodogram peak in the same 10 s windows (same centres).
%   Writes outputs/round1/ubfc_<split>_cp3_per_subject.csv and _summary.csv.
arguments
    split (1,1) string {mustBeMember(split, ["dev" "test"])}
end
root = fileparts(fileparts(mfilename('fullpath')));
addpath(root); addpath(fullfile(root, 'config')); addpath(genpath(fullfile(root, 'src')));
cfg = getConfig();
dev = "subject" + [1 3 5 8 9 10 11 12 13 14 15 16];
if split == "dev", subs = dev; else, subs = setdiff(string(listUBFCSubjects(cfg)), dev, 'stable'); end
est = []; ref = []; rows = {};
for s = subs
    c = cfg; c.subject = char(s); [~, gt] = loadUBFC(c);
    S = load(fullfile("D:\SPCup Data\work\anat_traces", s + ".mat"), 'A'); A = S.A;
    nP = size(A.rgb, 3);
    X = cell(1, nP); W = cell(1, nP);
    for q = 1:nP
        x = A.rgb(:, :, q); x(~A.valid(:, q), :) = NaN; X{q} = x; W{q} = A.npix(:, q);
    end
    [hr, tc] = snrFusionHR(A.t, X, W);
    r = refHr(gt, A.t, tc, cfg);
    est = [est; hr]; ref = [ref; r]; %#ok<AGROW>
    m = computeMetrics(hr, r);
    rows(end+1, :) = {s, m.MAE, m.RMSE, m.coverage, numel(hr)}; %#ok<AGROW>
    fprintf('%s: MAE %.2f, RMSE %.2f, coverage %.0f%%, %d windows\n', s, m.MAE, m.RMSE, m.coverage, numel(hr));
end
P = cell2table(rows, 'VariableNames', {'Subject', 'MAE', 'RMSE', 'Coverage', 'nWin'});
g = computeMetrics(est, ref);
T = table(g.MAE, g.RMSE, g.r, g.bias, g.loaLow, g.loaHigh, g.SUC3, g.SUC5, g.coverage, g.nWindows, ...
    median(P.MAE, 'omitnan'), 'VariableNames', {'PooledMAE', 'RMSE', 'r', 'bias', 'LoA_low', 'LoA_high', ...
    'SUC3', 'SUC5', 'Coverage', 'nWin', 'MedianSubjectMAE'});
out = fullfile(root, 'outputs', 'round1'); if ~isfolder(out), mkdir(out); end
writetable(P, fullfile(out, "ubfc_" + split + "_cp3_per_subject.csv"));
writetable(T, fullfile(out, "ubfc_" + split + "_cp3_summary.csv"));
disp(T);
end

function r = refHr(gt, t, tc, cfg)
fs = 30;
tu = (t(1) : 1/fs : t(end) + 1e-9)';
[tg, ~, ic] = unique(gt.t); ppg = accumarray(ic, gt.ppg, [], @mean);
p = fillmissing(interp1(tg, ppg, tu, 'linear'), 'nearest');
pf = preprocessPulse(p, fs, cfg);
L = round(10 * fs); nfft = 2^nextpow2(max(8192, L));
f = (0:nfft/2)' * fs / nfft; m = f >= cfg.hrBand(1) & f <= cfg.hrBand(2); fb = f(m);
r = NaN(numel(tc), 1);
for i = 1:numel(tc)
    sel = tu >= tc(i) - 5 & tu < tc(i) + 5;
    if nnz(sel) < 0.9 * L, continue; end
    y = pf(sel) - mean(pf(sel));
    P = abs(fft(y .* hann(nnz(sel)), nfft)).^2; P = P(1:nfft/2+1);
    [~, k] = max(P(m)); r(i) = 60 * fb(k);
end
end
