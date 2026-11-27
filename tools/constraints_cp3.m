function G = constraints_cp3(opts)
%CONSTRAINTS_CP3 Score the constraint suite with the frozen CP3 chain from the cached traces.
%   Uses the anatomical-patch traces cached by tools/run_constraints.m
%   (<WorkRoot>\anat_traces_constraints\30s_<Condition>_<subject>.mat; subjects 4, 17, 18, 20; 30 s)
%   and the CP3 signal chain src/pipeline/snrFusionHR.m, with the plain-chain PPG reference on the
%   same window centres (as tools/eval_ubfc_cp3.m). Writes outputs/round1/constraints_cp3_*.csv.
arguments
    opts.Subjects (1,:) double = [4 17 18 20]
    opts.Conditions (1,:) string = ["Baseline" "LowLight50" "LowRes320x240" "LowBandwidth10fps" "Aging0.1" "MaskSurgical"]
    opts.Constraints (1,:) string = ["None" "C1" "C4" "C5" "C6" "C7"]
    opts.WorkRoot (1,1) string = "D:\SPCup Data\work"
end
root = fileparts(fileparts(mfilename('fullpath')));
addpath(root); addpath(fullfile(root, 'config')); addpath(genpath(fullfile(root, 'src')));
cfg = getConfig();
rows = {};
for sNum = opts.Subjects
    s = "subject" + sNum; c = cfg; c.subject = char(s); [~, gt] = loadUBFC(c);
    for k = 1:numel(opts.Conditions)
        f = fullfile(opts.WorkRoot, "anat_traces_constraints", "30s_" + opts.Conditions(k) + "_" + s + ".mat");
        if ~isfile(f)
            rows(end+1, :) = {s, opts.Conditions(k), opts.Constraints(k), NaN, NaN, NaN, NaN, NaN, 0}; %#ok<AGROW>
            fprintf('%s %s: no cached traces\n', s, opts.Conditions(k)); continue
        end
        S = load(f, 'A'); A = S.A; nP = size(A.rgb, 3);
        X = cell(1, nP); W = cell(1, nP);
        for q = 1:nP, x = A.rgb(:, :, q); x(~A.valid(:, q), :) = NaN; X{q} = x; W{q} = A.npix(:, q); end
        [hr, tc] = snrFusionHR(A.t, X, W);
        r = refHr(gt, A.t, tc, cfg);
        m = computeMetrics(hr, r);
        rows(end+1, :) = {s, opts.Conditions(k), opts.Constraints(k), m.MAE, m.RMSE, m.SUC5, m.coverage, ...
            1 / median(diff(A.t)), numel(hr)}; %#ok<AGROW>
        fprintf('%s %-18s MAE %.2f coverage %.0f%%\n', s, opts.Conditions(k), m.MAE, m.coverage);
    end
end
T = cell2table(rows, 'VariableNames', {'Subject', 'Condition', 'Constraint', 'MAE', 'RMSE', 'SUC5', 'Coverage', 'Fps', 'nWin'});
G = groupsummary(T, {'Condition', 'Constraint'}, {'mean', 'nnz'}, {'MAE', 'RMSE', 'SUC5', 'Coverage'});
out = fullfile(root, 'outputs', 'round1');
writetable(T, fullfile(out, 'constraints_cp3_per_subject.csv')); writetable(G, fullfile(out, 'constraints_cp3_summary.csv'));
disp(G);
end

function r = refHr(gt, t, tc, cfg)
fs = 30; tu = (t(1) : 1/fs : t(end) + 1e-9)';
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
