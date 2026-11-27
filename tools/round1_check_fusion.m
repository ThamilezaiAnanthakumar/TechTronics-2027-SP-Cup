function T = round1_check_fusion(opts)
%ROUND1_CHECK_FUSION Regression: runOffline (Traces=cached anatomical traces) must reproduce the
%   anat10 row of tools/ubfc_patch_ablation.m (UBFC dev, pooled MAE 1.37 bpm) on the same
%   PPG reference windows (hr_results.mat of the seeded run). Dev subjects only, cached data only.
arguments
    opts.Subjects (1,:) double = [1 3 5 8 9 10 11 12 13 14 15 16]
    opts.ResultsRoot (1,1) string = "D:\SPCup Data\work\ubfc2_seeded"
    opts.AnatRoot (1,1) string = "D:\SPCup Data\work\anat_traces"
end
assert(all(ismember(opts.Subjects, [1 3 5 8:16])), 'dev subjects only');
root = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root, 'config')); addpath(genpath(fullfile(root, 'src')));
cfg = getConfig('filterOrder', 2);
ab = readtable(fullfile(root, 'outputs', 'ablation_roi', 'per_subject.csv'));
est = []; estG = []; refAll = []; rows = zeros(numel(opts.Subjects), 3);
for s = 1:numel(opts.Subjects)
    subj = "subject" + opts.Subjects(s);
    L = load(fullfile(opts.ResultsRoot, subj, 'traces.mat'), 'traces'); N = numel(L.traces.t);
    H = load(fullfile(opts.ResultsRoot, subj, 'hr_results.mat'), 'hrRef');
    S = load(fullfile(opts.AnatRoot, subj + ".mat"), 'A'); A = S.A;
    n = min(N, numel(A.t));
    A.t = A.t(1:n); A.rgb = A.rgb(1:n, :, :); A.npix = A.npix(1:n, :); A.valid = A.valid(1:n, :);
    R = runOffline("", Traces=A, Config=cfg);
    m = min(numel(R.processed_signal.hr_raw), numel(H.hrRef));
    e = R.processed_signal.hr_raw(1:m); g = R.processed_signal.hr_bpm(1:m); r = H.hrRef(1:m);
    est = [est; e]; estG = [estG; g]; refAll = [refAll; r]; %#ok<AGROW>
    rows(s, :) = [mean(abs(e - r), 'omitnan'), ab.anat10(ab.Subject == subj), 100 * mean(R.processed_signal.masked)];
    fprintf('%s: runOffline %.4f | ablation anat10 %.4f | fs %.3f | masked %.0f %% | status OK %d/%d\n', ...
        subj, rows(s, 1), rows(s, 2), R.info.fs_proc, rows(s, 3), nnz(strcmp(R.processed_signal.status, 'OK')), numel(g));
end
mm = computeMetrics(est, refAll); mg = computeMetrics(estG, refAll);
fprintf('pooled MAE ungated %.3f (ablation 1.37), gated %.3f with coverage %.1f %%\n', mm.MAE, mg.MAE, mg.coverage);
T = array2table(rows, 'VariableNames', {'runOffline', 'ablation', 'pctMasked'});
T.Subject = "subject" + opts.Subjects(:);
end
