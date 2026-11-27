function T = eval_ubfc_cp2(split)
%EVAL_UBFC_CP2 Frozen Round-1 pipeline (runOffline = process_round1 code path) on a UBFC2 split.
%   T = eval_ubfc_cp2("dev")    subjects 1, 3, 5, 8-16
%   T = eval_ubfc_cp2("test")   the other 30 subjects — run ONCE, after the freeze (README 8.2)
%   Uses cached anatomical traces (D:\SPCup Data\work\anat_traces). Pooled metrics over 10 s windows
%   (gated output = HR only where status OK/MASKED; ungated = every window) + per-subject MAE.
%   Writes outputs/round1/ubfc_<split>_per_subject.csv and _summary.csv.
arguments
    split (1,1) string {mustBeMember(split, ["dev" "test"])}
end
root = fileparts(fileparts(mfilename('fullpath')));
addpath(root); addpath(fullfile(root, 'config')); addpath(genpath(fullfile(root, 'src')));
cfg = getConfig();
dev = "subject" + [1 3 5 8 9 10 11 12 13 14 15 16];
if split == "dev", subs = dev; else, subs = setdiff(string(listUBFCSubjects(cfg)), dev, 'stable'); end
est = []; estU = []; ref = []; rows = {};
for s = subs
    c = cfg; c.subject = char(s); [vp, gt] = loadUBFC(c);
    S = load(fullfile("D:\SPCup Data\work\anat_traces", s + ".mat"), 'A');
    R = runOffline(vp, Reference=gt, Traces=S.A);
    h = R.processed_signal.hr_bpm(:); hu = R.processed_signal.hr_raw(:); r = R.reference.hr_win(:);
    n = min([numel(h), numel(hu), numel(r)]);
    est = [est; h(1:n)]; estU = [estU; hu(1:n)]; ref = [ref; r(1:n)]; %#ok<AGROW>
    m = computeMetrics(h(1:n), r(1:n)); mu = computeMetrics(hu(1:n), r(1:n));
    rows(end+1, :) = {s, m.MAE, mu.MAE, m.coverage, n}; %#ok<AGROW>
    fprintf('%s: MAE %.2f (ungated %.2f), coverage %.0f%%, %d windows\n', s, m.MAE, mu.MAE, m.coverage, n);
end
P = cell2table(rows, 'VariableNames', {'Subject', 'MAE', 'MAE_ungated', 'Coverage', 'nWin'});
g = computeMetrics(est, ref); u = computeMetrics(estU, ref);
T = table(["gated"; "ungated"], [g.MAE; u.MAE], [g.RMSE; u.RMSE], [g.r; u.r], [g.bias; u.bias], ...
    [g.loaLow; u.loaLow], [g.loaHigh; u.loaHigh], [g.SUC3; u.SUC3], [g.SUC5; u.SUC5], [g.coverage; u.coverage], ...
    [g.nWindows; u.nWindows], [median(P.MAE, 'omitnan'); median(P.MAE_ungated, 'omitnan')], ...
    'VariableNames', {'Output', 'PooledMAE', 'RMSE', 'r', 'bias', 'LoA_low', 'LoA_high', 'SUC3', 'SUC5', ...
    'Coverage', 'nWin', 'MedianSubjectMAE'});
out = fullfile(root, 'outputs', 'round1'); if ~isfolder(out), mkdir(out); end
writetable(P, fullfile(out, "ubfc_" + split + "_per_subject.csv"));
writetable(T, fullfile(out, "ubfc_" + split + "_summary.csv"));
disp(T);
end
