function T = run_constraints(opts)
%RUN_CONSTRAINTS C1/C4/C5/C6/C7 suite with the frozen Round-1 pipeline (README 5), after the freeze.
%   Subjects fixed in advance (first six UBFC2 test subjects by number): 4, 17, 18, 20, 22, 23.
%   For each subject and condition: generate the degraded copy (src/degrade, official recipes,
%   JSON sidecar), extract anatomical traces, run runOffline (= process_round1) against the
%   contact-PPG reference. Baseline uses the original video. Writes outputs/round1/constraints_*.csv.
arguments
    opts.Subjects (1,:) double = [4 17 18 20 22 23]
    opts.WorkRoot (1,1) string = "D:\SPCup Data\work"
end
root = fileparts(fileparts(mfilename('fullpath')));
addpath(root); addpath(fullfile(root, 'config')); addpath(genpath(fullfile(root, 'src')));
cfg = getConfig();
degRoot = fullfile(opts.WorkRoot, "degraded");
traceRoot = fullfile(opts.WorkRoot, "anat_traces_constraints");
if ~isfolder(traceRoot), mkdir(traceRoot); end
C = { ...
    "Baseline",          "None", @(vp, s) struct('output', struct('path', vp));
    "LowLight50",        "C1",   @(vp, s) lowLight(vp, Lux=50, Noise="poisson", OutRoot=degRoot, Subject=s);
    "LowLight10",        "C1",   @(vp, s) lowLight(vp, Lux=10, Noise="poisson", OutRoot=degRoot, Subject=s);
    "LowRes320x240",     "C4",   @(vp, s) lowRes(vp, Size=[240 320], OutRoot=degRoot, Subject=s);
    "LowBandwidth10fps", "C5",   @(vp, s) lowBandwidth(vp, Fps=10, BitrateKbps=500, OutRoot=degRoot, Subject=s);
    "Aging0.1",          "C6",   @(vp, s) aging(vp, Strength=0.1, OutRoot=degRoot, Subject=s);
    "MaskSurgical",      "C7",   @(vp, s) mask(vp, Variant="surgical", OutRoot=degRoot, Subject=s)};
rows = {};
for sNum = opts.Subjects
    s = "subject" + sNum;
    c = cfg; c.subject = char(s); [vp, gt] = loadUBFC(c);
    for k = 1:size(C, 1)
        cond = C{k, 1}; con = C{k, 2};
        try
            t0 = tic;
            if cond == "Baseline"
                cache = fullfile(traceRoot, "Baseline_" + s + ".mat"); src = vp;   % own cache (polygons kept)
            else
                info = C{k, 3}(vp, s); src = string(info.output.path);
                cache = fullfile(traceRoot, cond + "_" + s + ".mat");
            end
            R = runOffline(src, Reference=gt, CacheFile=cache);
            m = R.metrics;
            rows(end+1, :) = {s, cond, con, m.MAE, m.RMSE, m.r, m.SUC5, m.coverage, m.ungated.MAE, ...
                m.pct_windows_masked, R.info.fps_measured, toc(t0)}; %#ok<AGROW>
            fprintf('%s %-18s MAE %.2f (ungated %.2f) cov %.0f%% masked %.0f%% (%.0f s)\n', s, cond, m.MAE, ...
                m.ungated.MAE, m.coverage, m.pct_windows_masked, toc(t0));
        catch e
            rows(end+1, :) = {s, cond, con, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN}; %#ok<AGROW>
            fprintf(2, '%s %s FAILED: %s\n', s, cond, e.message);
        end
    end
end
T = cell2table(rows, 'VariableNames', {'Subject', 'Condition', 'Constraint', 'MAE', 'RMSE', 'r', 'SUC5', ...
    'Coverage', 'MAE_ungated', 'PctMasked', 'FpsMeasured', 'Seconds'});
out = fullfile(root, 'outputs', 'round1'); if ~isfolder(out), mkdir(out); end
writetable(T, fullfile(out, 'constraints_per_subject.csv'));
G = groupsummary(T, {'Condition', 'Constraint'}, 'mean', {'MAE', 'RMSE', 'SUC5', 'Coverage', 'MAE_ungated', 'PctMasked'});
writetable(G, fullfile(out, 'constraints_summary.csv'));
disp(G);
fprintf('== CONSTRAINTS DONE\n');
end
