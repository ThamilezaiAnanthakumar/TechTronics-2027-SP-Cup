function build_submission(opts)
%BUILD_SUBMISSION Round-1 submission folder (README 1.3) with the frozen CP3 pipeline.
%   Baseline: UBFC test subject 4 (first of the pre-specified constraint subjects), full length:
%     packageRound -> target_video.mp4 (pulse-preserving x264), reference_signal.csv,
%     processed_output.csv, Round1_Baseline_None.mat (six variables; facial_video limited by MaxFramesInMat).
%   Constraints: process_round1 on the degraded copies of the same subject ->
%     Round1_<Condition>_<Constraint>.mat + processed_output.csv in constraints/<Condition>_<Constraint>/.
%   C5 is omitted until re-encoded with ffmpeg (VideoWriter MPEG-4 destroys the pulse; docs/API_NOTES.md).
%   Code: process_round1.m, src/, config/, shared/patches.json, models/face_landmarker.task.
arguments
    opts.OutRoot (1,1) string = "D:\SPCup Data\work\submission"
    opts.Subject (1,1) string = "subject4"
    opts.MaxFramesInMat (1,1) double = 300
end
root = fileparts(fileparts(mfilename('fullpath')));
addpath(root); addpath(fullfile(root, 'config')); addpath(genpath(fullfile(root, 'src')));
cfg = getConfig('subject', char(opts.Subject)); [~, gt] = loadUBFC(cfg);
if ~isfolder(opts.OutRoot), mkdir(opts.OutRoot); end

% ---- baseline package ----
P = packageRound(opts.Subject, OutRoot=opts.OutRoot, MaxFramesInMat=opts.MaxFramesInMat, ...
    Condition="Baseline", Constraint="None");
M = load(P.mat, 'metrics');
fprintf('Baseline %s: MAE %.2f, coverage %.0f%%, %s\n', opts.Subject, M.metrics.MAE, M.metrics.coverage, P.mat);
pkg = fileparts(P.mat);

% ---- constraint .mat files ----
deg = "D:\SPCup Data\work\degraded"; s4 = opts.Subject;
C = { "LowLight",      "C1", fullfile(deg, "LowLight_C1_50lux_poisson", s4 + ".mj2");
      "MultiFace",     "C3", fullfile("D:\SPCup Data\work\wt_face_detect\data\degraded\MultiFace", s4 + ".avi");
      "LowResolution", "C4", fullfile(deg, "LowRes_C4_320x240", s4 + ".mj2");
      "Aging",         "C6", fullfile(deg, "Aging_C6_s0.1", s4 + ".mj2");
      "Masked",        "C7", fullfile(deg, "Masked_C7_surgical", s4 + ".mj2") };
rows = {"Baseline", "None", M.metrics.MAE, M.metrics.coverage};
for k = 1:size(C, 1)
    src = C{k, 3};
    if ~isfile(src), fprintf(2, 'missing %s\n', src); continue; end
    od = fullfile(pkg, "constraints", C{k, 1} + "_" + C{k, 2});
    out = process_round1(char(src), 'Reference', gt, 'OutDir', od, 'Condition', C{k, 1}, ...
        'Constraint', C{k, 2}, 'MaxFramesInMat', min(opts.MaxFramesInMat, 150));
    rows(end+1, :) = {C{k, 1}, C{k, 2}, out.metrics.MAE, out.metrics.coverage}; %#ok<AGROW>
    fprintf('%s %s: MAE %.2f, coverage %.0f%%\n', C{k, 1}, C{k, 2}, out.metrics.MAE, out.metrics.coverage);
end
T = cell2table(rows, 'VariableNames', {'Condition', 'Constraint', 'MAE', 'Coverage'});
writetable(T, fullfile(pkg, 'submission_summary.csv'));

% ---- code ----
code = fullfile(opts.OutRoot, 'code');
if ~isfolder(code), mkdir(code); end
copyfile(fullfile(root, 'process_round1.m'), code);
copyfile(fullfile(root, 'src'), fullfile(code, 'src'));
copyfile(fullfile(root, 'config'), fullfile(code, 'config'));
mkdir(fullfile(code, 'shared')); copyfile(fullfile(root, 'shared', 'patches.json'), fullfile(code, 'shared'));
mkdir(fullfile(code, 'models')); copyfile(fullfile(root, 'models', 'face_landmarker.task'), fullfile(code, 'models'));
[~, h] = system(sprintf('git -C "%s" rev-parse --short HEAD', root));
writelines(["git " + strtrim(string(h)), "frozen config: config/round1_freeze.json (CP3, tag round1-freeze-cp3)"], ...
    fullfile(code, 'VERSION.txt'));
disp(T); fprintf('== SUBMISSION BUILT: %s\n', opts.OutRoot);
end
