function S = round1_integration_ubfc(opts)
%ROUND1_INTEGRATION_UBFC process_round1 end to end on the first 30 s of UBFC dev subject 1.
%   (a) packageRound: MPEG-4 target_video + reference_signal.csv -> process_round1 -> CSV + .mat
%   (b) process_round1 directly on the lossless source vid.avi (same 30 s, reference struct)
%   Checks the CSV columns and the six .mat variables; prints MAE (dev data, informational only).
%   Outputs: <work>\submission_demo\integration_30s (never inside the repo).
arguments
    opts.Subject (1,1) string = "subject1"
    opts.Seconds (1,1) double = 30
end
assert(ismember(opts.Subject, "subject" + [1 3 5 8:16]), 'dev subjects only');
root = fileparts(fileparts(mfilename('fullpath')));
addpath(root); addpath(fullfile(root, 'config')); addpath(genpath(fullfile(root, 'src')));
cfg = getConfig(); cfg.subject = char(opts.Subject);
base = fullfile(fileparts(cfg.datasetRoot), 'work', 'submission_demo', 'integration_30s');
tic; P = packageRound(opts.Subject, OutRoot=base, MaxSeconds=opts.Seconds, MaxFramesInMat=60); tA = toc;
T = readtable(P.processed_csv, 'TextType', 'string');
assert(isequal(T.Properties.VariableNames, {'time_s', 'hr_bpm', 'confidence', 'status'}));
w = whos('-file', P.mat);
assert(isequal(sort(string({w.name})), sort(["facial_video" "roi_data" "reference_signal" "processed_signal" "metrics" "params"])));
M = load(P.mat);
assert(isa(M.facial_video, 'uint8') && ndims(M.facial_video) == 4);
fprintf('(a) mp4 package: %d windows, MAE %.2f, RMSE %.2f, r %.3f, coverage %.0f %%, status %s, %.0f s\n', ...
    height(T), M.metrics.MAE, M.metrics.RMSE, M.metrics.r, M.metrics.coverage, jsonencode(M.metrics.status_counts), tA);
fprintf('    target_video %dx%d %.2f fps, %.0f kbps; facial_video %s; git %s\n', P.target_video.width, ...
    P.target_video.height, P.target_video.fps, P.target_video.bitrate_kbps, mat2str(size(M.facial_video)), M.params.git_hash);
[vp, gt] = loadUBFC(cfg);
v = VideoReader(vp); n = floor(opts.Seconds * v.FrameRate + 1e-9); clear v
tic; B = process_round1(vp, 'Reference', gt, 'OutDir', fullfile(base, 'avi_direct'), 'MaxFrames', n, 'MaxFramesInMat', 0); tB = toc;
fprintf('(b) lossless avi: MAE %.2f, RMSE %.2f, ungated MAE %.2f, coverage %.0f %%, %.0f s\n', ...
    B.metrics.MAE, B.metrics.RMSE, B.metrics.ungated.MAE, B.metrics.coverage, tB);
S = struct('package', P, 'avi', B.metrics);
end
