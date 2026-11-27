function T = diagnose_aipex_timing()
%DIAGNOSE_AIPEX_TIMING README 2.2 item 1 on AiPEX-dev only: header fps vs frame count vs timestamps.
%   Reads shared/splits/aipex_dev.txt; never opens AiPEX-test videos.
%   Writes outputs/aipex_iter/timing_dev.csv.
root = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root, 'config'));
cfg = getConfig();
ids = readlines(fullfile(root, 'shared', 'splits', 'aipex_dev.txt'), 'EmptyLineRule', 'skip');
rows = cell(numel(ids), 9);
for i = 1:numel(ids)
    parts = split(ids(i), '/'); c = parts(1); v = parts(2);
    f = fullfile(cfg.aipexRoot, 'Data', c, v + "_forehead.avi");
    r = VideoReader(f);
    t = zeros(r.NumFrames, 1); k = 0;
    while hasFrame(r), k = k + 1; t(k) = r.CurrentTime; readFrame(r); end
    t = t(1:k); dt = diff(t);
    rows(i, :) = {c, v, r.FrameRate, r.NumFrames, r.Duration, r.NumFrames / r.Duration, ...
                  1 / median(dt), max(dt) / min(dt), k};
end
T = cell2table(rows, 'VariableNames', {'Country', 'Video', 'HeaderFps', 'NumFrames', 'Duration_s', ...
    'FramesPerDuration', 'TimestampFps', 'MaxMinDtRatio', 'FramesRead'});
writetable(T, fullfile(root, 'outputs', 'aipex_iter', 'timing_dev.csv'));
fprintf('header fps: %s\n', mat2str(unique(round(T.HeaderFps, 3))'));
fprintf('timestamp fps: min %.3f median %.3f max %.3f; max/min dt ratio max %.3f\n', ...
    min(T.TimestampFps), median(T.TimestampFps), max(T.TimestampFps), max(T.MaxMinDtRatio));
fprintf('duration (s): min %.1f median %.1f max %.1f; frames: min %d median %d max %d\n', ...
    min(T.Duration_s), median(T.Duration_s), max(T.Duration_s), min(T.NumFrames), median(T.NumFrames), max(T.NumFrames));
end
