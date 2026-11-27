function [videoPath, gt] = loadUBFC(cfg)
%LOADUBFC Locate a UBFC-rPPG subject (DATASET_1 or DATASET_2) and parse its ground truth.
%   [videoPath, gt] = loadUBFC(cfg)
%   Inputs : cfg.dataRoot, cfg.subject
%   Outputs: videoPath - full path to vid.avi
%            gt.ppg      - contact PPG waveform (column)
%            gt.hrSensor - HR reported by the oximeter, bpm (column)
%            gt.t        - timestamps, seconds (column)
%
%   DATASET_2 ground_truth.txt: line 1 PPG, line 2 sensor HR, line 3 time (s).
%   DATASET_1 gtdump.xmp: CSV rows [time (ms), sensor HR, SpO2, PPG].
%   If cfg.videoPath is non-empty it is used instead of <subject>/vid.avi.

subjDir = findSubjectDir(cfg.dataRoot, cfg.subject);

videoPath = fullfile(subjDir, 'vid.avi');
if isfield(cfg, 'videoPath') && ~isempty(cfg.videoPath)
    videoPath = cfg.videoPath;
end
if ~isfile(videoPath)
    error('loadUBFC:noVideo', 'Video not found: %s', videoPath);
end

xmpPath = fullfile(subjDir, 'gtdump.xmp');
gtPath  = fullfile(subjDir, 'ground_truth.txt');
if ~isfile(gtPath) && isfile(xmpPath)
    gt = readXmp(xmpPath);
    return
end
if ~isfile(gtPath)
    error('loadUBFC:noGT', 'Ground truth not found: %s (or gtdump.xmp)', gtPath);
end

lines = readlines(gtPath, 'EmptyLineRule', 'skip');
if numel(lines) < 3
    error('loadUBFC:gtFormat', ...
        '%s: expected 3 lines (PPG, HR, time), found %d.', gtPath, numel(lines));
end
vals = cell(3, 1);
for k = 1:3
    vals{k} = sscanf(char(lines(k)), '%f');
end
n = cellfun(@numel, vals);
if any(n ~= n(1))
    error('loadUBFC:gtLength', ...
        '%s: line lengths differ (PPG %d, HR %d, time %d).', gtPath, n(1), n(2), n(3));
end

gt.ppg      = vals{1}(:);
gt.hrSensor = vals{2}(:);
gt.t        = vals{3}(:);
gt.path     = gtPath;
end

function gt = readXmp(xmpPath)
% DATASET_1: each row "time_ms, HR, SpO2, PPG".
M = readmatrix(xmpPath, 'FileType', 'text', 'Delimiter', ',');
M = M(all(isfinite(M(:, 1:4)), 2), :);
if size(M, 2) < 4 || isempty(M)
    error('loadUBFC:xmpFormat', '%s: expected 4 comma-separated columns.', xmpPath);
end
gt.ppg      = M(:, 4);
gt.hrSensor = M(:, 2);
gt.t        = M(:, 1) / 1000;
gt.path     = xmpPath;
end

function subjDir = findSubjectDir(root, subject)
% Look for <root>/<subject>, then <root>/<any>/<subject>.
if ~isfolder(root)
    error('loadUBFC:noRoot', 'Data root not found: %s', root);
end
subjDir = fullfile(root, subject);
if isfolder(subjDir)
    return
end
d = dir(root);
d = d([d.isdir] & ~startsWith({d.name}, '.'));
for k = 1:numel(d)
    cand = fullfile(root, d(k).name, subject);
    if isfolder(cand)
        subjDir = cand;
        return
    end
end
error('loadUBFC:noSubject', 'Subject folder "%s" not found under %s', subject, root);
end
