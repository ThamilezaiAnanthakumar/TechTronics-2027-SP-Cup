function out = run_ubfc1(varargin)
%RUN_UBFC1 Run the HR pipeline on UBFC-rPPG DATASET_1 and summarise.
%   out = run_ubfc1()          all usable DATASET_1 subjects
%   out = run_ubfc1('maxFrames', 900, ...)   any getConfig override
%
%   DATASET_1 layout as downloaded from Drive (under <datasetRoot>/UBFC-rPPG/Dataset1):
%     <name>/vid.avi + <name>/gtdump.xmp   complete subjects (e.g. 8-gt)
%     <name>/gtdump.xmp only               video downloaded separately as a
%                                          loose "vid-001 (n).avi" file
%   Loose videos are paired with GT-only folders by agreement between the
%   POS/combined rPPG HR and each candidate reference (plus duration). The
%   pairing is printed so it can be checked; fix it with cfg.ubfc1VideoMap
%   below if it is wrong.
%
%   Outputs: <resultsRoot>/UBFC1/<subject>/... and UBFC1_summary/*.csv

repoRoot = fileparts(mfilename('fullpath'));
addpath(fullfile(repoRoot, 'config'));
addpath(genpath(fullfile(repoRoot, 'src')));
base = getConfig(varargin{:});
root = fullfile(base.datasetRoot, 'UBFC-rPPG', 'Dataset1');
resRoot = fullfile(base.resultsRoot, 'UBFC1');
common = [varargin, {'dataRoot', root, 'resultsRoot', resRoot}];

% ---- Inventory ----
d = dir(fullfile(root, '*', 'gtdump.xmp'));
gtOnly = {}; complete = {};
for k = 1:numel(d)
    [~, name] = fileparts(d(k).folder);
    v = dir(fullfile(d(k).folder, 'vid.avi'));
    if ~isempty(v) && v.bytes > 0
        complete{end+1} = name; %#ok<AGROW>
    else
        gtOnly{end+1} = name; %#ok<AGROW>
    end
end
loose = dir(fullfile(root, '*.avi'));
loose = fullfile(root, {loose.name});
fprintf('Complete subjects : %s\n', strjoin(complete, ', '));
fprintf('GT-only folders   : %s\n', strjoin(gtOnly, ', '));
fprintf('Loose videos      : %d\n', numel(loose));

% ---- Pair loose videos with GT-only folders ----
pairs = pairLooseVideos(loose, gtOnly, root, resRoot, common);

% ---- Run every subject ----
jobs = [cellfun(@(s) {s, ''}, complete, 'uni', 0), ...
        arrayfun(@(k) {pairs.gt{k}, pairs.video{k}}, 1:numel(pairs.gt), 'uni', 0)];
per = table(); hrAll = {}; refAll = {}; names = {};
for j = 1:numel(jobs)
    subj = jobs{j}{1}; vid = jobs{j}{2};
    try
        o = run_pipeline(common{:}, 'subject', subj, 'videoPath', vid);
    catch ME
        fprintf(2, '%s failed: %s\n', subj, ME.message);
        continue
    end
    T = o.table; T.Subject = repmat(string(subj), height(T), 1);
    per = [per; T]; %#ok<AGROW>
    hrAll(:, end+1)  = {o.R.hr}';                       %#ok<AGROW>
    refAll(:, end+1) = repmat({o.hrRef}, numel(o.R), 1); %#ok<AGROW>
    names{end+1} = subj; %#ok<AGROW>
    labels = [{o.R.method}; {o.R.roi}]';
end

M = [];
for k = 1:size(hrAll, 1)
    M = [M; computeMetrics(vertcat(hrAll{k, :}), vertcat(refAll{k, :}))]; %#ok<AGROW>
end
pooled = table(string(labels(:, 1)), string(labels(:, 2)), [M.MAE]', [M.RMSE]', ...
    [M.MAPE]', [M.r]', [M.bias]', [M.loaLow]', [M.loaHigh]', [M.nWindows]', ...
    'VariableNames', {'Method', 'ROI', 'MAE', 'RMSE', 'MAPE', 'r', 'bias', 'LoA_low', 'LoA_high', 'nWin'});
wide = pooled(:, {'Method', 'ROI'});
for s = 1:numel(names)
    wide.(matlab.lang.makeValidName(['MAE_' names{s}])) = per.MAE(per.Subject == names{s});
end

outDir = fullfile(base.resultsRoot, 'UBFC1_summary');
if ~isfolder(outDir), mkdir(outDir); end
writetable(per, fullfile(outDir, 'per_subject.csv'));
writetable(pooled, fullfile(outDir, 'pooled.csv'));
writetable(wide, fullfile(outDir, 'mae_by_subject.csv'));
writetable(pairs.table, fullfile(outDir, 'loose_video_pairing.csv'));

fprintf('\n===== UBFC DATASET_1: MAE (bpm) per subject =====\n'); disp(wide);
fprintf('===== UBFC DATASET_1: pooled over %d subjects =====\n', numel(names)); disp(pooled);
out = struct('subjects', {names}, 'perSubject', per, 'pooled', pooled, ...
    'maeWide', wide, 'pairing', pairs.table);
end

function pairs = pairLooseVideos(loose, gtOnly, root, resRoot, common)
% Score every (video, GT) combination with POS/combined MAE, then choose
% the one-to-one assignment with the lowest total MAE.
pairs = struct('gt', {{}}, 'video', {{}}, 'table', table());
if isempty(loose) || isempty(gtOnly), return, end
nV = numel(loose); nG = numel(gtOnly);
mae = nan(nV, nG); vidDur = nan(nV, 1); gtDur = nan(1, nG);
for v = 1:nV
    [~, vn] = fileparts(loose{v});
    tag = ['loose_' matlab.lang.makeValidName(vn)];
    cfg = getConfig(common{:}, 'subject', tag);
    traces = extractRGBTraces(loose{v}, cfg);       % cached per video
    vidDur(v) = traces.t(end);
    p  = rgbToPulse(traces.rgb.combined, traces.fps, 'POS', cfg);
    [~, hr] = estimateHR(preprocessPulse(p, traces.fps, cfg), traces.fps, cfg);
    for g = 1:nG
        cg = getConfig(common{:}, 'subject', gtOnly{g}, 'videoPath', loose{v});
        [~, gt] = loadUBFC(cg);
        gtDur(g) = gt.t(end) - gt.t(1);
        [~, hrRef] = referenceHR(gt, traces, cg);
        mae(v, g) = mean(abs(hr - hrRef));
    end
end
% Brute-force assignment (few files)
best = Inf; bestP = [];
P = perms(1:nG); P = unique(P(:, 1:min(nV, nG)), 'rows');
for i = 1:size(P, 1)
    c = sum(mae(sub2ind(size(mae), 1:size(P, 2), P(i, :))));
    if c < best, best = c; bestP = P(i, :); end
end
rows = {};
for v = 1:numel(bestP)
    g = bestP(v);
    others = mae(v, setdiff(1:nG, g));
    rows(end+1, :) = {string(loose{v}), string(gtOnly{g}), vidDur(v), gtDur(g), ...
        mae(v, g), min(others)}; %#ok<AGROW>
    pairs.gt{end+1} = gtOnly{g};
    pairs.video{end+1} = loose{v};
end
pairs.table = cell2table(rows, 'VariableNames', ...
    {'Video', 'GT', 'VideoDur_s', 'GTDur_s', 'MAE_matched', 'MAE_nextBest'});
fprintf('\nLoose video pairing (POS/combined MAE vs each GT):\n');
[~, vn] = cellfun(@fileparts, loose, 'uni', 0);
disp(array2table(mae, 'RowNames', vn, 'VariableNames', matlab.lang.makeValidName(gtOnly)));
disp(pairs.table);
end
