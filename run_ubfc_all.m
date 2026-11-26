function out = run_ubfc_all(varargin)
%RUN_UBFC_ALL Run run_pipeline on every UBFC subject found and aggregate.
%   out = run_ubfc_all()            all subjects with a vid.avi under cfg.dataRoot
%   out = run_ubfc_all('maxFrames', 900, ...)   any getConfig override
%   Writes <resultsRoot>/UBFC_summary/: per_subject.csv (MAE per subject,
%   method, ROI) and pooled.csv (metrics over all windows of all subjects).

repoRoot = fileparts(mfilename('fullpath'));
addpath(fullfile(repoRoot, 'config'));
addpath(genpath(fullfile(repoRoot, 'src')));
cfg = getConfig(varargin{:});
subjects = listUBFCSubjects(cfg);
fprintf('UBFC subjects found: %s\n', strjoin(subjects, ', '));

per = table();
hrAll = {}; refAll = {};
for s = 1:numel(subjects)
    o = run_pipeline(varargin{:}, 'subject', subjects{s});
    T = o.table;
    T.Subject = repmat(string(subjects{s}), height(T), 1);
    per = [per; T]; %#ok<AGROW>
    hrAll(:, s)  = {o.R.hr}';     %#ok<AGROW>  % rows = method x ROI
    refAll(:, s) = repmat({o.hrRef}, numel(o.R), 1); %#ok<AGROW>
    labels = [{o.R.method}; {o.R.roi}]';
end

% Pooled over all windows of all subjects
M = [];
for k = 1:size(hrAll, 1)
    M = [M; computeMetrics(vertcat(hrAll{k, :}), vertcat(refAll{k, :}))]; %#ok<AGROW>
end
pooled = table(string(labels(:, 1)), string(labels(:, 2)), [M.MAE]', [M.RMSE]', ...
    [M.MAPE]', [M.r]', [M.bias]', [M.loaLow]', [M.loaHigh]', [M.nWindows]', ...
    'VariableNames', {'Method', 'ROI', 'MAE', 'RMSE', 'MAPE', 'r', 'bias', 'LoA_low', 'LoA_high', 'nWin'});

% MAE per subject, wide: rows method/ROI, one column per subject
wide = pooled(:, {'Method', 'ROI'});
for s = 1:numel(subjects)
    wide.(['MAE_' subjects{s}]) = per.MAE(per.Subject == subjects{s});
end

outDir = fullfile(cfg.resultsRoot, 'UBFC_summary');
if ~isfolder(outDir), mkdir(outDir); end
writetable(per, fullfile(outDir, 'per_subject.csv'));
writetable(pooled, fullfile(outDir, 'pooled.csv'));
writetable(wide, fullfile(outDir, 'mae_by_subject.csv'));

fprintf('\n===== UBFC: MAE (bpm) per subject =====\n'); disp(wide);
fprintf('===== UBFC: pooled over all windows of %d subjects =====\n', numel(subjects)); disp(pooled);
out = struct('subjects', {subjects}, 'perSubject', per, 'pooled', pooled, 'maeWide', wide);
end
