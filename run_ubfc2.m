function out = run_ubfc2(varargin)
%RUN_UBFC2 Integrated pipeline on the full UBFC-rPPG DATASET_2.
%   out = run_ubfc2()                          every subject under cfg.dataRoot
%   out = run_ubfc2('subjects', {'subject1'})  a subset
%   out = run_ubfc2('maxFrames', 900)          quick check (first 30 s)
%   Any getConfig field can be overridden as a name/value pair.
%
%   video --[Sulojan: process_video, RetinaFace + tracking + skin/occlusion]-->
%   per-frame ROI RGB --[Thamil: kalman_denoise, optional]--> resample 30 Hz
%   --[Muftee: rgbToPulse -> preprocessPulse -> estimateHR]--> HR per 10 s window
%   vs reference HR from the contact PPG (same windows, referenceHR).
%
%   Needs MATLAB R2025a+ with the RetinaFace add-on for the face module
%   (only on a cache miss; cached traces in cfg.traceDir are reused).
%
%   Writes <resultsRoot>/UBFC2_integrated/:
%     per_subject.csv   metrics per subject x denoise x method x ROI
%     summary.csv       per denoise x method x ROI: mean-over-subjects and
%                       pooled-over-windows metrics, sorted by mean MAE
%     mae_by_subject.csv, face_stats.csv, results.mat, run_log.txt

repoRoot = fileparts(mfilename('fullpath'));
addpath(fullfile(repoRoot, 'config'));
addpath(genpath(fullfile(repoRoot, 'src')));

cfg = getConfig(varargin{:});
outDir = fullfile(cfg.resultsRoot, 'UBFC2_integrated');
if ~isfolder(outDir), mkdir(outDir); end
diary(fullfile(outDir, 'run_log.txt'));
cleanup = onCleanup(@() diary('off'));

subjects = cfg.subjects;
if isempty(subjects), subjects = listUBFCSubjects(cfg); end
subjects = cellstr(subjects);
p = face_params(cfg.faceConfig);
fprintf('=== UBFC2 integrated pipeline: %d subjects, %s ===\n', numel(subjects), ...
    string(datetime('now')));
fprintf('Face config %s | denoise {%s} | methods {%s} | ROIs {%s}\n', cfg.faceConfig, ...
    strjoin(cfg.denoiseSet, ','), strjoin(cfg.methods, ','), strjoin(cfg.intRois, ','));

fd = [];
per = table();
face = table();
store = struct('subject', {}, 'R', {}, 'ref', {});
tAll = tic;
for s = 1:numel(subjects)
    subj = subjects{s};
    fprintf('\n[%d/%d] %s\n', s, numel(subjects), subj);
    try
        c = cfg; c.subject = subj;
        [videoPath, gt] = loadUBFC(c);
        [fv, fd] = faceTraces(videoPath, subj, cfg, p, fd);
        [T, R, ref] = subjectHR(fv, gt, cfg);
    catch err
        if strcmp(err.identifier, 'faceTraces:noDetector'), rethrow(err); end
        fprintf(2, '  FAILED: %s\n', err.message);
        continue
    end
    T.Subject = repmat(string(subj), height(T), 1);
    per = [per; T]; %#ok<AGROW>
    store(end+1) = struct('subject', subj, 'R', R, 'ref', ref); %#ok<AGROW>

    ms = NaN;
    if isfield(fv, 'meta') && isfield(fv.meta, 'timing'), ms = fv.meta.timing.ms_per_frame; end
    face = [face; table(string(subj), numel(fv.t), 100 * mean(fv.quality > 0), ...
        100 * mean(any(fv.occluded, 2)), 100 * ref.validFrac.face, ms, ...
        'VariableNames', {'Subject', 'nFrames', 'targetFoundPct', 'anyOccludedPct', ...
        'faceValidPct', 'msPerFrame'})]; %#ok<AGROW>

    b = T(T.Denoise == cfg.denoiseSet{end} & T.Method == "POS" & T.ROI == "face", :);
    if ~isempty(b)
        fprintf('  %s/POS/face: MAE %.2f bpm, RMSE %.2f, SUC@5 %.0f%% (%d windows)\n', ...
            cfg.denoiseSet{end}, b.MAE, b.RMSE, b.SUC5, b.nWin);
    end
end
if isempty(store)
    error('run_ubfc2:noResults', 'No subject was processed.');
end

% ---- Aggregate over subjects ----
keys = unique(per(:, {'Denoise', 'Method', 'ROI'}), 'stable');
S = keys;
for k = 1:height(keys)
    sel = per.Denoise == keys.Denoise(k) & per.Method == keys.Method(k) & per.ROI == keys.ROI(k);
    S.meanMAE(k)  = mean(per.MAE(sel), 'omitnan');
    S.meanRMSE(k) = mean(per.RMSE(sel), 'omitnan');
    S.medianMAE(k) = median(per.MAE(sel), 'omitnan');
    % pooled over all windows of all subjects
    hr = []; hrRef = [];
    for s = 1:numel(store)
        Rs = store(s).R;
        i = find(strcmp({Rs.denoise}, keys.Denoise(k)) & strcmp({Rs.method}, keys.Method(k)) ...
                 & strcmp({Rs.roi}, keys.ROI(k)), 1);
        hr = [hr; Rs(i).hr(:)];                 %#ok<AGROW>
        hrRef = [hrRef; store(s).ref.hrRef(:)]; %#ok<AGROW>
    end
    m = computeMetrics(hr, hrRef);
    S.pooledMAE(k) = m.MAE;   S.pooledRMSE(k) = m.RMSE;  S.r(k) = m.r;
    S.SUC3(k) = m.SUC3;       S.SUC5(k) = m.SUC5;
    S.bias(k) = m.bias;       S.LoA_low(k) = m.loaLow;   S.LoA_high(k) = m.loaHigh;
    S.nWin(k) = m.nWindows;   S.coverage(k) = m.coverage;
end
S = sortrows(S, 'meanMAE');

wide = keys;
subjDone = {store.subject};
for s = 1:numel(subjDone)
    T = per(per.Subject == subjDone{s}, :);
    [~, ia, ib] = intersect(keys, T(:, {'Denoise', 'Method', 'ROI'}), 'stable');
    col = NaN(height(keys), 1);  col(ia) = T.MAE(ib);
    wide.(subjDone{s}) = col;
end

writetable(per,  fullfile(outDir, 'per_subject.csv'));
writetable(S,    fullfile(outDir, 'summary.csv'));
writetable(wide, fullfile(outDir, 'mae_by_subject.csv'));
writetable(face, fullfile(outDir, 'face_stats.csv'));
params = cfg;
save(fullfile(outDir, 'results.mat'), 'per', 'S', 'wide', 'face', 'store', 'params', 'subjects');

fprintf('\n===== UBFC2: %d / %d subjects, %.1f min =====\n', numel(store), numel(subjects), toc(tAll) / 60);
fprintf('Face module: target found %.1f%% of frames (min %.1f%%), %.1f ms/frame\n', ...
    mean(face.targetFoundPct), min(face.targetFoundPct), mean(face.msPerFrame, 'omitnan'));
disp(S);
fprintf('Saved to %s\n', outDir);
out = struct('summary', S, 'perSubject', per, 'maeWide', wide, 'face', face, ...
             'store', store, 'cfg', cfg);
end
