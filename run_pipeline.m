function out = run_pipeline(varargin)
%RUN_PIPELINE Round 1 baseline: rPPG heart rate on one UBFC DATASET_2 subject.
%   run_pipeline                        full video, defaults from getConfig
%   run_pipeline('maxFrames', 900)      quick 30 s check
%   run_pipeline('subject', 'subject3') another subject
%   Any getConfig field can be overridden as a name/value pair.
%
%   Outputs go to <resultsRoot>/<subject>/ (default outputs/, gitignored):
%     traces.mat, hr_results.mat, metrics_table.csv, processed_output.csv,
%     figures/*.png

repoRoot = fileparts(mfilename('fullpath'));
addpath(fullfile(repoRoot, 'config'));
addpath(genpath(fullfile(repoRoot, 'src')));

cfg = getConfig(varargin{:});
outDir = fullfile(cfg.resultsRoot, cfg.subject);
if ~isfolder(outDir), mkdir(outDir); end

% ---- Data ----
[videoPath, gt] = loadUBFC(cfg);
fprintf('\n== %s ==\nVideo: %s\n', cfg.subject, videoPath);
fprintf('Ground truth: %d samples, %.2f-%.2f s\n', numel(gt.t), gt.t(1), gt.t(end));

traces = extractRGBTraces(videoPath, cfg);
traces.info.videoPath = videoPath;   % cache may hold an old location
info = traces.info;
fprintf('Video: %dx%d, %.3f fps, %.2f s (file), %d frames used (%.2f s)\n', ...
    info.width, info.height, traces.fps, info.videoDuration, info.numFrames, info.duration);
fprintf('Face OK: %d / %d frames (%.1f%%)\n', nnz(traces.faceOK), ...
    numel(traces.faceOK), 100 * mean(traces.faceOK));

% ---- Reference ----
[tRef, hrRef, ref] = referenceHR(gt, traces, cfg);
dS = abs(hrRef - ref.hrSensor);
fprintf('Reference HR: %.1f-%.1f bpm (%d windows); |ref - oximeter| < 5 bpm in %.0f%% of windows (median %.1f)\n', ...
    min(hrRef), max(hrRef), numel(hrRef), 100 * mean(dS < 5), median(dS));
if any(hrRef < 40 | hrRef > 180)
    warning('run_pipeline:refRange', 'Reference HR outside 40-180 bpm in some windows.');
end

% ---- Methods x ROIs ----
R = struct('method', {}, 'roi', {}, 'pf', {}, 'tHR', {}, 'hr', {}, 'spec', {}, 'm', {});
for i = 1:numel(cfg.methods)
    for j = 1:numel(cfg.roiSet)
        method = cfg.methods{i}; roi = cfg.roiSet{j};
        p  = rgbToPulse(traces.rgb.(roi), traces.fps, method, cfg);
        pf = preprocessPulse(p, traces.fps, cfg);
        [tHR, hr, spec] = estimateHR(pf, traces.fps, cfg);
        m  = computeMetrics(hr, hrRef);
        R(end+1) = struct('method', method, 'roi', roi, 'pf', pf, 'tHR', tHR, ...
                          'hr', hr, 'spec', spec, 'm', m); %#ok<AGROW>
    end
end

M = [R.m];
T = table(string({R.method})', string({R.roi})', [M.MAE]', [M.RMSE]', [M.MAPE]', ...
    [M.r]', [M.bias]', [M.loaLow]', [M.loaHigh]', [M.nWindows]', ...
    'VariableNames', {'Method', 'ROI', 'MAE', 'RMSE', 'MAPE', 'r', 'bias', 'LoA_low', 'LoA_high', 'nWin'});
fprintf('\nResults (bpm; MAPE in %%):\n');
disp(T);

[~, best] = min([M.MAE]);
fprintf('Best: %s / %s, MAE %.2f bpm, RMSE %.2f, r %.3f\n', ...
    R(best).method, R(best).roi, R(best).m.MAE, R(best).m.RMSE, R(best).m.r);

% ---- Save ----
writetable(T, fullfile(outDir, 'metrics_table.csv'));
bestRoi = R(best).roi;
csvT = table(tRef, hrRef, ref.hrSensor, 'VariableNames', {'time_s', 'HR_ref', 'HR_sensor'});
for k = find(strcmp({R.roi}, bestRoi))
    csvT.(['HR_' R(k).method]) = R(k).hr;
end
writetable(csvT, fullfile(outDir, 'processed_output.csv'));

params = cfg; %#ok<NASGU>  name kept close to the submission format
metrics = T;  %#ok<NASGU>
save(fullfile(outDir, 'hr_results.mat'), 'R', 'T', 'tRef', 'hrRef', 'ref', ...
    'best', 'params', 'metrics', 'videoPath');

figFiles = makeFigures(traces, R, ref, tRef, hrRef, best, cfg);
fprintf('\nSaved to %s\n', outDir);
fprintf('  %s\n', figFiles{:});

out = struct('cfg', cfg, 'table', T, 'R', R, 'best', best, ...
    'tRef', tRef, 'hrRef', hrRef, 'figures', {figFiles});
end
