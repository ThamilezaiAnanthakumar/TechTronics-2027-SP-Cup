function out = run_aipex(varargin)
%RUN_AIPEX Evaluate the pipeline on the AiPEX rppg_biases dataset.
%   out = run_aipex()                 all videos, defaults from getConfig
%   out = run_aipex('useParallel', false, ...)
%   AiPEX gives pre-cropped ROI videos (forehead / left / right cheek) and a
%   single mean HR per video (Masimo oximeter). Per video and per
%   method x ROI, one HR is estimated with videoHR (peak of the averaged
%   window spectra; median of window HRs kept as a secondary estimate) and
%   compared with the GT across videos, overall and per country.
%   Also recomputes the paper's own CHROM/POS/... errors from
%   Results_Final_Main.csv for comparison.
%   Outputs in <resultsRoot>/AiPEX/.

repoRoot = fileparts(mfilename('fullpath'));
addpath(fullfile(repoRoot, 'config'));
addpath(genpath(fullfile(repoRoot, 'src')));
cfg = getConfig(varargin{:});

V = loadAiPEX(cfg);
nV = height(V);
rois = fieldnames(cfg.aipexRoiFiles);
outDir = fullfile(cfg.resultsRoot, 'AiPEX');
if ~isfolder(outDir), mkdir(outDir); end
fprintf('AiPEX: %d videos (%s)\n', nV, strjoin(cfg.aipexCountries, ', '));

nM = numel(cfg.methods); nR = numel(cfg.roiSet);
resSpec = cell(nV, 1); resMed = cell(nV, 1); dur = zeros(nV, 1);
pathRows = cell(nV, 1);
for i = 1:nV
    for r = 1:numel(rois), ps.(rois{r}) = V.(rois{r}){i}; end
    pathRows{i} = ps;
end
cacheFiles = fullfile(outDir, 'traces', V.Country, V.Video + "_traces.mat");

tic;
if cfg.useParallel && isempty(gcp('nocreate')), parpool('Processes'); end
nW = 0; if cfg.useParallel, nW = Inf; end
parfor (i = 1:nV, nW)
    tr = extractCropTraces(pathRows{i}, cfg, char(cacheFiles(i)));
    dur(i) = tr.info.duration;
    eS = nan(nM, nR); eM = nan(nM, nR);
    for a = 1:nM
        for b = 1:nR
            try
                p  = rgbToPulse(tr.rgb.(cfg.roiSet{b}), tr.fps, cfg.methods{a}, cfg);
                pf = preprocessPulse(p, tr.fps, cfg);
                [~, hr, spec] = estimateHR(pf, tr.fps, cfg);
                [eS(a, b), eM(a, b)] = videoHR(spec, hr);
            catch err
                fprintf('  %s/%s %s-%s failed: %s\n', V.Country(i), V.Video(i), ...
                    cfg.methods{a}, cfg.roiSet{b}, err.message);
            end
        end
    end
    resSpec{i} = eS; resMed{i} = eM;
    fprintf('  [%3d/%d] %s %s (%.0f s) done\n', i, nV, V.Country(i), V.Video(i), dur(i));
end
fprintf('Processed %d videos in %.1f min\n', nV, toc / 60);

estSpec = cat(3, resSpec{:});   % nM x nR x nV
estMed  = cat(3, resMed{:});

% ---- Metrics: method x ROI x subset ----
subsets = [{'All'}, cfg.aipexCountries];
T  = metricsTable(estSpec, V, cfg, subsets);
Tm = metricsTable(estMed,  V, cfg, subsets);

% ---- Per-video CSV ----
P = V(:, {'Country', 'Video', 'gtHR'});
P.duration_s = dur;
for a = 1:nM
    for b = 1:nR
        P.(sprintf('HR_%s_%s', cfg.methods{a}, cfg.roiSet{b})) = squeeze(estSpec(a, b, :));
    end
end

% ---- Paper baseline from Results_Final_Main.csv ----
Tp = paperBaseline(cfg);

writetable(T,  fullfile(outDir, 'metrics_avgspec.csv'));
writetable(Tm, fullfile(outDir, 'metrics_medianwin.csv'));
writetable(P,  fullfile(outDir, 'per_video_hr.csv'));
writetable(Tp, fullfile(outDir, 'paper_baseline.csv'));

fprintf('\n===== AiPEX: video-level HR, averaged-spectrum estimator =====\n');
disp(T);
fprintf('===== AiPEX: video-level HR, median-of-windows estimator (All only) =====\n');
disp(Tm(Tm.Subset == "All", :));
fprintf('===== Published methods (Results_Final_Main.csv) vs Masimo =====\n');
disp(Tp);

% ---- Figure: best (All) combination ----
TA = T(T.Subset == "All", :);
[~, ib] = min(TA.MAE);
a = find(strcmp(cfg.methods, TA.Method(ib)));
b = find(strcmp(cfg.roiSet, TA.ROI(ib)));
figFile = aipexFigure(squeeze(estSpec(a, b, :)), V, TA(ib, :), cfg, outDir);
fprintf('Best: %s / %s, MAE %.2f bpm\nFigure: %s\nSaved to %s\n', ...
    TA.Method(ib), TA.ROI(ib), TA.MAE(ib), figFile, outDir);

save(fullfile(outDir, 'aipex_results.mat'), 'T', 'Tm', 'Tp', 'P', 'estSpec', 'estMed', 'V', 'cfg');
out = struct('metrics', T, 'metricsMedian', Tm, 'paper', Tp, 'perVideo', P);
end

% ------------------------------------------------------------------------
function T = metricsTable(est, V, cfg, subsets)
rows = {};
for a = 1:numel(cfg.methods)
    for b = 1:numel(cfg.roiSet)
        for s = 1:numel(subsets)
            sel = true(height(V), 1);
            if ~strcmp(subsets{s}, 'All'), sel = V.Country == subsets{s}; end
            e = squeeze(est(a, b, :));
            ok = sel & ~isnan(e);
            m = computeMetrics(e(ok), V.gtHR(ok));
            rows(end+1, :) = {string(cfg.methods{a}), string(cfg.roiSet{b}), ...
                string(subsets{s}), m.MAE, m.RMSE, m.MAPE, m.r, m.bias, ...
                m.loaLow, m.loaHigh, m.nWindows, mean(abs(e(ok) - V.gtHR(ok)) <= 5) * 100}; %#ok<AGROW>
        end
    end
end
T = cell2table(rows, 'VariableNames', {'Method', 'ROI', 'Subset', 'MAE', 'RMSE', ...
    'MAPE', 'r', 'bias', 'LoA_low', 'LoA_high', 'nVideos', 'pctWithin5'});
end

function Tp = paperBaseline(cfg)
% Columns like Chrom_India_M paired with Masimo_India_M (same row = same subject).
f = fullfile(cfg.aipexRoot, 'Results_Final_Main.csv');
Tp = table();
if ~isfile(f), return; end
R = readtable(f, 'VariableNamingRule', 'preserve');
names = R.Properties.VariableNames;
names = regexprep(names, '^\W+', '');      % strip BOM
R.Properties.VariableNames = names;
methods = {'Chrom', 'POS', 'BKF', 'Sph', 'DP'};
groups = {'India', 'SL'};
rows = {};
for k = 1:numel(methods)
    eAll = []; gAll = [];
    for g = 1:numel(groups)
        eG = []; gG = [];
        for sex = {'M', 'F'}
            ce = sprintf('%s_%s_%s', methods{k}, groups{g}, sex{1});
            cg = sprintf('Masimo_%s_%s', groups{g}, sex{1});
            e = R.(ce); gt = R.(cg);
            ok = ~isnan(e) & ~isnan(gt);
            eG = [eG; e(ok)]; gG = [gG; gt(ok)]; %#ok<AGROW>
        end
        m = computeMetrics(eG, gG);
        rows(end+1, :) = {string(methods{k}), string(groups{g}), m.MAE, m.RMSE, m.r, m.bias, m.nWindows}; %#ok<AGROW>
        eAll = [eAll; eG]; gAll = [gAll; gG]; %#ok<AGROW>
    end
    m = computeMetrics(eAll, gAll);
    rows(end+1, :) = {string(methods{k}), "All", m.MAE, m.RMSE, m.r, m.bias, m.nWindows}; %#ok<AGROW>
end
Tp = cell2table(rows, 'VariableNames', {'Method', 'Subset', 'MAE', 'RMSE', 'r', 'bias', 'n'});
end

function f = aipexFigure(est, V, row, cfg, outDir)
figDir = fullfile(outDir, 'figures');
if ~isfolder(figDir), mkdir(figDir); end
fig = figure('Visible', 'off', 'Position', [100 100 1100 450]);
cols = lines(numel(cfg.aipexCountries));
subplot(1, 2, 1); hold on; grid on;
for c = 1:numel(cfg.aipexCountries)
    s = V.Country == cfg.aipexCountries{c};
    plot(V.gtHR(s), est(s), 'o', 'Color', cols(c, :));
end
lim = [min([V.gtHR; est]) max([V.gtHR; est])] + [-5 5];
plot(lim, lim, 'k--'); xlim(lim); ylim(lim); axis square;
xlabel('GT mean HR (bpm)'); ylabel('Estimated HR (bpm)');
legend(strrep(cfg.aipexCountries, '_', ' '), 'Location', 'best');
title(sprintf('%s / %s: MAE %.2f, r %.2f', row.Method, row.ROI, row.MAE, row.r));
subplot(1, 2, 2); hold on; grid on;
d = est - V.gtHR; avg = (est + V.gtHR) / 2;
for c = 1:numel(cfg.aipexCountries)
    s = V.Country == cfg.aipexCountries{c};
    plot(avg(s), d(s), 'o', 'Color', cols(c, :));
end
yline(row.bias, 'k-', sprintf('bias %.2f', row.bias));
yline(row.LoA_low, 'r--', sprintf('%.1f', row.LoA_low));
yline(row.LoA_high, 'r--', sprintf('%.1f', row.LoA_high));
xlabel('Mean of est. and GT (bpm)'); ylabel('Est. - GT (bpm)');
title('Bland-Altman (one point per video)');
f = fullfile(figDir, 'aipex_best_scatter_ba.png');
exportgraphics(fig, f, 'Resolution', 150);
close(fig);
end
