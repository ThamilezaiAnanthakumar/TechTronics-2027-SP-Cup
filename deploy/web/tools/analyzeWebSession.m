function out = analyzeWebSession(tracesCsv, refsCsv, fs)
%ANALYZEWEBSESSION Run the MATLAB pipeline on a session recorded in the browser.
%   out = analyzeWebSession('facevitals_..._traces.csv')
%   out = analyzeWebSession(tracesCsv, refsCsv)      also score logged references
%   out = analyzeWebSession(tracesCsv, refsCsv, fs)  resampling rate (default 30)
%   The browser records spatial-mean RGB per ROI at irregular camera times.
%   They are resampled to a uniform grid, then rgbToPulse -> preprocessPulse
%   -> estimateHR exactly as run_pipeline does. If refsCsv is given, each
%   logged reference is compared with the window ending at that time.

if nargin < 3, fs = 30; end
webDir   = fileparts(fileparts(mfilename('fullpath')));
repoRoot = fileparts(fileparts(webDir));
addpath(fullfile(repoRoot, 'config'));
addpath(genpath(fullfile(repoRoot, 'src')));
cfg = getConfig();

T = readtable(tracesCsv);
base = {'forehead', 'leftCheek', 'rightCheek'};
t = T.t_s;
tq = (t(1):1/fs:t(end))';
fprintf('%s: %.1f s, %d frames (%.1f fps), face in %.0f%% of frames\n', tracesCsv, ...
    t(end) - t(1), height(T), (height(T) - 1) / (t(end) - t(1)), 100 * mean(T.faceOK));

rgb = struct(); totS = zeros(numel(tq), 3); totN = zeros(numel(tq), 1);
for r = 1:numel(base)
    M = [T.([base{r} '_R']) T.([base{r} '_G']) T.([base{r} '_B'])];
    n = T.([base{r} '_n']);
    ok = n > 0 & all(isfinite(M), 2);
    rgb.(base{r}) = interp1(t(ok), M(ok, :), tq, 'linear', 'extrap');
    nq = interp1(t, n, tq, 'previous', 'extrap');
    totS = totS + rgb.(base{r}) .* nq; totN = totN + nq;
end
rgb.combined = totS ./ max(totN, 1);

R = struct('method', {}, 'roi', {}, 'tHR', {}, 'hr', {});
for i = 1:numel(cfg.methods)
    for j = 1:numel(cfg.roiSet)
        p  = rgbToPulse(rgb.(cfg.roiSet{j}), fs, cfg.methods{i}, cfg);
        pf = preprocessPulse(p, fs, cfg);
        [tHR, hr] = estimateHR(pf, fs, cfg);
        R(end+1) = struct('method', cfg.methods{i}, 'roi', cfg.roiSet{j}, 'tHR', tHR + tq(1), 'hr', hr); %#ok<AGROW>
    end
end

tbl = table(string({R.method})', string({R.roi})', arrayfun(@(r) median(r.hr), R)', ...
    'VariableNames', {'Method', 'ROI', 'medianHR'});
if nargin >= 2 && ~isempty(refsCsv)
    F = readtable(refsCsv);
    halfWin = (round(cfg.hrWindowSec * fs) - 1) / 2 / fs;
    mae = zeros(numel(R), 1);
    for k = 1:numel(R)
        e = arrayfun(@(tr) interp1(R(k).tHR + halfWin, R(k).hr, tr, 'nearest', 'extrap'), F.t_s);
        mae(k) = mean(abs(e - F.hr_ref));
    end
    tbl.MAE_vs_refs = mae;
    fprintf('%d logged references; browser-side MAE (median estimate): %.2f bpm\n', ...
        height(F), mean(abs(F.hr_est - F.hr_ref), 'omitnan'));
end
disp(tbl);
out = struct('table', tbl, 'R', R, 'rgb', rgb, 'tq', tq);
end
