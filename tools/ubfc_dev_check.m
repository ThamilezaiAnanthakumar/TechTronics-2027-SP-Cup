function m = ubfc_dev_check(varargin)
%UBFC_DEV_CHECK Regression check on the UBFC2 dev split (README 8.2, 8.5 step 3), cached traces.
%   m = ubfc_dev_check()                         baseline: POS on the combined-ROI RGB trace
%   m = ubfc_dev_check('fuse','spectral','topK',3)
%   Options mirror tools/aipex_iter.m where they apply: fuse ('rgb'|'spectral'), topK, kalman,
%   method, harmonic. Dev subjects: 1, 3, 5, 8-16. Traces: seeded run_ubfc_all outputs.
%   Reference = PPG windows saved by run_pipeline (plain chain, no fusion/tracking).
%   Returns pooled MAE over all dev windows (m.MAE) and the per-subject MAEs.
C = struct('fuse', 'rgb', 'topK', 3, 'kalman', false, 'method', 'POS', 'harmonic', 0, 'filterOrder', 4, ...
    'ResultsRoot', "D:\SPCup Data\work\ubfc2_seeded");
for k = 1:2:numel(varargin), C.(varargin{k}) = varargin{k+1}; end
root = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root, 'config')); addpath(genpath(fullfile(root, 'src')));
cfg = getConfig('filterOrder', C.filterOrder);
subs = [1 3 5 8 9 10 11 12 13 14 15 16];
estAll = []; refAll = []; perSub = NaN(size(subs));
for s = 1:numel(subs)
    subj = "subject" + subs(s);
    L = load(fullfile(C.ResultsRoot, subj, 'traces.mat'), 'traces'); tr = L.traces;
    H = load(fullfile(C.ResultsRoot, subj, 'hr_results.mat'), 'hrRef');
    fs = tr.fps;
    switch C.fuse
        case 'rgb'
            sp = winSpectra(tr.rgb.combined, fs, C, cfg);
            P = sp.P;
        case 'spectral'
            rois = {'forehead', 'leftCheek', 'rightCheek'};
            Ps = cell(1, 3); snr = [];
            for r = 1:3
                sp = winSpectra(tr.rgb.(rois{r}), fs, C, cfg);
                Ps{r} = sp.P ./ sum(sp.P, 2);
                snr(:, r) = arrayfun(@(w) snrDb(Ps{r}(w, :), sp.f), (1:size(sp.P, 1))'); %#ok<AGROW>
            end
            P = zeros(size(Ps{1}));
            for w = 1:size(P, 1)
                wt = max(10.^(snr(w, :) / 10) - 1, 0);
                if all(wt == 0), wt = ones(1, 3); end
                [~, o] = sort(wt, 'descend'); o = o(1:min(C.topK, 3));
                for r = o, P(w, :) = P(w, :) + wt(r) * Ps{r}(w, :); end
                P(w, :) = P(w, :) / sum(wt(o));
            end
    end
    hr = arrayfun(@(w) peakHr(P(w, :), sp.f, C.harmonic), (1:size(P, 1))');
    n = min(numel(hr), numel(H.hrRef));
    estAll = [estAll; hr(1:n)]; refAll = [refAll; H.hrRef(1:n)]; %#ok<AGROW>
    perSub(s) = mean(abs(hr(1:n) - H.hrRef(1:n)));
end
mm = computeMetrics(estAll, refAll);
m = struct('MAE', mm.MAE, 'RMSE', mm.RMSE, 'nWin', mm.nWindows, 'perSubject', perSub, 'subjects', subs);
fprintf('UBFC dev (%s): pooled MAE %.3f, RMSE %.2f over %d windows; per subject %s\n', ...
    strjoin(string(struct2cell(rmfield(C, 'ResultsRoot')))', ','), m.MAE, m.RMSE, m.nWin, mat2str(round(perSub, 2)));
end

function sp = winSpectra(x, fs, C, cfg)
x = fillmissing(x, 'linear', 'EndValues', 'nearest');
if C.kalman, x = kalman_denoise(x, fs); end
pf = preprocessPulse(rgbToPulse(x, fs, C.method, cfg), fs, cfg);
[~, ~, s] = estimateHR(pf, fs, cfg);
sp = struct('f', s.f, 'P', s.P);
end

function hr = peakHr(m, f, alpha)
if alpha > 0, m = m + alpha * interp1(f, m, 2 * f, 'linear', 0); end
[~, k] = max(m); d = 0;
if k > 1 && k < numel(m)
    a = m(k-1); b = m(k); c = m(k+1); den = a - 2*b + c;
    if den ~= 0, d = 0.5 * (a - c) / den; end
end
hr = 60 * (f(k) + d * (f(2) - f(1)));
end

function s = snrDb(m, f)
[~, k] = max(m); f0 = f(k); bw = 6 / 60;
sig = abs(f - f0) <= bw | abs(f - 2 * f0) <= bw;
s = 10 * log10(sum(m(sig)) / max(sum(m(~sig)), eps));
end
