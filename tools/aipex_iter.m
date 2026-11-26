function R = aipex_iter(expId, hypothesis, change, varargin)
%AIPEX_ITER One experiment of the AiPEX iteration loop (README Section 8.5), AiPEX-dev only.
%   R = aipex_iter('E002', 'hypothesis text', 'change text', 'name', value, ...)
%   Name/value pairs override the default configuration below (the "config_diff").
%   Uses traces cached by aipex_cache_dev (dev videos only). Per experiment it writes
%   outputs/aipex_iter/<expId>/per_video.csv and diag.png, and appends a row to
%   outputs/aipex_iter/log.csv with the suggested keep/reject decision.
%   Comparison: paired bootstrap (5,000 resamples over videos) against the current best
%   (outputs/aipex_iter/best_per_video.csv). The first run with 'SetBest', true defines it.
%
%   Configuration (defaults = E000 baseline = integrated pipeline):
%     pixels      'skin' | 'all'          pixels averaged inside each crop
%     fuse        'rgb'  | 'spectral'     'rgb': pixel-weighted mean RGB of the regions -> one trace;
%                                         'spectral': per-region spectra, SNR-weighted top-k fusion
%     regions     ["forehead" "leftcheek" "rightcheek"]
%     topK        3                       (spectral fusion)
%     kalman      true                    kalman_denoise (Thamil), Npix-weighted
%     method      'POS' | 'CHROM'
%     windowSec   10
%     videoEst    'avgspec' | 'median' | 'mean' | 'viterbi' (spectral fusion only; viterbiLambda 0.1)
%     harmonic    0                       alpha of P(f) + alpha*P(2f) before the video-level peak
%     posWindowSec 1.6, detrendLambda 100, filterOrder 4, hrBand [0.7 3.0]   (signal chain, getConfig)
%     cvSeed      2027

arguments
    expId (1,1) string
    hypothesis (1,1) string
    change (1,1) string
end
arguments (Repeating)
    varargin
end
root = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root, 'config')); addpath(genpath(fullfile(root, 'src')));

C = struct('pixels', 'skin', 'fuse', 'rgb', 'regions', ["forehead" "leftcheek" "rightcheek"], ...
    'topK', 3, 'kalman', true, 'method', 'POS', 'windowSec', 10, 'videoEst', 'avgspec', ...
    'harmonic', 0, 'posWindowSec', 1.6, 'detrendLambda', 100, 'filterOrder', 4, 'hrBand', [0.7 3.0], ...
    'viterbiLambda', 0.1, 'motionGamma', 0, 'motionBeta', 0.5, 'cvSeed', 2027, 'SetBest', false, ...
    'TraceRoot', "D:\SPCup Data\work\aipex_iter\traces", 'TraceRoot2', "D:\SPCup Data\work\aipex_iter\traces2", 'TraceRoot3', "D:\SPCup Data\work\aipex_iter\traces3");
diffTxt = strings(0);
for k = 1:2:numel(varargin)
    name = varargin{k};
    if ~isfield(C, name), error('aipex_iter:opt', 'Unknown option %s', name); end
    C.(name) = varargin{k+1};
    if ~ismember(name, ["SetBest" "TraceRoot" "TraceRoot2" "TraceRoot3"]), diffTxt(end+1) = name + "=" + string(join(string(varargin{k+1}), '|')); end %#ok<AGROW>
end
cfg = getConfig('hrWindowSec', C.windowSec, 'posWindowSec', C.posWindowSec, ...
    'detrendLambda', C.detrendLambda, 'filterOrder', C.filterOrder, 'hrBand', C.hrBand);

% ---- dev videos and reference ----
ids = readlines(fullfile(root, 'shared', 'splits', 'aipex_dev.txt'), 'EmptyLineRule', 'skip');
nV = numel(ids);
country = strings(nV, 1); video = strings(nV, 1); ref = NaN(nV, 1);
gtCache = struct();
for i = 1:nV
    p = split(ids(i), '/'); country(i) = p(1); video(i) = p(2);
    key = matlab.lang.makeValidName(country(i));
    if ~isfield(gtCache, key)
        gtCache.(key) = readtable(fullfile(cfg.aipexRoot, 'Ground_Truth', country(i) + "_GT.csv"), ...
            'TextType', 'string', 'VariableNamingRule', 'preserve');
    end
    G = gtCache.(key);
    ref(i) = G{strtrim(G{:, 1}) == video(i), 2};
end

% ---- estimate per video ----
est = NaN(nV, 1);
for i = 1:nV
    S = load(fullfile(C.TraceRoot, country(i) + "_" + video(i) + ".mat"), 'R');
    if C.motionGamma > 0                          % ladder 13: crop motion (aipex_cache_dev3)
        S3 = load(fullfile(C.TraceRoot3, country(i) + "_" + video(i) + ".mat"), 'M');
        for r = fieldnames(S3.M)', S.R.(r{1}).motion = S3.M.(r{1}); end
    end
    if ismember(C.pixels, {'trim', 'tex'})        % ladder 12: extra pixel statistics (aipex_cache_dev2)
        S2 = load(fullfile(C.TraceRoot2, country(i) + "_" + video(i) + ".mat"), 'R2');
        for r = fieldnames(S2.R2)'
            S.R.(r{1}).rgbTrim = S2.R2.(r{1}).rgbTrim;
            S.R.(r{1}).rgbTex = S2.R2.(r{1}).rgbTex;
            S.R.(r{1}).npixTex = S2.R2.(r{1}).npixTex;
        end
    end
    est(i) = videoEstimate(S.R, C, cfg);
end

% ---- metrics, CV folds (stratified by country), bootstrap ----
err = est - ref; ae = abs(err);
rng(C.cvSeed, 'twister');
fold = zeros(nV, 1);
for c = unique(country)'
    idx = find(country == c); idx = idx(randperm(numel(idx)));
    fold(idx) = mod(0:numel(idx)-1, 5)' + 1;
end
foldMae = arrayfun(@(f) mean(ae(fold == f)), 1:5);
cvMae = mean(foldMae);
rng(1, 'twister');
B = 5000; bootMae = zeros(B, 1);
for b = 1:B, j = randi(nV, nV, 1); bootMae(b) = mean(ae(j)); end
ci = prctile(bootMae, [2.5 97.5]);
indiaMae = mean(ae(country == "India")); slMae = mean(ae(country == "Sierra_Leone"));
within5 = 100 * mean(ae < 5); bias = mean(err);

% ---- compare with current best (paired bootstrap) ----
bestFile = fullfile(root, 'outputs', 'aipex_iter', 'best_per_video.csv');
delta = NaN; pval = NaN; dCi = [NaN NaN]; decision = "baseline";
if isfile(bestFile)
    Bt = readtable(bestFile, 'TextType', 'string');
    [~, ia, ib] = intersect(country + "/" + video, Bt.country + "/" + Bt.video, 'stable');
    aeBest = NaN(nV, 1); aeBest(ia) = abs(Bt.est(ib) - Bt.ref(ib));
    g = aeBest - ae;                         % >0: this experiment is better
    delta = mean(ae) - mean(aeBest);
    rng(2, 'twister'); bg = zeros(B, 1);
    for b = 1:B, j = randi(nV, nV, 1); bg(b) = mean(g(j)); end
    dCi = prctile(bg, [2.5 97.5]);
    pval = mean(bg <= 0);
    bestGap = abs(mean(aeBest(country == "India")) - mean(aeBest(country == "Sierra_Leone")));
    gapOk = abs(indiaMae - slMae) <= bestGap + 0.5;
    if delta < 0 && dCi(1) > 0 && gapOk, decision = "keep (pending UBFC check)"; else, decision = "reject"; end
end

% ---- outputs ----
outDir = fullfile(root, 'outputs', 'aipex_iter', expId);
if ~isfolder(outDir), mkdir(outDir); end
P = table(country, video, ref, est, err, fold);
writetable(P, fullfile(outDir, 'per_video.csv'));
diagFigure(P, fullfile(outDir, 'diag.png'), expId, cvMae);
if C.SetBest
    writetable(P, bestFile);
    decision = "baseline (set as best)";
end
[~, h] = system(sprintf('git -C "%s" rev-parse --short HEAD', root));
row = {expId, string(datetime('now', 'Format', 'yyyy-MM-dd')), strtrim(string(h)), hypothesis, change, ...
    strjoin(diffTxt, '; '), round(cvMae, 3), sprintf('[%.2f, %.2f]', ci), round(delta, 3), round(pval, 4), ...
    round(indiaMae, 3), round(slMae, 3), round(within5, 1), round(bias, 3), "", decision, ...
    sprintf('fold MAE %s; gain CI [%.2f, %.2f]', mat2str(round(foldMae, 2)), dCi)};
L = cell2table(row, 'VariableNames', {'exp_id', 'date', 'git_hash', 'hypothesis', 'change', 'config_diff', ...
    'cv_mae', 'cv_mae_ci95', 'delta_vs_best', 'p_paired_bootstrap', 'india_mae', 'sl_mae', ...
    'within5_pct', 'bias', 'ubfc_dev_mae', 'decision', 'notes'});
logFile = fullfile(root, 'outputs', 'aipex_iter', 'log.csv');
writetable(L, logFile, 'WriteMode', 'append');
fprintf('%s: CV MAE %.2f %s | India %.2f SL %.2f | within5 %.0f%% | bias %+.2f | delta %+.2f (gain CI [%.2f %.2f]) -> %s\n', ...
    expId, cvMae, sprintf('[%.2f, %.2f]', ci), indiaMae, slMae, within5, bias, delta, dCi, decision);
R = struct('table', P, 'cvMae', cvMae, 'ci', ci, 'delta', delta, 'decision', decision, 'config', C);
end

% =====================================================================
function hr = videoEstimate(R, C, cfg)
regs = C.regions;
N = min(arrayfun(@(r) numel(R.(r).t), regs));
t = R.(regs(1)).t(1:N);
fs = 1 / median(diff(t));
X = struct(); W = struct(); Mo = struct();
for r = regs
    Q = R.(r);
    switch C.pixels
        case 'skin', x = Q.rgbSkin(1:N, :); w = Q.npix(1:N);
        case 'all',  x = Q.rgbAll(1:N, :);  w = repmat(Q.nAll, N, 1);
        case 'trim', x = Q.rgbTrim(1:N, :); w = Q.npix(1:N);       % skin, no specular, 10/10 trimmed mean
        case 'tex',  x = Q.rgbTex(1:N, :);  w = Q.npixTex(1:N);    % + top-15 % gradient pixels removed
        otherwise, error('aipex_iter:pixels', 'Unknown pixels option "%s"', C.pixels);
    end
    X.(r) = x; W.(r) = w;
    if isfield(Q, 'motion'), Mo.(r) = Q.motion(1:N, :); else, Mo.(r) = []; end
end
switch C.fuse
    case 'rgb'                                   % pixel-weighted mean RGB -> one trace
        num = zeros(N, 3); den = zeros(N, 1);
        for r = regs
            ok = all(isfinite(X.(r)), 2);
            num(ok, :) = num(ok, :) + W.(r)(ok) .* X.(r)(ok, :);
            den(ok) = den(ok) + W.(r)(ok);
        end
        spec = traceSpectrum(num ./ den, den, fs, C, cfg);
        hr = specToHr(spec, C);
    case 'spectral'                              % per-region spectra, SNR-weighted top-k
        if strcmp(C.videoEst, 'viterbi')         % per-window fusion -> Viterbi -> conf-weighted mean
            Pw = {}; Sw = [];
            for r = regs
                spec = traceSpectrum(X.(r), W.(r), fs, C, cfg, Mo.(r));
                if isempty(spec), continue; end
                Pn = spec.P ./ sum(spec.P, 2);
                Pw{end+1} = Pn; %#ok<AGROW>
                Sw(:, end+1) = arrayfun(@(w) snrDb(Pn(w, :), spec.f), (1:size(Pn, 1))'); %#ok<AGROW>
            end
            if isempty(Pw), hr = NaN; return; end
            nWin = size(Pw{1}, 1); Pf = zeros(nWin, size(Pw{1}, 2));
            for w = 1:nWin
                wt = max(10.^(Sw(w, :) / 10) - 1, 0);
                if all(wt == 0), wt = ones(size(wt)); end
                [~, o] = sort(wt, 'descend'); o = o(1:min(C.topK, numel(o)));
                for j = o, Pf(w, :) = Pf(w, :) + wt(j) * Pw{j}(w, :); end
                Pf(w, :) = Pf(w, :) / sum(wt(o));
            end
            [path, conf] = viterbiHR(Pf, spec.f, Lambda=C.viterbiLambda);
            hr = sum(conf .* path) / max(sum(conf), eps);
            return
        end
        Ps = []; snr = [];
        for r = regs
            spec = traceSpectrum(X.(r), W.(r), fs, C, cfg, Mo.(r));
            if isempty(spec), continue; end
            m = mean(spec.P ./ sum(spec.P, 2), 1);
            Ps = [Ps; m]; snr(end+1) = snrDb(m, spec.f); %#ok<AGROW>
        end
        if isempty(Ps), hr = NaN; return; end
        w = max(10.^(snr / 10) - 1, 0);
        if all(w == 0), w = ones(size(w)); end
        [~, o] = sort(w, 'descend'); o = o(1:min(C.topK, numel(o)));
        S.f = spec.f; S.P = sum(w(o)' .* Ps(o, :), 1) / sum(w(o));
        S.hrWin = []; S.avg = true;
        hr = specToHr(S, C);
end
end

function spec = traceSpectrum(x, npix, fs, C, cfg, mo)
if nargin < 6, mo = []; end
ok = all(isfinite(x), 2);
if nnz(ok) < C.windowSec * fs, spec = []; return; end
x = fillmissing(x, 'linear', 'EndValues', 'nearest');
if C.kalman, x = kalman_denoise(x, fs, Npix=npix); end
p  = rgbToPulse(x, fs, C.method, cfg);
pf = preprocessPulse(p, fs, cfg);
[~, hrWin, s] = estimateHR(pf, fs, cfg);
if C.motionGamma > 0 && ~isempty(mo)              % README 4.8.2: attenuate where the crop's own motion has power
    nfft = s.nfft; F = (0:nfft/2)' * fs / nfft; band = F >= cfg.hrBand(1) & F <= cfg.hrBand(2);
    L = s.winLen; hw = hann(L);
    for w = 1:size(s.P, 1)
        seg = mo(s.starts(w):s.starts(w)+L-1, :); seg = seg - mean(seg, 1);
        Pm = sum(abs(fft(seg .* hw, nfft)).^2, 2); Pm = Pm(1:nfft/2+1); Pm = Pm(band);
        Mn = Pm / max(max(Pm), eps);
        att = 1 - C.motionGamma * Mn .* (Mn > C.motionBeta);
        s.P(w, :) = s.P(w, :) .* att';
    end
end
spec = struct('f', s.f, 'P', s.P, 'hrWin', hrWin, 'avg', false);
end

function hr = specToHr(spec, C)
if isempty(spec), hr = NaN; return; end
switch C.videoEst
    case 'median', if ~isempty(spec.hrWin), hr = median(spec.hrWin); return; end
    case 'mean',   if ~isempty(spec.hrWin), hr = mean(spec.hrWin); return; end
end
if spec.avg, m = spec.P; else, m = mean(spec.P ./ sum(spec.P, 2), 1); end
if C.harmonic > 0                                 % H(f) = P(f) + alpha * P(2f)
    m2 = reshape(interp1(spec.f(:), m(:), 2 * spec.f(:), 'linear', 0), size(m));
    m = m + C.harmonic * m2;
end
[~, k] = max(m); delta = 0;
if k > 1 && k < numel(m)
    a = m(k-1); b = m(k); c = m(k+1); den = a - 2*b + c;
    if den ~= 0, delta = 0.5 * (a - c) / den; end
end
hr = 60 * (spec.f(k) + delta * (spec.f(2) - spec.f(1)));
end

function s = snrDb(m, f)
[~, k] = max(m); f0 = f(k); bw = 6 / 60;
sig = abs(f - f0) <= bw | abs(f - 2 * f0) <= bw;
s = 10 * log10(sum(m(sig)) / max(sum(m(~sig)), eps));
end

function diagFigure(P, path, expId, cvMae)
f = figure('Visible', 'off', 'Position', [0 0 900 320]);
if exist('theme', 'file'), theme(f, 'light'); end
subplot(1, 2, 1);
gscatter(P.ref, P.est, P.country); hold on; plot([40 180], [40 180], 'k--');
c = polyfit(P.ref, P.est, 1);
title(sprintf('%s: CV MAE %.2f, fit est = %.2f ref %+.1f', expId, cvMae, c(1), c(2)));
xlabel('reference HR (bpm)'); ylabel('estimate (bpm)'); axis([40 180 40 180]); grid on;
subplot(1, 2, 2);
histogram(P.est ./ P.ref, 0.3:0.05:2.6); xlabel('estimate / reference'); grid on;
xline([0.5 1 1.2 2], ':');
exportgraphics(f, path); close(f);
end
