function [tRef, hrRef, ref] = referenceHR(gt, traces, cfg)
%REFERENCEHR Reference HR from the contact PPG, paired with the rPPG windows.
%   [tRef, hrRef, ref] = referenceHR(gt, traces, cfg)
%   The contact PPG is interpolated (linear, by time) onto the video time
%   base traces.t, preprocessed with the same chain as the rPPG signal and
%   passed through the same estimateHR, so windows match one to one.
%   ref.ppg       : preprocessed PPG on traces.t
%   ref.spec      : spectra from estimateHR
%   ref.hrSensor  : mean oximeter HR per window (sanity check only)

t = traces.t;

% GT timestamps are jittery oximeter arrival times (ms resolution) and can
% repeat; merge duplicates by averaging so the time axis is strictly increasing.
[tg, ~, ic] = unique(gt.t);
if numel(tg) < numel(gt.t)
    fprintf('referenceHR: merged %d duplicate GT timestamp(s).\n', numel(gt.t) - numel(tg));
end
ppgG = accumarray(ic, gt.ppg, [], @mean);
hrG  = accumarray(ic, gt.hrSensor, [], @mean);

ppg = interp1(tg, ppgG, t, 'linear');
nOut = nnz(isnan(ppg));
if nOut > 0
    % Video frames slightly beyond the last GT timestamp: hold edge values.
    ppg = fillmissing(ppg, 'nearest');
    fprintf('referenceHR: %d frame(s) outside GT time range, edge-held.\n', nOut);
end

ref.ppg = preprocessPulse(ppg, traces.fps, cfg);
[tRef, hrRef, ref.spec] = estimateHR(ref.ppg, traces.fps, cfg);

hrS = fillmissing(interp1(tg, hrG, t, 'linear'), 'nearest');
L = ref.spec.winLen;
ref.hrSensor = arrayfun(@(s) mean(hrS(s:s+L-1)), ref.spec.starts);
end
