function [T, R, ref] = subjectHR(fv, gt, cfg)
%SUBJECTHR Integrated pipeline for one subject, after the face module.
%   [T, R, ref] = subjectHR(fv, gt, cfg)
%   fv  : process_video output (Sulojan's face module)
%   gt  : loadUBFC ground truth (.t, .ppg, .hrSensor)
%   For each cfg.denoiseSet variant: tracesForSignal (+ Thamil's Kalman)
%   -> signalHR (Muftee's chain) against referenceHR from the contact PPG.
%   T   : one row per denoise x method x ROI with the metrics
%   R   : signalHR entries for all variants
%   ref : reference struct (+ .tRef, .hrRef)

R = [];
ref = [];
for k = 1:numel(cfg.denoiseSet)
    traces = tracesForSignal(fv, cfg, cfg.denoiseSet{k});
    if k == 1   % same time grid for every variant -> one reference
        [tRef, hrRef, ref] = referenceHR(gt, traces, cfg);
        ref.tRef = tRef;
        ref.hrRef = hrRef;
        ref.validFrac = traces.validFrac;
    end
    R = [R, signalHR(traces, ref.hrRef, cfg)]; %#ok<AGROW>
end

M = [R.m];
T = table(string({R.denoise})', string({R.method})', string({R.roi})', ...
    [M.MAE]', [M.RMSE]', [M.MAPE]', [M.r]', [M.SUC3]', [M.SUC5]', ...
    [M.bias]', [M.loaLow]', [M.loaHigh]', [M.nWindows]', [M.coverage]', ...
    'VariableNames', {'Denoise', 'Method', 'ROI', 'MAE', 'RMSE', 'MAPE', 'r', ...
    'SUC3', 'SUC5', 'bias', 'LoA_low', 'LoA_high', 'nWin', 'coverage'});
end
