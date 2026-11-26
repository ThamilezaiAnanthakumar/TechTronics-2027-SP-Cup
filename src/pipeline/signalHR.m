function R = signalHR(traces, hrRef, cfg)
%SIGNALHR Muftee's signal chain on every method x ROI of one set of traces.
%   R = signalHR(traces, hrRef, cfg)
%   traces : tracesForSignal output (uniform grid, .valid per ROI)
%   hrRef  : reference HR per window (referenceHR on the same traces.t)
%   R      : struct array, one entry per cfg.methods x cfg.intRois:
%            .method .roi .denoise .tHR .hr .validFrac .m (computeMetrics)
%
%   Per pair: rgbToPulse -> preprocessPulse (detrend, bandpass, z-score)
%   -> estimateHR (10 s windows, 1 s step). A window's HR is set to NaN when
%   fewer than cfg.minValidFrac of its frames were valid in the face module
%   (lost target / occluded region); metrics skip those windows.

R = struct('method', {}, 'roi', {}, 'denoise', {}, 'tHR', {}, 'hr', {}, ...
           'winValid', {}, 'm', {});
fs = traces.fps;
for i = 1:numel(cfg.methods)
    for j = 1:numel(cfg.intRois)
        method = cfg.methods{i};
        roi = cfg.intRois{j};
        rgb = traces.rgb.(roi);
        if any(~isfinite(rgb(:)))
            hr = NaN(size(hrRef)); tHR = NaN(size(hrRef)); winOK = false(size(hrRef));
        else
            p  = rgbToPulse(rgb, fs, method, cfg);
            pf = preprocessPulse(p, fs, cfg);
            [tHR, hr, spec] = estimateHR(pf, fs, cfg);
            v = traces.valid.(roi);
            L = spec.winLen;
            winOK = arrayfun(@(s) mean(v(s:s+L-1)), spec.starts) >= cfg.minValidFrac;
            hr(~winOK) = NaN;
        end
        m = computeMetrics(hr, hrRef);
        R(end+1) = struct('method', method, 'roi', roi, 'denoise', traces.denoise, ...
                          'tHR', tHR, 'hr', hr, 'winValid', winOK, 'm', m); %#ok<AGROW>
    end
end
end
