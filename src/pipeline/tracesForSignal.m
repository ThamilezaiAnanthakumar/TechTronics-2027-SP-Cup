function traces = tracesForSignal(fv, cfg, denoise)
%TRACESFORSIGNAL Bridge: face-module output -> uniform RGB traces for the signal chain.
%   traces = tracesForSignal(fv, cfg, denoise)
%   fv      : process_video output (Sulojan). Real, possibly uneven frame
%             times fv.t; NaN RGB on lost frames; per-region npix/occluded.
%   denoise : 'none' | 'kalman' (Thamil's kalman_denoise, pixel-count weighted)
%
%   For every ROI in cfg.intRois ('face' = fv.rgb_face, others = fv.rgb_<roi>):
%     1. frames are invalid when the target is lost (quality 0), the region
%        is occluded (not applied to 'face', which already excludes occluded
%        regions) or the RGB is not finite; their RGB is set to NaN
%     2. optional Kalman denoising on the native frame times
%     3. linear resampling onto a uniform cfg.resampleFs grid; gaps are
%        bridged by the interpolation and edges held
%   traces.t (N x 1, s), traces.fps, traces.rgb.(roi) (N x 3),
%   traces.valid.(roi) (N x 1 logical, nearest original frame valid),
%   traces.denoise, traces.validFrac.(roi)

t = fv.t(:);
fsNative = 1 / median(diff(t));
names = string(fv.roi_names);
q = fv.quality(:) > 0;

fs = cfg.resampleFs;
tu = (t(1) : 1/fs : t(end))';

traces.t = tu;
traces.fps = fs;
traces.denoise = denoise;
traces.fsNative = fsNative;
for roi = string(cfg.intRois)
    if roi == "face"
        x = fv.rgb_face;
        npix = sum(fv.npix .* ~fv.occluded, 2);
        valid = q;
    else
        r = find(names == roi, 1);
        if isempty(r)
            error('tracesForSignal:roi', 'ROI "%s" not in face module output (%s).', ...
                roi, strjoin(names, ', '));
        end
        x = fv.("rgb_" + roi);
        npix = fv.npix(:, r);
        valid = q & ~fv.occluded(:, r);
    end
    valid = valid & all(isfinite(x), 2) & npix > 0;
    x(~valid, :) = NaN;

    switch lower(denoise)
        case 'none'
        case 'kalman'
            x = kalman_denoise(x, fsNative, Npix=npix, Cutoff=cfg.kalman.cutoff);
        otherwise
            error('tracesForSignal:denoise', 'Unknown denoise "%s".', denoise);
    end

    if nnz(valid) >= 2
        xu = interp1(t(valid), x(valid, :), tu, 'linear');
        xu = fillmissing(xu, 'nearest');
        vu = interp1(t, double(valid), tu, 'nearest', 0) > 0.5;
    else
        xu = NaN(numel(tu), 3);
        vu = false(numel(tu), 1);
    end
    traces.rgb.(roi) = xu;
    traces.valid.(roi) = vu;
    traces.validFrac.(roi) = mean(vu);
end
end
