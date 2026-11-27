function frame = degradeFrame(frame, dcfg)
%DEGRADEFRAME Apply simulated capture constraints to one RGB frame.
%   frame = degradeFrame(frame, cfg.degrade)
%   Returns the frame unchanged when dcfg.enabled is false.
%
%   Stub for the constraint tests (C1 low light, C4 low resolution).
%   Supported now: optional resize, optional gain + noise. Extend here.

if ~isfield(dcfg, 'enabled') || ~dcfg.enabled
    return
end

cls = class(frame);
if isfield(dcfg, 'resizeTo') && ~isempty(dcfg.resizeTo)
    frame = imresize(frame, dcfg.resizeTo, 'bicubic');
end
if isfield(dcfg, 'lowLightGain') && ~isempty(dcfg.lowLightGain)
    f = im2double(frame) * dcfg.lowLightGain;
    switch lower(dcfg.noiseType)
        case 'poisson'
            f = imnoise(f, 'poisson');
        case 'gaussian'
            f = imnoise(f, 'gaussian', 0, 1e-4);
    end
    frame = cast(f * double(intmax(cls)), cls);
end
end
