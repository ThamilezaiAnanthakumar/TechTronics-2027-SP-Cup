function pf = preprocessPulse(p, fs, cfg)
%PREPROCESSPULSE Detrend, bandpass and z-normalise a pulse signal.
%   pf = preprocessPulse(p, fs, cfg)
%   1. smoothness-priors detrend (cfg.detrendLambda)
%   2. zero-phase Butterworth bandpass (cfg.hrBand, cfg.filterOrder)
%   3. zero mean, unit variance
%   Used for both the rPPG signal and the contact-PPG reference.

pf = detrendSmoothnessPriors(p(:), cfg.detrendLambda);
pf = bandpassFilter(pf, fs, cfg);
sd = std(pf);
if sd > 0
    pf = (pf - mean(pf)) / sd;
else
    pf = pf - mean(pf);
end
end
