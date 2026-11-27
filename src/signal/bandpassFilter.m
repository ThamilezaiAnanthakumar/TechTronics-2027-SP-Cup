function y = bandpassFilter(x, fs, cfg)
%BANDPASSFILTER Zero-phase Butterworth bandpass over cfg.hrBand.
%   y = bandpassFilter(x, fs, cfg)
%   x : N x M (filters each column), fs in Hz.
%   Order cfg.filterOrder, applied forward-backward with filtfilt (SOS form).

[z, p, k] = butter(cfg.filterOrder, cfg.hrBand / (fs/2), 'bandpass');
[sos, g]  = zp2sos(z, p, k);
y = filtfilt(sos, g, x);
end
