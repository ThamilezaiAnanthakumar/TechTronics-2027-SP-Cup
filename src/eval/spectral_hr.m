function [hr, tc] = spectral_hr(x, tu, fs, opts)
%SPECTRAL_HR  HR (bpm) from the spectral peak in sliding windows.
%   [hr, tc] = spectral_hr(x, tu, fs, Win=10, Step=1, Band=[0.7 3], Centers=[])
%   x, tu  uniformly sampled signal and its times (s);  hr, tc  Kx1

arguments
    x (:,1) double
    tu (:,1) double
    fs (1,1) double
    opts.Win (1,1) double = 10
    opts.Step (1,1) double = 1
    opts.Band (1,2) double = [0.7 3.0]
    opts.Centers double = []
end

tc = opts.Centers(:);
if isempty(tc)
    tc = (tu(1) + opts.Win/2 : opts.Step : tu(end) - opts.Win/2).';
end
hr = NaN(numel(tc), 1);
L = round(opts.Win * fs);
nfft = 2^nextpow2(max(8192, L));
for i = 1:numel(tc)
    seg = x(tu >= tc(i) - opts.Win/2 & tu < tc(i) + opts.Win/2);
    if numel(seg) < 0.9 * L || any(~isfinite(seg)), continue; end
    [P, f] = periodogram(seg - mean(seg), hann(numel(seg)), nfft, fs);
    inBand = f >= opts.Band(1) & f <= opts.Band(2);
    fb = f(inBand);
    [~, j] = max(P(inBand));
    hr(i) = 60 * fb(j);
end
end
