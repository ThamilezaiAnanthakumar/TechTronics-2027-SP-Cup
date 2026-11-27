function [tHR, hr, spec] = estimateHR(signal, fs, cfg)
%ESTIMATEHR Sliding-window spectral-peak heart rate.
%   [tHR, hr, spec] = estimateHR(signal, fs, cfg)
%   Windows of cfg.hrWindowSec, step cfg.hrStepSec. Per window: remove mean,
%   Hann window, zero-padded FFT, strongest peak inside cfg.hrBand,
%   parabolic interpolation around the peak bin, HR = 60*f.
%   tHR  : window-centre times (s), relative to sample 1 at t = 0
%   hr   : HR (bpm), column
%   spec : .f (Hz, band only), .P (nWin x nf power), .fPeak (Hz)

x = signal(:);
N = numel(x);
L = round(cfg.hrWindowSec * fs);
S = max(1, round(cfg.hrStepSec * fs));
if N < L
    error('estimateHR:short', 'Signal (%d samples) shorter than one window (%d).', N, L);
end

starts = (1:S:(N - L + 1))';
nWin = numel(starts);
nfft = max(cfg.nfftMin, 2^nextpow2(L));
f = (0:nfft/2)' * fs / nfft;
band = find(f >= cfg.hrBand(1) & f <= cfg.hrBand(2));
w = hann(L);

hr = zeros(nWin, 1);
fPeak = zeros(nWin, 1);
P = zeros(nWin, numel(band));
for i = 1:nWin
    seg = x(starts(i):starts(i)+L-1);
    seg = (seg - mean(seg)) .* w;
    X = fft(seg, nfft);
    pw = abs(X(1:nfft/2+1)).^2;
    P(i, :) = pw(band);

    [~, j] = max(pw(band));
    k = band(j);
    delta = 0;
    if k > 1 && k < numel(pw)
        a = pw(k-1); b = pw(k); c = pw(k+1);
        den = a - 2*b + c;
        if den ~= 0
            delta = 0.5 * (a - c) / den;   % in bins, |delta| <= 0.5
        end
    end
    fPeak(i) = (k - 1 + delta) * fs / nfft;
    hr(i) = 60 * fPeak(i);
end

tHR = (starts - 1 + (L - 1)/2) / fs;
spec = struct('f', f(band), 'P', P, 'fPeak', fPeak, 'nfft', nfft, ...
              'winLen', L, 'starts', starts);
end
