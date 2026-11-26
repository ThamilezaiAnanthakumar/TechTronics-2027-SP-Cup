function y = kalman_denoise(x, fs, opts)
%KALMAN_DENOISE  Causal Kalman denoising of frame-mean RGB traces (per channel).
%   y = kalman_denoise(x, fs)
%   y = kalman_denoise(x, fs, Npix=npix, Cutoff=4)
%
%   x     NxC frame means (e.g. rgb_face), NaN rows allowed (kept NaN)
%   fs    frame rate (Hz)
%   Npix  Nx1 skin pixels behind each frame mean (optional)
%
%   Model per channel: level = random walk (process noise Q),
%   measurement = level + noise R_k. R_k = R0 * median(Npix) / Npix(k): a mean
%   over fewer skin pixels is noisier, so it is trusted less. R0 is estimated
%   from the frame-to-frame differences (var(diff)/2). Q is set so that, at
%   the median pixel count, the steady-state filter is a first-order low-pass
%   with its -3 dB point at Cutoff Hz (default 4 Hz, above the 0.7-3 Hz HR
%   band), so the pulse itself is kept while frame-level noise is reduced.

arguments
    x double
    fs (1,1) double {mustBePositive}
    opts.Npix (:,1) double = []
    opts.Cutoff (1,1) double {mustBePositive} = 4
end

[N, C] = size(x);
% steady-state gain alpha of a first-order smoother with -3 dB at Cutoff
% README 4.4: -3 dB at min(Cutoff, 0.8*fs/2) (was 0.45*fs; changed 26 Nov 2026, CHANGELOG_PARAMS.md)
w = 2 * pi * min(opts.Cutoff, 0.4 * fs) / fs;
c = 4 - 2 * cos(w);
beta  = (c - sqrt(c^2 - 4)) / 2;
alpha = 1 - beta;

if isempty(opts.Npix)
    scale = ones(N, 1);
else
    np = opts.Npix(:);
    scale = median(np(np > 0)) ./ max(np, 1);
end

y = NaN(N, C);
for ch = 1:C
    v = x(:, ch);
    d = diff(v);  d = d(isfinite(d));
    R0 = var(d) / 2;
    if ~(R0 > 0), y(:, ch) = v; continue; end
    Q = R0 * alpha^2 / (1 - alpha);
    s = NaN;  P = NaN;
    for k = 1:N
        if ~isnan(s), P = P + Q; end                   % predict
        if ~isfinite(v(k)), continue; end
        if isnan(s)
            s = v(k);  P = R0 * scale(k);              % initialise on first valid frame
        else
            R = R0 * scale(k);
            G = P / (P + R);                           % update
            s = s + G * (v(k) - s);
            P = (1 - G) * P;
        end
        y(k, ch) = s;
    end
end
end
