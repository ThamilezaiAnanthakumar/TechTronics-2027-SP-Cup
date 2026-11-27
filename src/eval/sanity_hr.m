function [hr, tc] = sanity_hr(t, rgb, opts)
%SANITY_HR  Baseline HR from RGB traces (POS), used to benchmark the face module.
%   [hr, tc] = sanity_hr(t, rgb, Win=10, Step=1, Band=[0.7 3], Centers=[])
%   t Nx1 (s), rgb Nx3 (NaN rows allowed). Resample to 30 Hz -> POS
%   (Wang et al., IEEE TBME 2017) -> Butterworth band-pass -> spectral_hr.

arguments
    t (:,1) double
    rgb (:,3) double
    opts.Win (1,1) double = 10
    opts.Step (1,1) double = 1
    opts.Band (1,2) double = [0.7 3.0]
    opts.Centers double = []
end

fs = 30;
valid = all(isfinite(rgb), 2) & isfinite(t);
if nnz(valid) < opts.Win * fs * 0.9
    tc = opts.Centers(:);  hr = NaN(numel(tc), 1);
    return
end
tv = t(valid);
tu = (tv(1) : 1/fs : tv(end)).';
h  = pos_rppg(interp1(tv, rgb(valid, :), tu, "linear"), fs);
[b, a] = butter(3, opts.Band / (fs/2), "bandpass");
h = filtfilt(b, a, h);
[hr, tc] = spectral_hr(h, tu, fs, Win=opts.Win, Step=opts.Step, Band=opts.Band, Centers=opts.Centers);
end

function h = pos_rppg(X, fs)
N = size(X, 1);
l = round(1.6 * fs);
h = zeros(N, 1);
Pm = [0 1 -1; -2 1 1];
for n = 1:N - l + 1
    C  = X(n:n + l - 1, :).';
    S  = Pm * (C ./ mean(C, 2));
    sd = std(S(2, :));
    if sd == 0, continue; end
    pp = S(1, :) + (std(S(1, :)) / sd) * S(2, :);
    h(n:n + l - 1) = h(n:n + l - 1) + (pp - mean(pp)).';
end
end
