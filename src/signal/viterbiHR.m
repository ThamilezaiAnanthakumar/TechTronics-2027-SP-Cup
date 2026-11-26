function [hr, conf, grid] = viterbiHR(P, f, opts)
%VITERBIHR Physiology-constrained HR track over a spectrogram (README Section 4.8.3, offline mode).
%   [hr, conf] = viterbiHR(P, f)
%   [hr, conf] = viterbiHR(P, f, Step=1, Grid=[42 180 0.5], MaxDown=3, MaxUp=6, Lambda=0.1)
%   P     nWin x nf  power spectra (any scaling), one row per analysis window
%   f     nf x 1     frequencies of P (Hz)
%   hr    nWin x 1   HR path (bpm), refined by parabolic interpolation on the emission
%   conf  nWin x 1   share of the window's (normalised) spectral mass within +-3 bpm of the path
%
%   States: HR grid (bpm). Emission: log(H_t(h) + eps) with H_t normalised to unit sum per window.
%   Transition h -> h' over Step seconds is allowed only if -MaxDown <= (h'-h)/Step <= MaxUp
%   (Tian et al. 2026: ~3 bpm/s decrease, ~6 bpm/s increase); inside that range the cost is
%   Lambda*|h'-h|. Full Viterbi (offline), so a bad start can be recovered later in the record.
arguments
    P double
    f (:,1) double
    opts.Step (1,1) double = 1
    opts.Grid (1,3) double = [42 180 0.5]
    opts.MaxDown (1,1) double = 3
    opts.MaxUp (1,1) double = 6
    opts.Lambda (1,1) double = 0.1
end
grid = (opts.Grid(1):opts.Grid(3):opts.Grid(2))';
nS = numel(grid); nW = size(P, 1);
E = zeros(nW, nS);
for w = 1:nW
    h = interp1(60 * f, P(w, :)', grid, 'linear', 0);
    h = max(h, 0); h = h / max(sum(h), eps);
    E(w, :) = log(h' + 1e-12);
end
d = grid' - grid;                                  % d(i,j) = h_j - h_i (from i to j)
allowed = d >= -opts.MaxDown * opts.Step & d <= opts.MaxUp * opts.Step;
T = -opts.Lambda * abs(d); T(~allowed) = -Inf;     % log transition score
score = E(1, :); back = zeros(nW, nS, 'uint32');
for w = 2:nW
    [best, arg] = max(score' + T, [], 1);           % over previous state i
    score = best + E(w, :);
    back(w, :) = arg;
end
path = zeros(nW, 1); [~, path(nW)] = max(score);
for w = nW:-1:2, path(w-1) = back(w, path(w)); end
hr = grid(path); conf = zeros(nW, 1);
for w = 1:nW
    e = exp(E(w, :)); k = path(w);
    if k > 1 && k < nS
        a = e(k-1); b = e(k); c = e(k+1); den = a - 2*b + c;
        if den ~= 0, hr(w) = hr(w) + opts.Grid(3) * max(-0.5, min(0.5, 0.5 * (a - c) / den)); end
    end
    conf(w) = sum(e(abs(grid - grid(k)) <= 3));
end
end
