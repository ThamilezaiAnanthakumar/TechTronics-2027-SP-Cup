function [hrAvgSpec, hrMedian] = videoHR(spec, hrWin)
%VIDEOHR One HR value per video from the sliding-window spectra.
%   [hrAvgSpec, hrMedian] = videoHR(spec, hrWin)
%   spec  : spectra struct from estimateHR (.f band frequencies, .P nWin x nf)
%   hrWin : per-window HR from estimateHR (bpm)
%   hrAvgSpec : peak of the average of the per-window spectra (each window
%               normalised to unit total power so no window dominates),
%               refined with parabolic interpolation. Primary estimate.
%   hrMedian  : median of the per-window HR values (secondary).
%   Used when the reference is a single mean HR per video (AiPEX).

P = spec.P ./ sum(spec.P, 2);
m = mean(P, 1);
[~, k] = max(m);
delta = 0;
if k > 1 && k < numel(m)
    a = m(k-1); b = m(k); c = m(k+1);
    den = a - 2*b + c;
    if den ~= 0, delta = 0.5 * (a - c) / den; end
end
df = spec.f(2) - spec.f(1);
hrAvgSpec = 60 * (spec.f(k) + delta * df);
hrMedian = median(hrWin);
end
