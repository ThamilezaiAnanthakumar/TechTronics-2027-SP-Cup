function m = computeMetrics(hrEst, hrRef)
%COMPUTEMETRICS Agreement metrics between paired HR vectors (bpm).
%   m = computeMetrics(hrEst, hrRef)
%   m.MAE, m.RMSE (bpm), m.MAPE (%), m.r, m.p (Pearson),
%   m.bias, m.loaLow, m.loaHigh (Bland-Altman, d = est - ref), m.nWindows

hrEst = hrEst(:); hrRef = hrRef(:);
if numel(hrEst) ~= numel(hrRef)
    error('computeMetrics:length', 'Length mismatch: est %d vs ref %d.', ...
        numel(hrEst), numel(hrRef));
end
d = hrEst - hrRef;
m.MAE  = mean(abs(d));
m.RMSE = sqrt(mean(d.^2));
m.MAPE = 100 * mean(abs(d) ./ hrRef);
if numel(d) >= 3 && std(hrEst) > 0 && std(hrRef) > 0
    [R, P] = corrcoef(hrEst, hrRef);
    m.r = R(1,2); m.p = P(1,2);
else
    m.r = NaN; m.p = NaN;
end
m.bias    = mean(d);
m.loaLow  = m.bias - 1.96 * std(d);
m.loaHigh = m.bias + 1.96 * std(d);
m.nWindows = numel(d);
end
