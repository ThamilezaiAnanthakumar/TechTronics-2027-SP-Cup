function y = detrendSmoothnessPriors(x, lambda)
%DETRENDSMOOTHNESSPRIORS Smoothness-priors detrending (Tarvainen et al., 2002).
%   y = detrendSmoothnessPriors(x, lambda)
%   x : N x 1 signal. Returns x minus the trend
%       trend = (I + lambda^2 * D2' * D2) \ x
%   with D2 the second-order difference matrix (sparse, so O(N)).

x = x(:);
N = numel(x);
if N < 3
    y = x - mean(x);
    return
end
e  = ones(N, 1);
D2 = spdiags([e -2*e e], 0:2, N-2, N);
I  = speye(N);
trend = (I + lambda^2 * (D2' * D2)) \ x;
y = x - trend;
end
