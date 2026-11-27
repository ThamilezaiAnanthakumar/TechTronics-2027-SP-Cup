function p = rgbToPulse(rgb, fs, method, cfg)
%RGBTOPULSE Convert an RGB trace into a pulse signal.
%   p = rgbToPulse(rgb, fs, method, cfg)
%   rgb    : N x 3 spatial-mean [R G B] per frame
%   method : 'GREEN' | 'CHROM' (de Haan & Jeanne 2013) | 'POS' (Wang et al. 2017)
%   p      : N x 1 pulse signal (not yet detrended / bandpassed)

rgb = double(rgb);
N = size(rgb, 1);
if size(rgb, 2) ~= 3
    error('rgbToPulse:size', 'rgb must be N x 3, got %d x %d.', size(rgb,1), size(rgb,2));
end

switch upper(method)
    case 'GREEN'
        g = rgb(:, 2);
        p = g / mean(g) - 1;

    case 'CHROM'
        Cn = rgb ./ mean(rgb, 1);
        X = 3*Cn(:,1) - 2*Cn(:,2);
        Y = 1.5*Cn(:,1) + Cn(:,2) - 1.5*Cn(:,3);
        Xf = bandpassFilter(X, fs, cfg);
        Yf = bandpassFilter(Y, fs, cfg);
        alpha = std(Xf) / std(Yf);
        p = Xf - alpha * Yf;

    case 'POS'
        l = round(cfg.posWindowSec * fs);
        P = [0 1 -1; -2 1 1];
        H = zeros(N, 1);
        for n = 1:(N - l + 1)
            C  = rgb(n:n+l-1, :)';            % 3 x l
            Cn = C ./ mean(C, 2);             % temporal normalisation
            S  = P * Cn;                      % 2 x l projection
            s2 = std(S(2,:));
            if s2 == 0, continue; end
            h  = S(1,:) + (std(S(1,:)) / s2) * S(2,:);
            H(n:n+l-1) = H(n:n+l-1) + (h - mean(h))';   % overlap-add
        end
        p = H;

    otherwise
        error('rgbToPulse:method', 'Unknown method "%s".', method);
end
p = p(:);
end
