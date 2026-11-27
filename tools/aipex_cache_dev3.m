function aipex_cache_dev3(opts)
%AIPEX_CACHE_DEV3 Per-frame motion of each AiPEX-dev crop (ladder item 13, README 4.8.2).
%   Inter-frame translation (dx, dy, px) of each crop by phase correlation of the luma
%   (Hann-windowed FFT cross-power spectrum, integer peak + parabolic sub-pixel refinement).
%   Saved to <TraceRoot>/<Country>_<video>.mat as M.(region) = [dx dy] (N x 2, first row 0).
arguments
    opts.TraceRoot (1,1) string = "D:\SPCup Data\work\aipex_iter\traces3"
    opts.Parallel (1,1) logical = true
end
root = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root, 'config')); cfg = getConfig();
ids = readlines(fullfile(root, 'shared', 'splits', 'aipex_dev.txt'), 'EmptyLineRule', 'skip');
if ~isfolder(opts.TraceRoot), mkdir(opts.TraceRoot); end
regions = ["forehead" "leftcheek" "rightcheek"];
aroot = string(cfg.aipexRoot); troot = opts.TraceRoot;
nW = 0; if opts.Parallel, nW = Inf; end
parfor (i = 1:numel(ids), nW)
    parts = split(ids(i), '/');
    out = fullfile(troot, parts(1) + "_" + parts(2) + ".mat");
    if isfile(out), continue; end
    M = struct();
    for r = regions
        M.(r) = cropMotion(fullfile(aroot, 'Data', parts(1), parts(2) + "_" + r + ".avi"));
    end
    saveM(out, M);
    fprintf('%s done\n', ids(i));
end
fprintf('cache3 complete: %s\n', troot);
end

function D = cropMotion(path)
v = VideoReader(path);
n = v.NumFrames; D = zeros(n, 2);
w = hann(v.Height) * hann(v.Width)';
prev = []; k = 0;
while hasFrame(v)
    k = k + 1;
    I = double(im2gray(readFrame(v)));
    F = fft2((I - mean(I(:))) .* w);
    if ~isempty(prev)
        X = F .* conj(prev); X = X ./ max(abs(X), eps);
        c = real(ifft2(X));
        [~, j] = max(c(:)); [r0, c0] = ind2sub(size(c), j);
        dy = subpix(c, r0, c0, 1); dx = subpix(c, r0, c0, 2);
        H = size(c, 1); W = size(c, 2);
        dy = mod(r0 - 1 + dy + H/2, H) - H/2; dx = mod(c0 - 1 + dx + W/2, W) - W/2;
        D(k, :) = [dx dy];
    end
    prev = F;
end
D = D(1:k, :);
end

function d = subpix(c, r0, c0, dim)
[H, W] = size(c);
if dim == 1, a = c(mod(r0 - 2, H) + 1, c0); b = c(r0, c0); e = c(mod(r0, H) + 1, c0);
else,        a = c(r0, mod(c0 - 2, W) + 1); b = c(r0, c0); e = c(r0, mod(c0, W) + 1); end
den = a - 2 * b + e; d = 0;
if den ~= 0, d = max(-0.5, min(0.5, 0.5 * (a - e) / den)); end
end

function saveM(out, M) %#ok<INUSD>
save(out, 'M', '-v7');
end
