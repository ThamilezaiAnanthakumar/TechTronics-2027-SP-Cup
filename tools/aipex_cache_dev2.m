function aipex_cache_dev2(opts)
%AIPEX_CACHE_DEV2 Extra per-frame pixel statistics of the AiPEX-dev crops (ladder item 12, README 4.3).
%   Per crop and frame, over the skin pixels (face module skin_mask) minus specular pixels
%   (Y > 0.95*255 or any channel >= 250):
%     rgbTrim  Nx3  10 %/10 % trimmed mean per channel
%     rgbTex   Nx3  trimmed mean after removing the top 15 % of pixels by gradient magnitude
%                   (Gaussian sigma 1 px on luma) — texture / wrinkle mask
%     npixTex  Nx1  pixels kept by the texture mask
%   Saved to <TraceRoot>/<Country>_<video>.mat as R2. Dev videos only.
arguments
    opts.TraceRoot (1,1) string = "D:\SPCup Data\work\aipex_iter\traces2"
    opts.Parallel (1,1) logical = true
    opts.Split (1,1) string {mustBeMember(opts.Split, ["dev" "test"])} = "dev"   % test: cache only, after-freeze use
end
root = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root, 'config')); addpath(genpath(fullfile(root, 'src')));
cfg = getConfig();
p = face_params(fullfile(root, 'config', 'face_config.json'));
ids = readlines(fullfile(root, 'shared', 'splits', "aipex_" + opts.Split + ".txt"), 'EmptyLineRule', 'skip');
if ~isfolder(opts.TraceRoot), mkdir(opts.TraceRoot); end
regions = ["forehead" "leftcheek" "rightcheek"];
aroot = string(cfg.aipexRoot); troot = opts.TraceRoot;
nW = 0; if opts.Parallel, nW = Inf; end
parfor (i = 1:numel(ids), nW)
    parts = split(ids(i), '/');
    out = fullfile(troot, parts(1) + "_" + parts(2) + ".mat");
    if isfile(out), continue; end
    R2 = struct();
    for r = regions
        R2.(r) = readCrop2(fullfile(aroot, 'Data', parts(1), parts(2) + "_" + r + ".avi"), p);
    end
    saveR2(out, R2);
    fprintf('%s done\n', ids(i));
end
fprintf('cache2 complete: %s\n', troot);
end

function Q = readCrop2(path, p)
v = VideoReader(path);
n = v.NumFrames;
Q.rgbTrim = NaN(n, 3); Q.rgbTex = NaN(n, 3); Q.npixTex = zeros(n, 1);
g = fspecial('gaussian', 7, 1);
k = 0;
while hasFrame(v)
    k = k + 1;
    I = readFrame(v);
    D = reshape(double(I), [], 3);
    Y = 0.299 * D(:, 1) + 0.587 * D(:, 2) + 0.114 * D(:, 3);
    M = skin_mask(I, p); M = M(:) & Y <= 0.95 * 255 & all(D < 250, 2);
    if nnz(M) >= 10
        Q.rgbTrim(k, :) = trimmean(D(M, :), 20, 1);
        Yi = imfilter(reshape(Y, v.Height, v.Width), g, 'replicate');
        [gx, gy] = gradient(Yi); G = hypot(gx, gy); G = G(:);
        thr = prctile(G(M), 85);
        Mt = M & G <= thr;
        Q.npixTex(k) = nnz(Mt);
        if nnz(Mt) >= 10, Q.rgbTex(k, :) = trimmean(D(Mt, :), 20, 1); end
    end
end
Q.rgbTrim = Q.rgbTrim(1:k, :); Q.rgbTex = Q.rgbTex(1:k, :); Q.npixTex = Q.npixTex(1:k);
end

function saveR2(out, R2) %#ok<INUSD>
save(out, 'R2', '-v7');
end
