function aipex_cache_dev(opts)
%AIPEX_CACHE_DEV Per-frame traces of the AiPEX-dev crops, cached once for the iteration loop.
%   aipex_cache_dev()                      all 62 dev videos (shared/splits/aipex_dev.txt)
%   aipex_cache_dev(Parallel=false)
%   For every dev video and each crop (forehead, leftcheek, rightcheek), per frame:
%     rgbAll  Nx3  mean RGB of all pixels
%     rgbSkin Nx3  mean RGB of skin pixels (face module skin_mask, face_config.json), NaN if none
%     npix    Nx1  number of skin pixels;  nAll = pixels per frame
%     t       Nx1  frame times from VideoReader.CurrentTime (15 fps in all dev files, see LOG E001)
%   Saved to <traceRoot>/<Country>_<video>.mat. Never opens AiPEX-test videos.
arguments
    opts.TraceRoot (1,1) string = "D:\SPCup Data\work\aipex_iter\traces"
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
    R = struct();
    for r = regions
        R.(r) = readCrop(fullfile(aroot, 'Data', parts(1), parts(2) + "_" + r + ".avi"), p);
    end
    saveTrace(out, R);
    fprintf('%s done\n', ids(i));
end
fprintf('cache complete: %s\n', troot);
end

function Q = readCrop(path, p)
v = VideoReader(path);
n = v.NumFrames;
Q.t = zeros(n, 1); Q.rgbAll = NaN(n, 3); Q.rgbSkin = NaN(n, 3); Q.npix = zeros(n, 1);
k = 0;
while hasFrame(v)
    k = k + 1;
    Q.t(k) = v.CurrentTime;
    I = readFrame(v);
    D = reshape(double(I), [], 3);
    Q.rgbAll(k, :) = mean(D, 1);
    M = skin_mask(I, p);
    Q.npix(k) = nnz(M);
    if Q.npix(k) > 0, Q.rgbSkin(k, :) = mean(D(M(:), :), 1); end
end
Q.t = Q.t(1:k); Q.rgbAll = Q.rgbAll(1:k, :); Q.rgbSkin = Q.rgbSkin(1:k, :); Q.npix = Q.npix(1:k);
Q.nAll = v.Width * v.Height; Q.fpsHeader = v.FrameRate; Q.path = path;
end

function saveTrace(out, R) %#ok<INUSD>
save(out, 'R', '-v7');
end
