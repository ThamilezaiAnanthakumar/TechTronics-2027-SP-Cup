function A = anatomicalPatchTraces(videoPath, opts)
%ANATOMICALPATCHTRACES Per-frame RGB traces of the 10 anatomical patches (README 4.2b / 4.3).
%   A = anatomicalPatchTraces(videoPath)
%   A = anatomicalPatchTraces(videoPath, MaxFrames=900, CacheFile="...mat")
%   Landmarks: MediaPipe FaceLandmarker (478 points) every frame. Target: the largest face in
%   the first frame, then the face whose landmark centroid is nearest the previous target
%   (re-acquired only within 0.5 face widths). Landmarks are 1-Euro smoothed (min cutoff 1 Hz,
%   beta 0.02; Sulojan's OneEuroFilter). Patch polygons: shared/patches.json (Li et al. 2026).
%   Pixel selection per patch and frame: skin mask (face_config.json YCbCr box), specular
%   removal (Y > 0.95*255 or any channel >= 250), 10 %/10 % trimmed mean.
%   Output: A.t (Nx1 s), A.rgb (Nx3xP), A.npix (NxP), A.valid (NxP), A.names (1xP), A.groups.
%   KeepPolygons=true additionally returns A.poly (NxP cell of Kx2 [x y] image polygons, empty
%   where no target face); off by default, all other outputs are identical either way.
arguments
    videoPath (1,1) string {mustBeFile}
    opts.MaxFrames (1,1) double = Inf
    opts.CacheFile (1,1) string = ""
    opts.NpixMin (1,1) double = 20
    opts.KeepPolygons (1,1) logical = false
end
if opts.CacheFile ~= "" && isfile(opts.CacheFile)
    S = load(opts.CacheFile, 'A');
    if ~opts.KeepPolygons || isfield(S.A, 'poly'), A = S.A; return; end
end
root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
J = jsondecode(fileread(fullfile(root, 'shared', 'patches.json')));
nP = numel(J.patches);
idx = arrayfun(@(p) p.mediapipe(:) + 1, J.patches, 'UniformOutput', false);   % 0-based -> 1-based
names = string({J.patches.name}); groups = string({J.patches.group});
p = face_params(fullfile(root, 'config', 'face_config.json'));
lmk = MediaPipeLandmarker();
smoother = OneEuroFilter(1.0, 0.02, 1.0);

rng(0, 'twister');                       % deterministic (no random calls expected; kept for safety)
v = VideoReader(videoPath);
n = min(opts.MaxFrames, v.NumFrames);
A.t = zeros(n, 1); A.rgb = NaN(n, 3, nP); A.npix = zeros(n, nP); A.valid = false(n, nP);
if opts.KeepPolygons, A.poly = cell(n, nP); end
prevC = []; prevW = []; k = 0;
while hasFrame(v) && k < n
    k = k + 1;
    A.t(k) = v.CurrentTime;
    I = readFrame(v);
    faces = lmk.detect(I, 1000 * A.t(k));
    if isempty(faces), smoother.reset(); prevC = []; continue; end
    C = cellfun(@(f) mean(f, 1), faces, 'UniformOutput', false); C = vertcat(C{:});
    Wd = cellfun(@(f) max(f(:, 1)) - min(f(:, 1)), faces);
    if isempty(prevC)
        if k > 1 && ~isempty(prevW), [d, j] = min(vecnorm(C - lastC, 2, 2)); if d > 0.5 * prevW, continue; end
        else, [~, j] = max(Wd); end
    else
        [d, j] = min(vecnorm(C - prevC, 2, 2));
        if d > 0.5 * prevW, smoother.reset(); prevC = []; continue; end
    end
    prevC = C(j, :); lastC = prevC; prevW = Wd(j);
    L = reshape(smoother.update(reshape(faces{j}, 1, []), A.t(k)), [], 2);
    D = reshape(double(I), [], 3);
    Y = 0.299 * D(:, 1) + 0.587 * D(:, 2) + 0.114 * D(:, 3);
    sk = skin_mask(I, p); sk = sk(:) & Y <= 0.95 * 255 & all(D < 250, 2);
    for q = 1:nP
        poly = L(idx{q}, :);
        if opts.KeepPolygons, A.poly{k, q} = poly; end
        M = poly2mask(poly(:, 1), poly(:, 2), v.Height, v.Width);
        sel = M(:) & sk;
        A.npix(k, q) = nnz(sel);
        if A.npix(k, q) >= opts.NpixMin
            A.rgb(k, :, q) = trimmean(D(sel, :), 20, 1);
            A.valid(k, q) = true;
        end
    end
end
A.t = A.t(1:k); A.rgb = A.rgb(1:k, :, :); A.npix = A.npix(1:k, :); A.valid = A.valid(1:k, :);
if opts.KeepPolygons, A.poly = A.poly(1:k, :); end
A.names = names; A.groups = groups; A.video = videoPath;
if opts.CacheFile ~= ""
    if ~isfolder(fileparts(opts.CacheFile)), mkdir(fileparts(opts.CacheFile)); end
    save(opts.CacheFile, 'A', '-v7');
end
end
