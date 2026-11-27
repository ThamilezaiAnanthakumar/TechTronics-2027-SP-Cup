function info = aging(videoPath, opts)
%AGING C6 synthetic skin-ageing texture (README 1.4 / 5): imfilter edge kernels applied as
%   localised multiplicative micro-shadows on forehead and cheeks.
%   info = aging(videoPath)                          Strength 0.1
%   info = aging(v, Strength=0.2, MaxFrames=300)     Strength 0.05-0.2 = maximum darkening
%   info = aging(v, FaceBox=[x y w h])               fixed box instead of landmarks (tests)
%   Texture (fixed per video, seeded): a 256x256 field of anisotropic Gaussian-smoothed noise
%   (horizontal creases, sigma [0.8 6] px) is passed through imfilter with
%   fspecial('laplacian',0.2) and fspecial('sobel') (edge emphasis), plus fine pore noise,
%   normalised to its 99th percentile and clipped to [0, 1] (shadows only). Each frame the
%   texture is resized to the face box (landmark bounding box) and applied as
%   J = I .* (1 - Strength * R .* T), R = feathered region mask: forehead (medial + lateral),
%   malar and submalar polygons of shared/patches.json (box mode: forehead and cheek fractions
%   of the face box). Frames before the first face detection are left unchanged.
%   Output: <OutRoot>/Aging_C6_s<strength>/<subject>.mj2 (lossless) + .json sidecar.
arguments
    videoPath (1,1) string {mustBeFile}
    opts.Strength (1,1) double {mustBeInRange(opts.Strength, 0, 0.5)} = 0.1
    opts.Seed (1,1) double = 0
    opts.FaceBox double = []
    opts.Condition (1,1) string = ""
    opts.OutRoot (1,1) string = ""
    opts.Subject (1,1) string = ""
    opts.MaxFrames (1,1) double = Inf
    opts.Profile (1,1) string = "Archival"
end
if opts.OutRoot == "", opts.OutRoot = degradeDefaultRoot(); end
cond = opts.Condition;
if cond == "", cond = sprintf("Aging_C6_s%g", opts.Strength); end
T = agingTexture(opts.Seed);
root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
Jp = jsondecode(fileread(fullfile(root, 'shared', 'patches.json')));
use = ismember({Jp.patches.name}, {'forehead_medial', 'forehead_lat_L', 'forehead_lat_R', ...
    'malar_L', 'malar_R', 'submalar_L', 'submalar_R'});
polyIdx = arrayfun(@(p) p.mediapipe(:) + 1, Jp.patches(use), 'UniformOutput', false);
st = struct('lmk', [], 'smoother', [], 'prevC', [], 'prevW', [], 'last', []);
noFace = 0; darkSum = 0; nDone = 0;
    function J = apply(I, k, t)
        [H, W, ~] = size(I);
        if isempty(opts.FaceBox)
            [L, st] = faceGeomStep(st, I, t);
            if isempty(L), J = I; noFace = noFace + 1; return; end
            R = false(H, W);
            for q = 1:numel(polyIdx)
                R = R | poly2mask(L(polyIdx{q}, 1), L(polyIdx{q}, 2), H, W);
            end
            box = [min(L(:, 1)) min(L(:, 2)) max(L(:, 1)) - min(L(:, 1)) max(L(:, 2)) - min(L(:, 2))];
        else
            box = opts.FaceBox(min(k, size(opts.FaceBox, 1)), :);
            R = boxRegions(box, H, W);
        end
        R = imgaussfilt(double(R), max(1, 0.01 * box(3)));            % feathered edges
        Tf = zeros(H, W);
        x0 = round(box(1)); y0 = round(box(2)); bw = max(2, round(box(3))); bh = max(2, round(box(4)));
        Tb = imresize(T, [bh bw], 'bilinear');
        xs = max(1, x0):min(W, x0 + bw - 1); ys = max(1, y0):min(H, y0 + bh - 1);
        Tf(ys, xs) = Tb(ys - y0 + 1, xs - x0 + 1);
        G = 1 - opts.Strength * R .* Tf;
        J = im2uint8(im2double(I) .* G);
        darkSum = darkSum + mean(1 - G(R > 0.5), 'omitnan'); nDone = nDone + 1;
    end
p = struct('generator', 'aging', 'constraint', 'C6', 'strength', opts.Strength, 'seed', opts.Seed, ...
    'kernels', 'imfilter with fspecial(laplacian,0.2) and fspecial(sobel) on a seeded crease field', ...
    'model', 'J = I .* (1 - strength * region .* texture), texture in [0,1]', ...
    'regions', 'forehead_medial, forehead_lat_L/R, malar_L/R, submalar_L/R (shared/patches.json)');
info = degradeRun(videoPath, @apply, cond, p, opts);
info.frames_without_face = noFace;
info.mean_darkening_in_region = darkSum / max(nDone, 1);
degradeAppend(info, 'measured', struct('frames_without_face', noFace, ...
    'mean_darkening_in_region', info.mean_darkening_in_region));
end

function T = agingTexture(seed)
s = RandStream('twister', 'Seed', seed);
n1 = imgaussfilt(randn(s, 256), [0.8 6]);                % horizontal creases
n2 = imgaussfilt(randn(s, 256), 0.7);                    % fine pores
T = abs(imfilter(n1, fspecial('laplacian', 0.2), 'replicate')) ...
    + 0.5 * abs(imfilter(n1, fspecial('sobel'), 'replicate')) + 0.3 * abs(n2);
T = min(T / prctile(T(:), 99), 1);
end

function R = boxRegions(box, H, W)
% forehead / cheeks as face-box fractions [x1 x2 y1 y2] (same convention as cfg.roi)
fr = [0.25 0.75 0.05 0.28; 0.12 0.40 0.45 0.72; 0.60 0.88 0.45 0.72];
R = false(H, W);
for i = 1:3
    xs = max(1, round(box(1) + fr(i, 1) * box(3))):min(W, round(box(1) + fr(i, 2) * box(3)));
    ys = max(1, round(box(2) + fr(i, 3) * box(4))):min(H, round(box(2) + fr(i, 4) * box(4)));
    R(ys, xs) = true;
end
end
