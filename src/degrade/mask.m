function info = mask(videoPath, opts)
%MASK C7 face-mask occlusion (README 1.4 / 5): insertShape filled polygon, landmark-aligned.
%   info = mask(videoPath)                                  surgical (light blue)
%   info = mask(v, Variant="cloth", MaxFrames=300)          cloth (dark)
%   info = mask(v, FaceBox=[x y w h])                       fixed box instead of landmarks (tests)
%   The polygon runs from below the nose (subnasale, MediaPipe landmark 2; TopFrac moves the top
%   edge toward the chin) to the chin along the face-oval contour, and is drawn with
%   insertShape('filled-polygon', Opacity=1). Per frame the covered share of the lower face
%   (oval below the nose tip) is measured; mean / min are written to the sidecar (spec >= 40 %).
%   Frames before the first face detection are left unchanged (counted in frames_without_face).
%   Output: <OutRoot>/Masked_C7_<variant>/<subject>.mj2 (lossless) + .json sidecar.
arguments
    videoPath (1,1) string {mustBeFile}
    opts.Variant (1,1) string {mustBeMember(opts.Variant, ["surgical" "cloth"])} = "surgical"
    opts.Color (1,3) double = NaN(1, 3)
    opts.TopFrac (1,1) double {mustBeInRange(opts.TopFrac, 0, 0.6)} = 0
    opts.FaceBox double = []
    opts.Condition (1,1) string = ""
    opts.OutRoot (1,1) string = ""
    opts.Subject (1,1) string = ""
    opts.MaxFrames (1,1) double = Inf
    opts.Profile (1,1) string = "Archival"
end
if opts.OutRoot == "", opts.OutRoot = degradeDefaultRoot(); end
col = opts.Color;
if any(isnan(col))
    if opts.Variant == "surgical", col = [150 200 230]; else, col = [45 45 55]; end
end
cond = opts.Condition;
if cond == "", cond = "Masked_C7_" + opts.Variant; end
st = struct('lmk', [], 'smoother', [], 'prevC', [], 'prevW', [], 'last', []);
cover = []; noFace = 0;
    function J = apply(I, k, t)
        if isempty(opts.FaceBox)
            [L, st] = faceGeomStep(st, I, t); box = [];
            if isempty(L), J = I; noFace = noFace + 1; cover(k) = NaN; return; end
        else
            L = []; box = opts.FaceBox(min(k, size(opts.FaceBox, 1)), :);
        end
        [mp, lp] = maskGeometry(L, box, opts.TopFrac);
        J = insertShape(I, 'filled-polygon', reshape(mp', 1, []), 'Color', col, 'Opacity', 1);
        [H, W, ~] = size(I);
        lo = poly2mask(lp(:, 1), lp(:, 2), H, W);
        cover(k) = nnz(lo & poly2mask(mp(:, 1), mp(:, 2), H, W)) / max(nnz(lo), 1);
    end
if isempty(opts.FaceBox)
    geom = 'MediaPipe landmarks: subnasale (2) to chin along the face oval';
else
    geom = 'fixed face box (box fractions)';
end
p = struct('generator', 'mask', 'constraint', 'C7', 'variant', char(opts.Variant), ...
    'color_rgb', col, 'top_frac', opts.TopFrac, 'opacity', 1, 'geometry', geom, ...
    'drawing', 'insertShape filled-polygon');
info = degradeRun(videoPath, @apply, cond, p, opts);
info.coverage_mean = mean(cover, 'omitnan'); info.coverage_min = min(cover);
info.frames_without_face = noFace;
degradeAppend(info, 'coverage_lower_face', struct('mean', info.coverage_mean, 'min', info.coverage_min, ...
    'frames_without_face', noFace, ...
    'definition', 'area(mask & lower face) / area(lower face); lower face = face oval below the nose tip'));
end
