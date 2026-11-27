function out = process_video(videoPath, opts)
%PROCESS_VIDEO  Face / ROI module: video -> per-frame ROI RGB traces of the target face.
%   out = process_video(videoPath)
%   out = process_video(videoPath, Name=Value)
%
%   Options: Params (face_params struct), Config (JSON path, used when Params
%   is not given), Detector (faceDetector to reuse), MaxFrames, Dataset,
%   Subject, Condition, Constraint, Verbose.
%
%   Output (R = number of regions in p.roi.names, N = frames; no frame is dropped):
%     t            Nx1 frame time (s), read from the video
%     rgb_<region> Nx3 mean RGB of skin pixels, one variable per region
%     rgb_face     Nx3 pixel-weighted mean over non-occluded regions
%     roi_names    1xR region names (column order of npix / skin_ratio / occluded)
%     npix, skin_ratio, occluded   NxR
%     quality      Nx1 0 lost (RGB = NaN), 1 tracked, 2 confirmed by detector
%     n_faces, target_id           Nx1
%     roi_data     Nx1 struct: face_box, one 4x2 polygon per region, all_boxes, track_ids
%     meta, debug  structs

arguments
    videoPath (1,1) string {mustBeFile}
    opts.Params struct = struct()
    opts.Config (1,1) string = fullfile(repo_root(), "config", "face_config.json")
    opts.Detector = []
    opts.MaxFrames (1,1) double {mustBePositive} = Inf
    opts.Dataset (1,1) string = "UBFC2"
    opts.Subject (1,1) string = ""
    opts.Condition (1,1) string = "Baseline"
    opts.Constraint (1,1) string = "None"
    opts.Verbose (1,1) logical = true
end

p = opts.Params;
if isempty(fieldnames(p)), p = face_params(opts.Config); end
fd = opts.Detector;
if isempty(fd), fd = faceDetector(p.det.model); end

names = p.roi.names;
nR = numel(names);
refIdx = find(names == p.occl.reference, 1);
if isempty(refIdx), refIdx = 1; useRel = false; else, useRel = true; end

v = VideoReader(videoPath);
H = v.Height;  W = v.Width;
n = min(opts.MaxFrames, max(1, v.NumFrames));

t        = zeros(n, 1);
rgbReg   = NaN(n, 3, nR);
rgbFace  = NaN(n, 3);
npix     = zeros(n, nR);
ratio    = NaN(n, nR);
occl     = zeros(n, nR);
quality  = zeros(n, 1);
nFaces   = zeros(n, 1);
targetId = NaN(n, 1);
faceBox  = NaN(n, 4);
polys    = NaN(n, 8, nR);
allBoxes = cell(n, 1);
allIds   = cell(n, 1);
refRaw   = NaN(n, 2);
numPts   = zeros(n, 1);
detFlag  = false(n, 1);

tracker  = FaceTracker(fd, p, [H W]);
smoother = OneEuroFilter(p.smooth.minCutoff, p.smooth.beta, p.smooth.dCutoff);
ratioEma = NaN(1, nR);
absState = false(1, nR);
relState = false(1, nR);
tm = struct("read", 0, "track", 0, "roi", 0);
tStart = tic;

k = 0;
while hasFrame(v) && k < opts.MaxFrames
    k = k + 1;
    t0 = tic;
    t(k) = v.CurrentTime;                        % read before readFrame
    I = readFrame(v);
    tm.read = tm.read + toc(t0);

    t0 = tic;
    st = tracker.step(I, t(k));
    tm.track = tm.track + toc(t0);

    quality(k)  = st.quality;
    nFaces(k)   = st.nFaces;
    targetId(k) = st.targetId;
    allBoxes{k} = st.boxes;
    allIds{k}   = st.ids;
    numPts(k)   = st.numPts;
    detFlag(k)  = st.detected;

    if st.quality == 0
        smoother.reset();
        ratioEma(:) = NaN;  absState(:) = false;  relState(:) = false;
        continue
    end

    t0 = tic;
    R = roi_polygons(st.anchorBox, st.A, p);
    refRaw(k, :) = mean(R.(names(refIdx)), 1);
    vec = zeros(1, 8 * nR + 4);
    for r = 1:nR
        vec(8*r-7 : 8*r) = R.(names(r))(:).';
    end
    vec(end-3:end) = R.box;
    if p.smooth.enable
        vec = smoother.update(vec, t(k));
    end
    faceBox(k, :) = vec(end-3:end);

    sums = zeros(nR, 3);  nS = zeros(1, nR);
    for r = 1:nR
        P = reshape(vec(8*r-7 : 8*r), 4, 2);
        polys(k, :, r) = P(:).';
        [sums(r, :), nS(r), nP] = roi_skin_stats(I, P, p);
        rr = nS(r) / max(nP, 1);
        ratio(k, r) = rr;
        if isnan(ratioEma(r))
            ratioEma(r) = rr;
        else
            ratioEma(r) = (1 - p.occl.emaAlpha) * ratioEma(r) + p.occl.emaAlpha * rr;
        end
    end

    % occlusion: absolute skin ratio AND (for non-reference regions) ratio vs reference region
    sumAll = zeros(1, 3);  nAll = 0;
    for r = 1:nR
        if absState(r)
            absState(r) = ratioEma(r) < p.occl.offAbove;
        else
            absState(r) = ratioEma(r) < p.occl.onBelow;
        end
        if ~useRel || r == refIdx
            relState(r) = true;
        else
            rel = ratioEma(r) / max(ratioEma(refIdx), eps);
            if relState(r)
                relState(r) = rel < p.occl.relOff;
            else
                relState(r) = rel < p.occl.relOn;
            end
        end
        isOcc = (absState(r) && relState(r)) || nS(r) < p.occl.minPix;

        npix(k, r) = nS(r);
        occl(k, r) = isOcc;
        if nS(r) > 0, rgbReg(k, :, r) = sums(r, :) / nS(r); end
        if ~isOcc
            sumAll = sumAll + sums(r, :);
            nAll   = nAll + nS(r);
        end
    end
    if nAll > 0, rgbFace(k, :) = sumAll / nAll; end
    tm.roi = tm.roi + toc(t0);

    if opts.Verbose && mod(k, 300) == 0
        fprintf("   frame %5d  t=%6.2fs  q=%d  faces=%d  occl=[%s]  %.1f ms/frame\n", ...
            k, t(k), st.quality, st.nFaces, num2str(occl(k, :)), 1000 * toc(tStart) / k);
    end
end
wall = toc(tStart);

t = t(1:k);  rgbReg = rgbReg(1:k, :, :);  rgbFace = rgbFace(1:k, :);
npix = npix(1:k, :);  ratio = ratio(1:k, :);  occl = occl(1:k, :);
quality = quality(1:k);  nFaces = nFaces(1:k);  targetId = targetId(1:k);
faceBox = faceBox(1:k, :);  polys = polys(1:k, :, :);
allBoxes = allBoxes(1:k);  allIds = allIds(1:k);
refRaw = refRaw(1:k, :);  numPts = numPts(1:k);  detFlag = detFlag(1:k);

rd = struct("face_box", NaN(1, 4));
for r = 1:nR, rd.(names(r)) = []; end
rd.all_boxes = zeros(0, 4);
rd.track_ids = zeros(0, 1);
roi_data = repmat(rd, k, 1);
for i = 1:k
    roi_data(i).face_box  = faceBox(i, :);
    roi_data(i).all_boxes = allBoxes{i};
    roi_data(i).track_ids = allIds{i};
    if quality(i) > 0
        for r = 1:nR
            roi_data(i).(names(r)) = reshape(polys(i, :, r), 4, 2);
        end
    end
end

out.t = t;
for r = 1:nR
    out.("rgb_" + names(r)) = rgbReg(:, :, r);
end
out.rgb_face   = rgbFace;
out.roi_names  = names;
out.npix       = npix;
out.skin_ratio = ratio;
out.quality    = quality;
out.occluded   = occl;
out.n_faces    = nFaces;
out.target_id  = targetId;
out.roi_data   = roi_data;

info = dir(videoPath);
meta.source_file    = videoPath;
meta.dataset        = opts.Dataset;
meta.subject        = opts.Subject;
meta.condition      = opts.Condition;
meta.constraint     = opts.Constraint;
meta.width          = W;
meta.height         = H;
meta.fps_nominal    = v.FrameRate;
meta.fps_measured   = 1 / median(diff(t));
meta.bitrate_kbps   = info.bytes * 8 / 1000 / max(v.Duration, eps);
meta.n_frames       = k;
meta.module_version = git_version();
meta.created        = string(datetime("now", "Format", "yyyy-MM-dd HH:mm:ss"));
meta.timing = struct( ...
    "ms_per_frame",     1000 * wall / k, ...
    "ms_read",          1000 * tm.read / k, ...
    "ms_track",         1000 * tm.track / k, ...
    "ms_detect",        1000 * tracker.DetectTime / k, ...
    "ms_roi_skin",      1000 * tm.roi / k, ...
    "n_detector_calls", tracker.NumDetections);
meta.params = p;
out.meta = meta;

out.debug.ref_center_raw = refRaw;          % reference-region centre before smoothing
out.debug.num_points     = numPts;
out.debug.detector_ran   = detFlag;

if opts.Verbose
    fprintf("   done: %d frames, %.1f s (%.1f ms/frame), target found in %.1f %% of frames\n", ...
        k, wall, meta.timing.ms_per_frame, 100 * mean(quality > 0));
end
end

function s = git_version()
s = "unknown";
try
    [status, txt] = system("git -C """ + repo_root() + """ rev-parse --short HEAD");
    if status == 0, s = strtrim(string(txt)); end
catch
end
end
