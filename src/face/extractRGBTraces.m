function traces = extractRGBTraces(videoPath, cfg)
%EXTRACTRGBTRACES Face detect/track, ROIs and spatial-mean RGB per frame.
%   traces = extractRGBTraces(videoPath, cfg)
%   The only step that touches the raw video. Reads frame by frame.
%   Face: Viola-Jones detection on frame 1 and every cfg.detectEvery frames,
%   KLT point tracking (similarity transform) in between. Largest face kept.
%   If both fail, the last valid box is reused and faceOK is false.
%
%   Output:
%     traces.fps, traces.t (N x 1, s), traces.rgb.(roi) (N x 3, mean RGB),
%     traces.box (N x 4, [x y w h]), traces.faceOK (N x 1 logical),
%     traces.info (resolution, numFrames, durations, path)
%   Cached in <resultsRoot>/<subject>/traces.mat; the cache is reused only
%   if the extraction settings match (traces.key).

outDir    = fullfile(cfg.resultsRoot, cfg.subject);
cacheFile = fullfile(outDir, 'traces.mat');
key       = extractionKey(cfg);

if isfile(cacheFile)
    S = load(cacheFile, 'traces');
    if isfield(S.traces, 'key') && isequal(S.traces.key, key)
        fprintf('Loaded cached traces: %s\n', cacheFile);
        traces = S.traces;
        return
    end
    fprintf('Cached traces were made with different settings; re-extracting.\n');
end

% Fixed RNG per video: estgeotform2d (MSAC) samples points at random, so an
% unseeded run is not repeatable (docs/CHANGELOG_PARAMS.md, 26 Nov 2026).
rngSaved = rng;
rng(0, 'twister');
rngRestore = onCleanup(@() rng(rngSaved));
v   = VideoReader(videoPath);
fps = v.FrameRate;
nMax = min(cfg.maxFrames, ceil(v.Duration * fps) + 10);
baseRois = fieldnames(cfg.roi);

sums   = struct();
counts = struct();
for r = 1:numel(baseRois)
    sums.(baseRois{r})   = nan(nMax, 3);
    counts.(baseRois{r}) = zeros(nMax, 1);
end
boxAll = nan(nMax, 4);
faceOK = false(nMax, 1);

detector = [];
tracker  = vision.PointTracker('MaxBidirectionalError', 2);
trackerReady = false;
corners = [];          % 4 x 2 face-box corners, tracked between detections
oldPts  = [];

fprintf('Extracting RGB traces from %s\n', videoPath);
tic;
k = 0;
while hasFrame(v) && k < cfg.maxFrames
    k = k + 1;
    frame = readFrame(v);
    frame = degradeFrame(frame, cfg.degrade);   % Section 7 hook (no-op by default)
    gray  = im2gray(frame);
    imSize = size(gray);
    if isempty(detector)
        minFace = round(0.15 * min(imSize)) * [1 1];
        detector = vision.CascadeObjectDetector('MinSize', minFace);
    end

    ok = false;

    % --- KLT tracking from the previous frame ---
    if trackerReady
        [pts, valid] = tracker(gray);
        if nnz(valid) >= cfg.minTrackPts
            [tform, inl, status] = estgeotform2d(oldPts(valid,:), pts(valid,:), ...
                'similarity', 'MaxDistance', 4);
            if status == 0 && nnz(inl) >= cfg.minTrackPts
                corners = transformPointsForward(tform, corners);
                p = pts(valid, :);
                oldPts = p(inl, :);
                setPoints(tracker, oldPts);
                ok = true;
            end
        end
        trackerReady = ok;
    end

    % --- Periodic (or recovery) detection ---
    if mod(k-1, cfg.detectEvery) == 0 || ~trackerReady
        det = detectLargestFace(detector, gray);
        if ~isempty(det)
            corners = boxToCorners(det);
            pts = detectMinEigenFeatures(gray, 'ROI', det);
            oldPts = pts.Location;
            if size(oldPts, 1) >= cfg.minTrackPts
                release(tracker);
                initialize(tracker, oldPts, gray);
                trackerReady = true;
            else
                trackerReady = false;
            end
            ok = true;
        end
    end

    faceOK(k) = ok;
    if isempty(corners)
        continue                     % no face seen yet: leave NaN
    end
    box = cornersToBox(corners, imSize);
    boxAll(k, :) = box;

    rb = roiBoxes(box, cfg, imSize);
    for r = 1:numel(baseRois)
        [s, n] = roiSumRGB(frame, rb.(baseRois{r}), cfg);
        sums.(baseRois{r})(k, :) = s;
        counts.(baseRois{r})(k)  = n;
    end

    if mod(k, cfg.progressEvery) == 0
        fprintf('  frame %5d  (%.0f s elapsed, face OK so far %.1f%%)\n', ...
            k, toc, 100 * mean(faceOK(1:k)));
    end
end
N = k;
fprintf('  done: %d frames in %.0f s\n', N, toc);

% --- Assemble per-ROI mean RGB, pooled 'combined' ROI ---
traces.fps = fps;
traces.t   = (0:N-1)' / fps;
totS = zeros(N, 3); totN = zeros(N, 1);
for r = 1:numel(baseRois)
    s = sums.(baseRois{r})(1:N, :);
    n = counts.(baseRois{r})(1:N);
    traces.rgb.(baseRois{r}) = fillGaps(s ./ n);
    totS = totS + fillmissing(s, 'constant', 0);
    totN = totN + n;
end
traces.rgb.combined = fillGaps(totS ./ totN);

traces.box    = boxAll(1:N, :);
traces.faceOK = faceOK(1:N);
traces.info = struct('width', v.Width, 'height', v.Height, ...
    'frameSize', imSize, 'numFrames', N, 'duration', N / fps, ...
    'videoDuration', v.Duration, 'videoPath', videoPath, ...
    'faceOKFraction', mean(faceOK(1:N)));
traces.key = key;

if ~isfolder(outDir), mkdir(outDir); end
save(cacheFile, 'traces');
fprintf('Saved traces: %s\n', cacheFile);
end

% ------------------------------------------------------------------------
function box = detectLargestFace(detector, gray)
bboxes = detector(gray);
if isempty(bboxes)
    box = [];
    return
end
[~, i] = max(bboxes(:,3) .* bboxes(:,4));
box = double(bboxes(i, :));
end

function c = boxToCorners(b)
c = [b(1) b(2); b(1)+b(3) b(2); b(1)+b(3) b(2)+b(4); b(1) b(2)+b(4)];
end

function box = cornersToBox(c, imSize)
% Axis-aligned box of the tracked quadrilateral, clipped to the image.
x1 = max(min(c(:,1)), 1); x2 = min(max(c(:,1)), imSize(2));
y1 = max(min(c(:,2)), 1); y2 = min(max(c(:,2)), imSize(1));
box = [x1 y1 x2-x1 y2-y1];
end

function x = fillGaps(x)
% Fill NaN frames (no face yet / empty ROI) from neighbouring frames.
if any(isnan(x(:)))
    x = fillmissing(x, 'linear', 1, 'EndValues', 'nearest');
end
end

function key = extractionKey(cfg)
% Settings that change the extracted traces (cache validity).
key = struct('subject', cfg.subject, 'maxFrames', cfg.maxFrames, ...
    'detectEvery', cfg.detectEvery, 'minTrackPts', cfg.minTrackPts, ...
    'roi', cfg.roi, 'useSkinMask', cfg.useSkinMask, 'skinCb', cfg.skinCb, ...
    'skinCr', cfg.skinCr, 'minSkinPixels', cfg.minSkinPixels, ...
    'degrade', cfg.degrade, 'rngSeed', 0);   % rngSeed: seeded since 26 Nov 2026
end
