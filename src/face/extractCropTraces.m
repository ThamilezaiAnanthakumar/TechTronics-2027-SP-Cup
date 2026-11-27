function traces = extractCropTraces(roiPaths, cfg, cacheFile)
%EXTRACTCROPTRACES Mean-RGB traces from videos that are already ROI crops.
%   traces = extractCropTraces(roiPaths, cfg, cacheFile)
%   roiPaths  : struct, roiPaths.(roi) = path to that ROI's video (e.g. AiPEX
%               videoN_forehead.avi). No face detection: the whole frame is
%               the ROI, with the same optional skin mask as extractRGBTraces.
%   cacheFile : .mat path; reused if it exists with matching settings.
%   Output has the same fields as extractRGBTraces (fps, t, rgb.(roi) incl.
%   'combined', faceOK all true, info). Videos are truncated to the
%   shortest ROI length so all ROIs share one time base.

rois = fieldnames(roiPaths);
key = struct('paths', roiPaths, 'maxFrames', cfg.maxFrames, ...
    'useSkinMask', cfg.useSkinMask, 'skinCb', cfg.skinCb, 'skinCr', cfg.skinCr, ...
    'minSkinPixels', cfg.minSkinPixels, 'degrade', cfg.degrade);
if nargin >= 3 && isfile(cacheFile)
    S = load(cacheFile, 'traces');
    if isfield(S.traces, 'key') && isequal(S.traces.key, key)
        traces = S.traces;
        return
    end
end

sums = struct(); counts = struct(); nFr = zeros(numel(rois), 1);
for r = 1:numel(rois)
    v = VideoReader(roiPaths.(rois{r}));
    fps = v.FrameRate;
    nMax = min(cfg.maxFrames, ceil(v.Duration * fps) + 10);
    s = nan(nMax, 3); n = zeros(nMax, 1);
    k = 0;
    while hasFrame(v) && k < cfg.maxFrames
        k = k + 1;
        frame = degradeFrame(readFrame(v), cfg.degrade);
        box = [1 size(frame, 1) 1 size(frame, 2)];
        [s(k, :), n(k)] = roiSumRGB(frame, box, cfg);
    end
    sums.(rois{r}) = s(1:k, :); counts.(rois{r}) = n(1:k); nFr(r) = k;
end

N = min(nFr);
traces.fps = fps;
traces.t = (0:N-1)' / fps;
totS = zeros(N, 3); totN = zeros(N, 1);
for r = 1:numel(rois)
    s = sums.(rois{r})(1:N, :); n = counts.(rois{r})(1:N);
    traces.rgb.(rois{r}) = s ./ n;
    totS = totS + s; totN = totN + n;
end
traces.rgb.combined = totS ./ totN;
traces.faceOK = true(N, 1);
traces.info = struct('numFrames', N, 'duration', N / fps, 'framesPerRoi', nFr');
traces.key = key;

if nargin >= 3
    d = fileparts(cacheFile);
    if ~isfolder(d), mkdir(d); end
    save(cacheFile, 'traces');
end
end
