function boxes = scan_target_boxes(videoPath, fd, p, opts)
%SCAN_TARGET_BOXES  Reference face box per frame of a one-person video (cached).
%   boxes = scan_target_boxes(videoPath, fd, p, MaxFrames=N, CachePath="x.mat")
%   Largest RetinaFace detection every p.clip.scanEvery frames, gaps
%   interpolated, smoothed. boxes: Nx4 [x y w h].

arguments
    videoPath (1,1) string {mustBeFile}
    fd
    p struct
    opts.MaxFrames (1,1) double = Inf
    opts.CachePath (1,1) string = ""
end

if opts.CachePath ~= "" && isfile(opts.CachePath)
    c = load(opts.CachePath, "boxes", "complete");
    if c.complete || size(c.boxes, 1) >= opts.MaxFrames
        boxes = c.boxes(1:min(end, opts.MaxFrames), :);
        return
    end
end

v = VideoReader(videoPath);
boxes = NaN(min(opts.MaxFrames, max(1, v.NumFrames)), 4);
k = 0;
while hasFrame(v) && k < opts.MaxFrames
    k = k + 1;
    I = readFrame(v);
    if mod(k - 1, p.clip.scanEvery) == 0
        b = detect_faces(fd, I, p);
        if ~isempty(b)
            [~, j] = max(b(:, 3) .* b(:, 4));
            boxes(k, :) = b(j, :);
        end
    end
end
complete = ~hasFrame(v);
boxes = boxes(1:k, :);
if all(isnan(boxes(:, 1)))
    error("scan_target_boxes:noFace", "No face found in %s.", videoPath);
end
boxes = fillmissing(boxes, "linear", 1, "EndValues", "nearest");
boxes = smoothdata(boxes, 1, "movmean", 5);

if opts.CachePath ~= ""
    folder = fileparts(opts.CachePath);
    if folder ~= "" && ~isfolder(folder), mkdir(folder); end
    save(opts.CachePath, "boxes", "complete", "-v7");
end
end
