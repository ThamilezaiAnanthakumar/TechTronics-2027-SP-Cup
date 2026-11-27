function outPath = debug_video(tracePath, opts)
%DEBUG_VIDEO  Overlay video of a trace file for visual checking.
%   debug_video("outputs/traces/UBFC2_subject1_MultiFace_C3.mat")
%   debug_video(tracePath, MaxFrames=300, GifPath="outputs/figures/x.gif")
%
%   Draws all faces (target green, others red, with track IDs), the ROI
%   polygons (yellow, red when occluded), skin pixels (green tint) and a
%   status line. Options: VideoPath, OutPath, MaxFrames, GifPath, GifEvery, ShowSkin.
%   Refuses UBFC subjects 21 and 27 (no consent for image use).

arguments
    tracePath (1,1) string {mustBeFile}
    opts.VideoPath (1,1) string = ""
    opts.OutPath (1,1) string = ""
    opts.MaxFrames (1,1) double {mustBePositive} = Inf
    opts.GifPath (1,1) string = ""
    opts.GifEvery (1,1) double {mustBePositive, mustBeInteger} = 3
    opts.ShowSkin (1,1) logical = true
end

S = load(tracePath);
src = opts.VideoPath;
if src == "", src = string(S.meta.source_file); end
if contains(src, ["subject27", "subject21"]) || any(string(S.meta.subject) == ["subject27" "subject21"])
    error("debug_video:consent", "Subject without image consent: never render its face.");
end
[~, base] = fileparts(tracePath);
if opts.OutPath == ""
    opts.OutPath = fullfile("outputs", "figures", base + "_debug.avi");
end
folder = fileparts(opts.OutPath);
if folder ~= "" && ~isfolder(folder), mkdir(folder); end

p = S.meta.params;
names = string(S.roi_names);
nR = numel(names);
v = VideoReader(src);
w = VideoWriter(opts.OutPath, "Motion JPEG AVI");
w.FrameRate = v.FrameRate;
w.Quality = 90;
open(w);

fs = 12;  lw = 2;
if v.Height < 400, fs = 8; lw = 1; end
N = min(numel(S.t), opts.MaxFrames);

for k = 1:N
    if ~hasFrame(v), break; end
    I = readFrame(v);
    J = I;
    rd = S.roi_data(k);
    q  = S.quality(k);

    if opts.ShowSkin && q > 0
        Jd = double(J);
        for r = 1:nR
            P = rd.(names(r));
            if isempty(P), continue; end
            [~, ~, ~, M, rect] = roi_skin_stats(I, P, p);
            if isempty(M) || ~any(M(:)), continue; end
            yy = rect(2):rect(2) + rect(4) - 1;
            xx = rect(1):rect(1) + rect(3) - 1;
            g = Jd(yy, xx, 2);
            g(M) = min(255, 0.6 * g(M) + 0.4 * 255);
            Jd(yy, xx, 2) = g;
        end
        J = uint8(Jd);
    end

    boxes = rd.all_boxes;  ids = rd.track_ids;
    if ~isempty(boxes)
        boxes = reshape(boxes, [], 4);
        ids = ids(:);
        isT = ids == S.target_id(k);
        cols = repmat(uint8([255 60 60]), numel(ids), 1);
        cols(isT, :) = repmat(uint8([60 255 60]), nnz(isT), 1);
        J = insertShape(J, "rectangle", boxes, "ShapeColor", cols, "LineWidth", lw);
        labels = "ID " + string(ids);
        labels(isT) = labels(isT) + " target";
        pos = [boxes(:, 1), max(1, boxes(:, 2) - fs - 6)];
        J = insertText(J, pos, labels, "FontSize", fs, "BoxColor", cols, "BoxOpacity", 0.6, "TextColor", "black");
    end

    if q > 0
        for r = 1:nR
            P = rd.(names(r));
            if isempty(P), continue; end
            c = uint8([255 230 0]);
            if S.occluded(k, r), c = uint8([255 0 0]); end
            J = insertShape(J, "polygon", {reshape(P.', 1, [])}, "ShapeColor", c, "LineWidth", lw);
        end
    end

    txt = sprintf("t=%.2fs q=%d faces=%d occl=[%s] npix=[%s]", S.t(k), q, S.n_faces(k), ...
                  num2str(S.occluded(k, :)), num2str(round(S.npix(k, :))));
    J = insertText(J, [3 3], txt, "FontSize", fs, "BoxOpacity", 0.6);
    writeVideo(w, J);

    if opts.GifPath ~= "" && mod(k - 1, opts.GifEvery) == 0
        [A, map] = rgb2ind(J, 128);
        if k == 1
            imwrite(A, map, opts.GifPath, "gif", "LoopCount", Inf, "DelayTime", opts.GifEvery / v.FrameRate);
        else
            imwrite(A, map, opts.GifPath, "gif", "WriteMode", "append", "DelayTime", opts.GifEvery / v.FrameRate);
        end
    end
end
close(w);
outPath = opts.OutPath;
fprintf("[debug] %s (%d frames)\n", outPath, k);
end
