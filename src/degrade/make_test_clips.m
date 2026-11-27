function make_test_clips(subjects, opts)
%MAKE_TEST_CLIPS  Generate C3 / C4 / C7 test clips from UBFC DATASET_2 videos.
%   make_test_clips([1 3])
%   make_test_clips([1 3], Conditions=["MultiFace" "Masked"], MaxFrames=300)
%
%   Writes <OutDir>/<Condition>/subjectN.avi (uncompressed) and subjectN_meta.mat
%   (reference target box per frame, bystander box, source and GT paths).
%     LowResolutionQVGA  bicubic resize to 320x240                     (C4)
%     MultiFace          next subject composited beside the target    (C3)
%     Masked             synthetic surgical mask on the target        (C7)
%     Stress             MultiFace + Masked + QVGA

arguments
    subjects (1,:) double {mustBeInteger, mustBePositive}
    opts.Conditions (1,:) string = ["LowResolutionQVGA" "MultiFace" "Masked" "Stress"]
    opts.RawDir (1,1) string = "data/raw/UBFC/DATASET_2"
    opts.OutDir (1,1) string = "data/degraded"
    opts.MaxFrames (1,1) double {mustBePositive} = Inf
    opts.Params struct = face_params()
    opts.Detector = []
    opts.Overwrite (1,1) logical = false
end

p  = opts.Params;
fd = opts.Detector;
if isempty(fd), fd = faceDetector(p.det.model); end
allC = clip_conditions();

for i = 1:numel(subjects)
    s    = subjects(i);
    subj = "subject" + s;
    src  = fullfile(opts.RawDir, subj, "vid.avi");
    gtp  = fullfile(opts.RawDir, subj, "ground_truth.txt");
    if ~isfile(src)
        warning("make_test_clips:missing", "%s not found, skipped.", src);
        continue
    end
    fprintf("[clips] %s\n", subj);
    boxes = scan_target_boxes(src, fd, p, MaxFrames=opts.MaxFrames, ...
                              CachePath=fullfile(opts.OutDir, "_cache", "UBFC2_" + subj + "_boxes.mat"));
    nF = size(boxes, 1);
    vt = VideoReader(src);
    H = vt.Height;  W = vt.Width;

    write_sidecar(opts.OutDir, allC([allC.name] == "Baseline"), subj, src, gtp, boxes, [1 1], [], "", nF);

    todo = allC(ismember([allC.name], opts.Conditions));
    keep = true(size(todo));
    for c = 1:numel(todo)
        outVid = fullfile(opts.OutDir, todo(c).name, subj + ".avi");
        side   = fullfile(opts.OutDir, todo(c).name, subj + "_meta.mat");
        if ~isempty(todo(c).resize) && all([H W] <= todo(c).resize)
            keep(c) = false;                       % source already at or below this size
        elseif isfile(outVid) && isfile(side) && ~opts.Overwrite
            keep(c) = false;
        end
    end
    todo = todo(keep);
    if isempty(todo), continue; end

    needBy = any([todo.multiface]);
    by = [];  vb = [];  bySubj = "";
    if needBy
        b = subjects(mod(i, numel(subjects)) + 1);
        flipBy = (b == s);                         % single subject: mirrored copy as bystander
        if flipBy
            vb = VideoReader(src);  bySubj = subj + "_mirrored";
        else
            vb = VideoReader(fullfile(opts.RawDir, "subject" + b, "vid.avi"));  bySubj = "subject" + b;
        end
        B1 = readFrame(vb);
        if flipBy, B1 = fliplr(B1); end
        bb = detect_faces(fd, B1, p);
        if isempty(bb), error("make_test_clips:noBystander", "No face in bystander %s.", bySubj); end
        [~, j] = max(bb(:, 3) .* bb(:, 4));
        by = setup_bystander(bb(j, :), boxes(1, :), [H W], size(B1), p);
        by.flip = flipBy;
        vb.CurrentTime = 0;
    end

    writers = cell(numel(todo), 1);
    for c = 1:numel(todo)
        folder = fullfile(opts.OutDir, todo(c).name);
        if ~isfolder(folder), mkdir(folder); end
        writers{c} = VideoWriter(fullfile(folder, subj + ".avi"), "Uncompressed AVI");
        writers{c}.FrameRate = vt.FrameRate;
        open(writers{c});
    end

    k = 0;  Bf = [];
    while hasFrame(vt) && k < nF
        k = k + 1;
        I = readFrame(vt);
        if needBy
            if by.flip
                Bf = fliplr(I);
            elseif hasFrame(vb)
                Bf = readFrame(vb);
            end
            Iby = paste_bystander(I, Bf, by);
        end
        for c = 1:numel(todo)
            J = I;
            if todo(c).multiface, J = Iby; end
            if todo(c).mask, J = draw_mask(J, boxes(k, :), p); end
            if ~isempty(todo(c).resize), J = imresize(J, todo(c).resize, "bicubic"); end
            writeVideo(writers{c}, J);
        end
    end
    for c = 1:numel(todo), close(writers{c}); end

    for c = 1:numel(todo)
        sxy = [1 1];
        if ~isempty(todo(c).resize), sxy = todo(c).resize([2 1]) ./ [W H]; end
        byBox = [];
        if todo(c).multiface, byBox = scale_boxes(by.faceBox, sxy); end
        write_sidecar(opts.OutDir, todo(c), subj, fullfile(opts.OutDir, todo(c).name, subj + ".avi"), ...
                      gtp, scale_boxes(boxes(1:k, :), sxy), sxy, byBox, bySubj, k);
        fprintf("   %s/%s.avi (%d frames)\n", todo(c).name, subj, k);
    end
end
end

function by = setup_bystander(fb, tgtBox, frameSize, srcSize, p)
H = frameSize(1);  W = frameSize(2);
c  = [fb(1) + fb(3)/2, fb(2) + fb(4)/2];
cw = p.clip.bystanderCrop * fb(3);
ch = p.clip.bystanderCrop * fb(4);
crop = clip_box([c(1) - cw/2, c(2) - ch/2, cw, ch], srcSize, true);
by.cropRows = crop(2):crop(2) + crop(4) - 1;
by.cropCols = crop(1):crop(1) + crop(3) - 1;

pw = min(round(crop(3) * p.clip.bystanderScale), W - 2);
ph = min(round(crop(4) * p.clip.bystanderScale), H - 2);
margin = round(0.04 * W);
if tgtBox(1) + tgtBox(3)/2 > W/2
    x0 = margin + 1;                               % target right of centre -> bystander left
else
    x0 = W - margin - pw + 1;
end
y0 = round(tgtBox(2) + tgtBox(4)/2 - ph/2);
x0 = min(max(1, x0), W - pw + 1);
y0 = min(max(1, y0), H - ph + 1);
by.rows = y0:y0 + ph - 1;
by.cols = x0:x0 + pw - 1;
by.ph = ph;  by.pw = pw;

[X, Y] = meshgrid(1:pw, 1:ph);
e = ((X - (pw + 1)/2) / (0.48 * pw)).^2 + ((Y - (ph + 1)/2) / (0.48 * ph)).^2 <= 1;
by.alpha = imgaussfilt(double(e), max(1, 0.03 * pw));
by.faceBox = [(fb(1) - crop(1)) * pw / crop(3) + x0, (fb(2) - crop(2)) * ph / crop(4) + y0, ...
              fb(3) * pw / crop(3), fb(4) * ph / crop(4)];
if box_iou([x0 y0 pw ph], tgtBox) > 0
    warning("make_test_clips:overlap", "Bystander patch overlaps the target face.");
end
end

function J = paste_bystander(I, Bf, by)
patch = imresize(Bf(by.cropRows, by.cropCols, :), [by.ph by.pw], "bicubic");
J = I;
J(by.rows, by.cols, :) = uint8(by.alpha .* double(patch) + (1 - by.alpha) .* double(I(by.rows, by.cols, :)));
end

function b = scale_boxes(b, sxy)
b = [(b(:,1) - 0.5) * sxy(1) + 0.5, (b(:,2) - 0.5) * sxy(2) + 0.5, b(:,3) * sxy(1), b(:,4) * sxy(2)];
end

function write_sidecar(outDir, cond, subj, videoPath, gtPath, targetBox, sxy, byBox, bySubj, n)
clip.video_path        = videoPath;
clip.gt_path           = gtPath;
clip.dataset           = "UBFC2";
clip.subject           = subj;
clip.condition         = cond.name;
clip.constraint        = cond.constraint;
clip.scale             = sxy;
clip.target_box        = targetBox;
clip.bystander_box     = byBox;
clip.bystander_subject = bySubj;
clip.expected_faces    = cond.expected_faces;
clip.n_frames          = n;
clip.created           = string(datetime("now", "Format", "yyyy-MM-dd HH:mm:ss"));
folder = fullfile(outDir, cond.name);
if ~isfolder(folder), mkdir(folder); end
clip = to_char_struct(clip);
save(fullfile(folder, subj + "_meta.mat"), "clip", "-v7");
end
