function L = make_lowlight_clips(opts)
%MAKE_LOWLIGHT_CLIPS  C1 low-light versions of UBFC videos, random 10-45 lux per video.
%   L = make_lowlight_clips()
%   L = make_lowlight_clips(Subjects=["subject1" "subject3"], LuxRange=[10 45], MaxFrames=300)
%
%   Constraint C1 (illuminance <= 50 lux), brightness attenuation only, as in
%   the SP Cup 2027 document ("scale pixel intensities of a well-lit baseline
%   video using im2double followed by a multiplicative gain factor to emulate
%   the target lux level"). No noise or other illumination effects are added.
%
%   Each subject gets its own random target illuminance, uniform in LuxRange,
%   drawn with seed Seed + subject number, so it is reproducible and does not
%   change when other subjects are added or skipped.
%   Gain = target_lux / OriginalLux. Light adds linearly but video pixels are
%   gamma-encoded (sRGB), so every frame is converted to linear light
%   (rgb2lin), multiplied by the gain, and converted back (lin2rgb).
%   OriginalLux: UBFC does not report its illuminance; 300 lux is the SP Cup
%   baseline for normal indoor lighting (Task 1, >= 300 lux).
%
%   Input   SrcRoot/subjectN/vid.avi and ground_truth.txt. Subjects whose
%           folder is missing are extracted from ZipDir/subjectN-*.zip first.
%           Exclude (default 11 18 20 24 27, as in the face_detect README)
%           are skipped; pass Exclude=string.empty to keep them.
%   Output  OutRoot/subjectN/vid.avi   (lossless; frame rate and frame count
%                                       as the source, MaxFrames aside)
%           OutRoot/subjectN/ground_truth.txt   (copied unchanged)
%           OutRoot/lowlight_log.csv   (subject, target lux, gain, brightness)
%   L       table of the subjects processed in this call

arguments
    opts.SrcRoot (1,1) string = fullfile(repo_root(), "Data", "raw", "UBFC", "DATASET_2")
    opts.ZipDir (1,1) string = fullfile(repo_root(), "Data", "UBFC _DAT_2")
    opts.OutRoot (1,1) string = fullfile(repo_root(), "Data", "degraded", "LowLight")
    opts.Subjects string = string.empty     % default: every subject found (folders + zips)
    % team exclusions (face_detect README): 27 no image consent; 11 18 20 24 PPG dropouts
    opts.Exclude string = ["subject11" "subject18" "subject20" "subject24" "subject27"]
    opts.LuxRange (1,2) double {mustBePositive} = [10 45]
    opts.OriginalLux (1,1) double {mustBePositive} = 300
    opts.Seed (1,1) double = 2027
    opts.MaxFrames (1,1) double {mustBePositive} = Inf
    opts.Profile (1,1) string = "Uncompressed AVI"
    opts.Overwrite (1,1) logical = false
end

assert(opts.LuxRange(2) > opts.LuxRange(1) && opts.LuxRange(2) < opts.OriginalLux, ...
    "LuxRange must be increasing and below OriginalLux.");
if ~isfolder(opts.SrcRoot), mkdir(opts.SrcRoot); end
if ~isfolder(opts.OutRoot), mkdir(opts.OutRoot); end

% subjects: extracted folders + zips not yet extracted
subs = string({dir(fullfile(opts.SrcRoot, "subject*")).name});
zips = dir(fullfile(opts.ZipDir, "subject*.zip"));
zipSub = strings(1, numel(zips));
for z = 1:numel(zips)
    zipSub(z) = string(regexp(zips(z).name, "^subject\d+", "match", "once"));
end
subs = unique([subs zipSub(zipSub ~= "")]);
if ~isempty(opts.Subjects), subs = intersect(subs, opts.Subjects); end
subs = setdiff(subs, opts.Exclude);
[~, order] = sort(str2double(extractAfter(subs, "subject")));
subs = subs(order);
if isempty(subs), error("make_lowlight_clips:none", "No UBFC subjects found."); end

rows = {};
for s = subs
    src = fullfile(opts.SrcRoot, s, "vid.avi");
    gt  = fullfile(opts.SrcRoot, s, "ground_truth.txt");
    if ~isfile(src)                                     % extract from its zip
        z = find(zipSub == s, 1);
        if isempty(z)
            warning("make_lowlight_clips:missing", "%s: no video and no zip, skipped.", s);
            continue
        end
        fprintf("[lowlight] %s: extracting %s ...\n", s, zips(z).name);
        unzip(fullfile(zips(z).folder, zips(z).name), opts.SrcRoot);
        if ~isfile(src)
            warning("make_lowlight_clips:layout", "%s: zip did not contain %s/vid.avi, skipped.", s, s);
            continue
        end
    end

    % random target illuminance for this subject (reproducible)
    num = str2double(extractAfter(s, "subject"));
    rs  = RandStream("mt19937ar", Seed=opts.Seed + num);
    lux = round(opts.LuxRange(1) + diff(opts.LuxRange) * rand(rs), 1);
    gain = lux / opts.OriginalLux;

    outDir = fullfile(opts.OutRoot, s);
    outVid = fullfile(outDir, "vid.avi");
    if isfile(outVid) && ~opts.Overwrite
        fprintf("[lowlight] %s: exists (%.1f lux), skipped. Use Overwrite=true to redo.\n", s, lux);
        continue
    end
    if ~isfolder(outDir), mkdir(outDir); end

    % attenuate every frame in linear light
    t0 = tic;
    vr = VideoReader(src);
    vw = VideoWriter(outVid, opts.Profile);
    vw.FrameRate = vr.FrameRate;
    open(vw);
    cleanup = onCleanup(@() close(vw));
    n = 0;  yIn = 0;  yOut = 0;
    while hasFrame(vr) && n < opts.MaxFrames
        F = readFrame(vr);
        G = im2uint8(lin2rgb(rgb2lin(im2double(F)) * gain));
        writeVideo(vw, G);
        n = n + 1;
        yIn  = yIn  + mean(F, "all");
        yOut = yOut + mean(G, "all");
    end
    clear cleanup                                       % closes the writer
    if isfile(gt), copyfile(gt, fullfile(outDir, "ground_truth.txt")); end

    rows(end+1, :) = {s, lux, gain, opts.OriginalLux, n, vr.FrameRate, ...
                      yIn / n, yOut / n, string(outVid)}; %#ok<AGROW>
    fprintf("[lowlight] %s: %.1f lux (gain %.3f), %d frames, mean pixel %.1f -> %.1f (%.0f s)\n", ...
        s, lux, gain, n, yIn / n, yOut / n, toc(t0));
end

L = cell2table(rows, VariableNames=["subject" "target_lux" "gain" "original_lux" ...
    "n_frames" "fps" "mean_pixel_in" "mean_pixel_out" "out_video"]);

% log: merge with earlier runs (one row per subject, newest wins)
logPath = fullfile(opts.OutRoot, "lowlight_log.csv");
if isfile(logPath) && height(L) > 0
    old = readtable(logPath, TextType="string");
    old = old(~ismember(old.subject, L.subject), :);
    L2 = [old; L];
elseif isfile(logPath)
    L2 = readtable(logPath, TextType="string");
else
    L2 = L;
end
if height(L2) > 0
    [~, o] = sort(str2double(extractAfter(L2.subject, "subject")));
    writetable(L2(o, :), logPath);
end
end
