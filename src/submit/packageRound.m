function info = packageRound(subject, opts)
%PACKAGEROUND Assemble the Round 1 submission folder (README 1.3) for one UBFC dev subject.
%   info = packageRound("subject1")
%   info = packageRound("subject1", MaxSeconds=30, OutRoot="D:\...\submission_demo\integration_30s")
%   <OutRoot>/Round1_<Condition>/
%     target_video.mp4             MPEG-4 (VideoWriter, Quality) re-encode of the source video,
%                                  first MaxSeconds; the pipeline runs on THIS file
%     reference_signal.csv         time_s, ppg, hr_bpm (UBFC ground_truth.txt, same time span)
%     processed_output.csv         time_s, hr_bpm, confidence, status (process_round1)
%     Round1_<Condition>_<Constraint>.mat   six variables (process_round1, -v7.3)
%     process_round1.m             entry point (copy; needs the repo's src/ and config/)
%   <OutRoot>/manifest_Round1_<Condition>.json   provenance: source, encoding, measured
%                                  bitrate, metrics, git hash (outside the required layout).
%   Large files go to OutRoot (default <work>\submission_demo), never into the repo.
%   Only UBFC dev subjects {1,3,5,8-16} are accepted.
arguments
    subject (1,1) string
    opts.OutRoot (1,1) string = ""
    opts.MaxSeconds (1,1) double = Inf
    opts.MaxFramesInMat (1,1) double = 300
    opts.Quality (1,1) double {mustBeInRange(opts.Quality, 0, 100)} = 95
    opts.Condition (1,1) string = "Baseline"
    opts.Constraint (1,1) string = "None"
    opts.Params struct = struct()
    opts.AllowTest (1,1) logical = false     % true only after the parameter freeze (submission build)
end
assert(opts.AllowTest || ismember(subject, "subject" + [1 3 5 8:16]), 'packageRound:dev', ...
    'Only UBFC dev subjects {1,3,5,8-16} may be packaged (AllowTest=true after the freeze).');
root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
cfg = getConfig(); cfg.subject = char(subject);
if opts.OutRoot == "", opts.OutRoot = fullfile(fileparts(cfg.datasetRoot), 'work', 'submission_demo'); end
pkg = fullfile(opts.OutRoot, "Round1_" + opts.Condition);
if ~isfolder(pkg), mkdir(pkg); end
[vp, gt] = loadUBFC(cfg);

% target_video.mp4
v = VideoReader(vp); fps = v.FrameRate;
nMax = min(v.NumFrames, floor(opts.MaxSeconds * fps + 1e-9));
tv = fullfile(pkg, 'target_video.mp4');
% Pulse-preserving H.264 (docs/API_NOTES.md, tools/probe_ffmpeg_mp4.m): crf 4, adaptive quantisation and
% deblocking off, tune grain, yuv420p (readable by MATLAB VideoReader on Windows). 30 s of dev subject 1:
% lossless MAE 1.43 -> this MP4 1.51 (100 % coverage) vs VideoWriter MPEG-4 29.9 (24 %).
ff = fullfile(getenv('LOCALAPPDATA'), 'Microsoft', 'WinGet', 'Packages', ...
    'Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe', 'ffmpeg-9.1-full_build', 'bin', 'ffmpeg.exe');
ffArgs = '-c:v libx264 -preset slow -crf 4 -pix_fmt yuv420p -tune grain -x264-params aq-mode=0:deblock=0,0:psy-rd=0,0';
if isfile(ff)
    k = min(v.NumFrames, nMax); tLast = (k - 1) / fps;
    cmd = sprintf('"%s" -y -hide_banner -loglevel error -i "%s" -frames:v %d %s -an "%s"', ff, vp, k, ffArgs, tv);
    [st, msg] = system(cmd);
    if st ~= 0, error('packageRound:ffmpeg', 'ffmpeg failed: %s', msg); end
    encoder = ['ffmpeg 9.1 ' ffArgs];
else                                     % fallback (destroys most of the pulse; see API_NOTES)
    w = VideoWriter(tv, 'MPEG-4'); w.FrameRate = fps; w.Quality = opts.Quality; open(w);
    k = 0; tLast = 0;
    while hasFrame(v) && k < nMax, k = k + 1; tLast = v.CurrentTime; writeVideo(w, readFrame(v)); end
    close(w);
    encoder = sprintf('VideoWriter MPEG-4 Quality %g (LOSSY - pulse largely lost)', opts.Quality);
end
clear v
r = VideoReader(tv); d = dir(tv);
vidInfo = struct('source', vp, 'frames', k, 'fps', fps, 'width', r.Width, 'height', r.Height, ...
    'duration_s', r.Duration, 'bytes', d.bytes, 'bitrate_kbps', 8 * d.bytes / r.Duration / 1000, ...
    'encoder', encoder);
clear r

% reference_signal.csv (same time span as the video, plus one frame)
sel = gt.t <= tLast + 1 / fps;
rf = fullfile(pkg, 'reference_signal.csv');
writetable(table(gt.t(sel), gt.ppg(sel), gt.hrSensor(sel), 'VariableNames', {'time_s', 'ppg', 'hr_bpm'}), rf);

% entry point + run
copyfile(fullfile(root, 'process_round1.m'), fullfile(pkg, 'process_round1.m'));
prm = opts.Params; prm.dataset = 'UBFC-rPPG DATASET_2 (dev split)'; prm.subject = char(subject);
prm.target_video = vidInfo;
out = process_round1(tv, 'Reference', rf, 'OutDir', char(pkg), 'Condition', char(opts.Condition), ...
    'Constraint', char(opts.Constraint), 'MaxFramesInMat', opts.MaxFramesInMat, 'Params', prm);

info = struct('package', char(pkg), 'subject', char(subject), 'target_video', vidInfo, ...
    'reference_csv', rf, 'processed_csv', out.csvFile, 'mat', out.matFile, ...
    'metrics', out.metrics, 'git_hash', gitInfo(), 'max_frames_in_mat', opts.MaxFramesInMat);
mf = fullfile(opts.OutRoot, "manifest_Round1_" + opts.Condition + ".json");
fid = fopen(mf, 'w'); fwrite(fid, jsonencode(info, 'PrettyPrint', true)); fclose(fid);
info.manifest = char(mf);
end
